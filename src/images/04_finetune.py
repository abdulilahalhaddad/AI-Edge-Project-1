"""Stage B (stretch): fine-tune ResNet-18 layer4 + head to predict the target from pixels.

PRD section 6. Trains on row years <= 2019 only, early-stops on within-metro Spearman on
the validation year, and writes img_score for validation and test rows only. img_score is
for blending (tabular prediction + w * img_score), never a tabular training column.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (IMAGENET_MEAN, IMAGENET_STD, REPO, load_config, path,  # noqa: E402
                    seed_everything, to_square_rgb)

from PIL import Image  # noqa: E402

SIZE = 224


class RowImages(Dataset):
    def __init__(self, paths: list[str], y: np.ndarray, train: bool):
        self.paths, self.y, self.train = paths, y.astype(np.float32), train

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = to_square_rgb(Image.open(self.paths[i])).resize((SIZE, SIZE), Image.BILINEAR)
        x = (np.asarray(img, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x.transpose(2, 0, 1).copy())
        if self.train:  # aerial views have no fixed orientation
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[2])
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[1])
            x = torch.rot90(x, int(torch.randint(0, 4, (1,)).item()), dims=[1, 2])
        return x, torch.tensor(self.y[i])


def within_metro_spearman(df: pd.DataFrame, pred: np.ndarray) -> float:
    df = df.assign(pred=pred)
    rhos = [spearmanr(g.pred, g.target).statistic
            for _, g in df.groupby(["city", "year"]) if len(g) >= 10]
    return float(np.mean(rhos))


def worker_init(wid: int) -> None:
    s = torch.initial_seed() % 2**32
    np.random.seed(s)


def build_model() -> torch.nn.Module:
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Linear(net.fc.in_features, 1)
    for name, p in net.named_parameters():
        p.requires_grad_(name.startswith("layer4") or name.startswith("fc"))
    return net


def set_train_mode(net: torch.nn.Module) -> None:
    """Train layer4 + head; frozen layers (and their BatchNorm stats) stay in eval mode."""
    net.eval()
    net.layer4.train()
    net.fc.train()


def predict(net, dl, device) -> np.ndarray:
    net.eval()
    out = []
    with torch.inference_mode():
        for x, _ in dl:
            out.append(net(x.to(device)).squeeze(1).float().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="subsample rows per split (smoke test)")
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config()
    sp = cfg["split"]
    seed = cfg["seed"]
    seed_everything(seed)
    torch.use_deterministic_algorithms(False)
    interim, processed, models, reports = (path(cfg, k) for k in ("interim", "processed", "models", "reports"))
    root = path(cfg, "image_root")

    feat = pd.read_parquet(processed / "image_features.parquet", columns=["zipcode", "year", "has_image", "img_year"])
    panel = pd.read_parquet(REPO / cfg["paths"]["panel"], columns=["zipcode", "year", "city", "target"])
    idx = pd.read_csv(interim / "image_index.csv").rename(columns={"image_year": "img_year"})
    df = (feat[feat.has_image == 1]
          .merge(panel, on=["zipcode", "year"], validate="one_to_one")
          .merge(idx[["zipcode", "img_year", "path"]], on=["zipcode", "img_year"], validate="many_to_one"))
    df = df.dropna(subset=["target"]).sort_values(["zipcode", "year"]).reset_index(drop=True)
    df["path"] = [str(root / p) for p in df.path]

    tr = df[df.year.between(*sp["train_years"])]
    va = df[df.year.between(*sp["valid_years"])]
    te = df[df.year.between(*sp["test_years"])]
    if args.limit:
        tr, va, te = (d.sample(min(args.limit, len(d)), random_state=seed) for d in (tr, va, te))
    assert tr.year.max() <= sp["train_years"][1], "training sample after last train year"

    sd = float(tr.target.std())
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    g = torch.Generator().manual_seed(seed)
    nw = int(cfg["embed"]["num_workers"])
    mk = lambda d, train: DataLoader(  # noqa: E731
        RowImages(d.path.tolist(), (d.target / sd).values, train), batch_size=64, shuffle=train,
        num_workers=nw, generator=g if train else None, worker_init_fn=worker_init,
        persistent_workers=True)
    dl_tr, dl_va = mk(tr, True), mk(va, False)

    net = build_model().to(device)
    opt = torch.optim.AdamW(
        [{"params": net.layer4.parameters(), "lr": 1e-4},
         {"params": net.fc.parameters(), "lr": 1e-3}],
        weight_decay=1e-4)
    loss_fn = torch.nn.HuberLoss()

    log, best_rho, best_state, best_epoch, bad = [], -np.inf, None, 0, 0
    print(f"train {len(tr):,} rows (row years {tr.year.min()}-{tr.year.max()}), "
          f"valid {len(va):,}, test {len(te):,}; target sd {sd:.4f}; device {device}")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        set_train_mode(net)
        tot, n = 0.0, 0
        for x, y in tqdm(dl_tr, desc=f"epoch {epoch}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(net(x).squeeze(1), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
        rho = within_metro_spearman(va, predict(net, dl_va, device))
        log.append({"epoch": epoch, "train_huber": tot / n, "valid_spearman": rho,
                    "seconds": round(time.time() - t0, 1)})
        print(f"epoch {epoch}: train Huber {tot / n:.4f}, valid within-metro Spearman {rho:.4f} "
              f"({time.time() - t0:.0f}s)")
        if rho > best_rho:
            best_rho, best_epoch, bad = rho, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 2:
                print("early stop: 2 epochs without improvement")
                break

    net.load_state_dict(best_state)
    suffix = "_smoke" if args.limit else ""
    torch.save(best_state, models / f"resnet18_finetuned{suffix}.pt")

    scores, metrics = [], {}
    for name, d in (("valid", va), ("test", te)):
        s = predict(net, mk(d, False), device)
        scores.append(d[["zipcode", "year"]].assign(img_score=s.astype(np.float32)))
        d = d.assign(pred=s * sd)
        tops = [g.nlargest(max(1, round(0.1 * len(g))), "pred").target.mean()
                for _, g in d.groupby(["city", "year"]) if len(g) >= 10]
        metrics[name] = {
            "rows": int(len(d)),
            "rmse": float(np.sqrt(np.mean((d.pred - d.target) ** 2))),
            "spearman_within_metro": within_metro_spearman(d, s),
            "top_decile_rel_growth": float(np.mean(tops)),
        }
    out = pd.concat(scores).sort_values(["zipcode", "year"]).reset_index(drop=True)
    out.to_parquet(processed / f"image_score{suffix}.parquet", index=False)

    summary = {"epochs": log, "best_epoch": best_epoch, "target_sd": sd, "train_rows": int(len(tr)),
               "train_max_row_year": int(tr.year.max()), "metrics": metrics,
               "panel": cfg["paths"]["panel"], "smoke_test": bool(args.limit)}
    (reports / f"stage_b{suffix}.json").write_text(json.dumps(summary, indent=2))
    print(f"best epoch {best_epoch}; valid rho {metrics['valid']['spearman_within_metro']:.4f}, "
          f"test rho {metrics['test']['spearman_within_metro']:.4f}")


if __name__ == "__main__":
    main()
