"""Step 2-3: frozen ResNet-50 embeddings (mask-weighted average of layer4) and colour stats.

Resumable: progress is kept in partial .npy files and a done-flag array, so a rerun
skips images already embedded. Use --limit N for a timing run and --reset to start over.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import COLOUR_COLS, ImageDataset, load_config, path, seed_everything  # noqa: E402

EMB_DIM = 2048


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def backbone() -> torch.nn.Module:
    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    body = torch.nn.Sequential(
        net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3, net.layer4
    )
    body.eval()
    for p in body.parameters():
        p.requires_grad_(False)
    return body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="embed at most N pending images")
    ap.add_argument("--reset", action="store_true", help="discard partial progress")
    args = ap.parse_args()

    cfg = load_config()
    ecfg = cfg["embed"]
    seed_everything(cfg["seed"])
    interim = path(cfg, "interim")
    root = path(cfg, "image_root")

    idx = pd.read_csv(interim / "image_index.csv")
    n = len(idx)
    size = int(ecfg["input_size"])

    emb_part = interim / "embeddings_resnet50.partial.npy"
    col_part = interim / "colour_stats.partial.npy"
    done_p = interim / "embed_done.npy"
    meta_p = interim / "embed_meta.json"

    meta = {"input_size": size, "n_images": n, "model": "resnet50/IMAGENET1K_V2"}
    if args.reset or not done_p.exists():
        for p in (emb_part, col_part, done_p):
            p.unlink(missing_ok=True)
        np.lib.format.open_memmap(emb_part, mode="w+", dtype=np.float16, shape=(n, EMB_DIM)).flush()
        np.lib.format.open_memmap(col_part, mode="w+", dtype=np.float32, shape=(n, len(COLOUR_COLS))).flush()
        np.save(done_p, np.zeros(n, dtype=bool))
        meta_p.write_text(json.dumps(meta, indent=2))
    else:
        old = json.loads(meta_p.read_text())
        assert old == meta, f"partial run used different settings {old}; rerun with --reset"

    emb = np.load(emb_part, mmap_mode="r+")
    col = np.load(col_part, mmap_mode="r+")
    done = np.load(done_p)

    pending = np.flatnonzero(~done).tolist()
    if args.limit is not None:
        pending = pending[: args.limit]
    if not pending:
        print("nothing pending")
    else:
        device = pick_device()
        net = backbone().to(device)
        paths = [str(root / p) for p in idx.path]
        ds = ImageDataset(paths, pending, size, int(ecfg["mask_threshold"]))
        dl = DataLoader(
            ds,
            batch_size=int(ecfg["batch_size"]),
            num_workers=int(ecfg["num_workers"]),
            shuffle=False,
            persistent_workers=False,
        )
        fmap_hw = size // 32
        t0 = time.time()
        n_empty = 0
        with torch.inference_mode():
            for b, (rows, x, m, stats) in enumerate(tqdm(dl, desc=f"embed@{size} on {device}")):
                fmap = net(x.to(device))  # B x 2048 x h x w
                # area-average the 512x512 mask down to the feature grid (on CPU: MPS
                # adaptive pooling needs divisible sizes)
                w = F.adaptive_avg_pool2d(m, (fmap_hw, fmap_hw)).to(device)
                wsum = w.sum(dim=(2, 3))  # B x 1
                pooled = (fmap * w).sum(dim=(2, 3)) / wsum.clamp_min(1e-8)
                empty = (wsum.squeeze(1) <= 0)
                if empty.any():  # no valid pixel: fall back to a plain average
                    n_empty += int(empty.sum())
                    pooled[empty] = fmap[empty].mean(dim=(2, 3))
                r = rows.numpy()
                emb[r] = pooled.float().cpu().numpy().astype(np.float16)
                col[r] = stats.numpy()
                done[r] = True
                if b % 20 == 0:
                    emb.flush(); col.flush(); np.save(done_p, done)
        emb.flush(); col.flush(); np.save(done_p, done)
        dt = time.time() - t0
        print(f"embedded {len(pending):,} images in {dt:.1f}s "
              f"({len(pending) / dt:.1f} img/s); empty-mask fallbacks: {n_empty}")
        if n_empty:
            meta["empty_mask_fallbacks"] = meta.get("empty_mask_fallbacks", 0) + n_empty
            meta_p.write_text(json.dumps(meta, indent=2))

    print(f"progress: {int(done.sum()):,} / {n:,}")
    if done.all():
        final = np.asarray(emb, dtype=np.float16)
        assert np.isfinite(final.astype(np.float32)).all(), "non-finite embedding"
        np.save(interim / "embeddings_resnet50.npy", final)
        stats = pd.DataFrame(np.asarray(col), columns=COLOUR_COLS)
        stats.insert(0, "image_year", idx.image_year.values)
        stats.insert(0, "zipcode", idx.zipcode.values)
        stats.to_csv(interim / "colour_stats.csv", index=False)
        print("wrote embeddings_resnet50.npy and colour_stats.csv")


if __name__ == "__main__":
    main()
