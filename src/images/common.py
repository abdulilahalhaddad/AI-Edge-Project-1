"""Shared helpers for the image pipeline: config, paths, and the image dataset."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[2]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

COLOUR_COLS = ["valid_frac", "mean_r", "mean_g", "mean_b", "green_idx", "texture"]


def load_config() -> dict:
    with open(REPO / "config.yaml") as f:
        return yaml.safe_load(f)


def path(cfg: dict, key: str) -> Path:
    p = REPO / cfg["paths"][key]
    if key in ("interim", "processed", "models", "reports"):
        p.mkdir(parents=True, exist_ok=True)
    return p


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def write_json(obj, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def colour_stats(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Six explainable colour statistics over valid (inside-ZIP) pixels.

    rgb: HxWx3 uint8. mask: HxW bool.
    """
    valid_frac = float(mask.mean())
    if not mask.any():
        return np.array([valid_frac] + [np.nan] * 5, dtype=np.float32)
    px = rgb[mask].astype(np.float32) / 255.0  # N x 3
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    green_idx = float(np.mean((2 * g - r - b) / (r + g + b + 1e-6)))
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.array(
        [valid_frac, r.mean(), g.mean(), b.mean(), green_idx, gray.std()],
        dtype=np.float32,
    )


FRAME = 512


def to_square_rgb(img: Image.Image) -> Image.Image:
    """Return a FRAME x FRAME RGB image.

    Most images are already 512x512 RGB. The DC metro's images are RGBA with the long side
    at 512 and transparent (near-black) pixels outside the ZIP; they are flattened to RGB
    and padded with black, centred, so the ground scale is not distorted.
    """
    rgb = img.convert("RGB")
    if rgb.size == (FRAME, FRAME):
        return rgb
    if max(rgb.size) != FRAME:
        rgb.thumbnail((FRAME, FRAME), Image.BILINEAR)
    canvas = Image.new("RGB", (FRAME, FRAME), (0, 0, 0))
    canvas.paste(rgb, ((FRAME - rgb.size[0]) // 2, (FRAME - rgb.size[1]) // 2))
    return canvas


class ImageDataset:
    """Returns (row, normalised CHW image, HxW valid mask at original size, colour stats)."""

    def __init__(self, paths: list[str], rows: list[int], input_size: int, threshold: int):
        self.paths = paths
        self.rows = rows
        self.input_size = input_size
        self.threshold = threshold

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        import torch

        row = self.rows[i]
        img = to_square_rgb(Image.open(self.paths[row]))
        rgb = np.asarray(img)
        mask = (rgb > self.threshold).any(axis=2)
        stats = colour_stats(rgb, mask)

        resized = img.resize((self.input_size, self.input_size), Image.BILINEAR)
        x = (np.asarray(resized, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x.transpose(2, 0, 1).copy())
        m = torch.from_numpy(mask.astype(np.float32))[None]  # 1 x H x W
        return row, x, m, torch.from_numpy(stats)
