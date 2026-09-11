"""Shared utilities: config loading, reproducibility, checkpoint I/O."""
import json
import os
import random
from pathlib import Path

import numpy as np
import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path: str):
    ensure_dir(str(Path(path).parent))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def save_checkpoint(model, optimizer, epoch: int, metrics: dict, path: str):
    import torch
    ensure_dir(str(Path(path).parent))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: str, model, optimizer=None, map_location="cpu", strict: bool = True):
    import torch
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("metrics", {})


def no_artist_leakage_split(items, artist_key_fn, val_frac=0.15, test_frac=0.15, seed=42):
    """Group-aware split so the same artist never appears in more than one split
    (spec Section 3, step 5: 'no artist leakage across train/test when possible')."""
    rng = random.Random(seed)
    artists = sorted(set(artist_key_fn(it) for it in items))
    rng.shuffle(artists)
    n = len(artists)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    test_artists = set(artists[:n_test])
    val_artists = set(artists[n_test:n_test + n_val])
    train_artists = set(artists[n_test + n_val:])

    splits = {"train": [], "val": [], "test": []}
    for it in items:
        a = artist_key_fn(it)
        if a in test_artists:
            splits["test"].append(it)
        elif a in val_artists:
            splits["val"].append(it)
        else:
            splits["train"].append(it)
    return splits
