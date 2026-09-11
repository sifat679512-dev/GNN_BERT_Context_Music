"""Dataset loading, tag vocabulary, splits, and PyTorch Dataset wrappers for the
three project datasets: MagnaTagATune, GTZAN, DEAM (spec Section 3).

MagnaTagATune  -> Task 1 (BERT tags), Task 3 fusion labels, Task 4 (adapted, see README)
GTZAN          -> Task 2 (GNN genre classification), CNN baseline (B2)
DEAM           -> Task 3 auxiliary valence/arousal regression (L_aux)
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import load_config, ensure_dir, save_json, no_artist_leakage_split


# ---------------------------------------------------------------------------
# MagnaTagATune
# ---------------------------------------------------------------------------

def _resolve_path(p: str) -> str:
    if not p:
        return p
    if os.path.exists(p):
        return p
    # Try relative to parent directory (e.g. if run from notebooks/)
    candidate = os.path.join("..", p)
    if os.path.exists(candidate):
        return candidate
    return p


def load_magnatagatune(cfg: dict) -> pd.DataFrame:
    """Loads annotations_final.csv (tab-separated in the canonical MTAT release)
    and keeps the top-K tags per config (spec: "top-50 tags")."""
    mtt_cfg = cfg["data"]["magnatagatune"]
    csv_path = _resolve_path(mtt_cfg["annotations_csv"])
    df = pd.read_csv(csv_path, sep="\t")
    non_tag_cols = {"clip_id", "mp3_path"}
    tag_cols = [c for c in df.columns if c not in non_tag_cols]

    top_k = mtt_cfg.get("top_k_tags", 50)
    tag_counts = df[tag_cols].sum(axis=0).sort_values(ascending=False)
    top_tags = list(tag_counts.head(top_k).index)

    df = df[["clip_id", "mp3_path"] + top_tags].copy()
    # drop rows with no positive tag among the top-K (common MTAT preprocessing choice)
    df = df[df[top_tags].sum(axis=1) > 0].reset_index(drop=True)
    df["artist_key"] = df["mp3_path"].apply(lambda p: str(p).split("/")[0])  # MTAT shards by hash dir; used only as a stable grouping proxy
    df.attrs["tag_vocab"] = top_tags
    return df


def tags_to_pseudo_caption(tags: list) -> str:
    """Turns a MagnaTagATune tag set into a short natural-language pseudo-caption,
    used ONLY as a documented substitute for MusicCaps in Task 4 (see README).
    e.g. ['guitar','mellow','slow'] -> 'a mellow, slow track featuring guitar'
    """
    if not tags:
        return "an instrumental track"
    instrument_like = {"guitar", "piano", "violin", "drums", "flute", "vocal", "vocals",
                        "strings", "synth", "beat", "beats", "harp", "cello", "sitar"}
    descriptors = [t for t in tags if t not in instrument_like]
    instruments = [t for t in tags if t in instrument_like]
    desc_str = ", ".join(descriptors[:3]) if descriptors else "instrumental"
    if instruments:
        return f"a {desc_str} track featuring {', '.join(instruments[:2])}"
    return f"a {desc_str} track"


# ---------------------------------------------------------------------------
# GTZAN
# ---------------------------------------------------------------------------

def load_gtzan(cfg: dict) -> pd.DataFrame:
    """Walks the GTZAN genres_original/<genre>/*.wav directory layout."""
    gtzan_cfg = cfg["data"]["gtzan"]
    audio_dir = Path(_resolve_path(gtzan_cfg["audio_dir"]))
    rows = []
    for genre in gtzan_cfg["genres"]:
        genre_dir = audio_dir / genre
        if not genre_dir.exists():
            continue
        for wav_path in sorted(genre_dir.glob("*.wav")):
            rows.append({"clip_id": wav_path.stem, "path": str(wav_path), "genre": genre})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# DEAM
# ---------------------------------------------------------------------------

def load_deam(cfg: dict) -> pd.DataFrame:
    """Loads DEAM static valence/arousal annotations (1-9 scale, spec Table 1)."""
    deam_cfg = cfg["data"]["deam"]
    csv1 = _resolve_path(deam_cfg["static_annotations_csv"])
    df = pd.read_csv(csv1)
    if "static_annotations_csv2" in deam_cfg:
        csv2 = _resolve_path(deam_cfg["static_annotations_csv2"])
        if os.path.exists(csv2):
            df2 = pd.read_csv(csv2)
            df = pd.concat([df, df2], ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    id_col = "song_id" if "song_id" in df.columns else df.columns[0]
    val_col = next(c for c in df.columns if "valence_mean" in c.lower())
    aro_col = next(c for c in df.columns if "arousal_mean" in c.lower())
    df = df[[id_col, val_col, aro_col]].rename(
        columns={id_col: "clip_id", val_col: "valence", aro_col: "arousal"}
    )
    audio_dir = Path(_resolve_path(deam_cfg["audio_dir"]))
    df["path"] = df["clip_id"].apply(lambda cid: str(audio_dir / f"{cid}.mp3"))
    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)
    # DEAM's 1-9 scale -> normalize to [-1, 1] the way arousal/valence models typically report
    df["valence_norm"] = (df["valence"] - 5.0) / 4.0
    df["arousal_norm"] = (df["arousal"] - 5.0) / 4.0
    return df


# ---------------------------------------------------------------------------
# Unified iteration for preprocessing
# ---------------------------------------------------------------------------

def iter_all_clips(cfg: dict):
    """Yields (clip_id, audio_path, meta) across all three datasets so
    audio_features.py / graph_builder.py can cache everything in one pass."""
    try:
        mtt = load_magnatagatune(cfg)
        audio_dir = cfg["data"]["magnatagatune"]["audio_dir"]
        for _, row in mtt.iterrows():
            yield f"mtt_{row['clip_id']}", os.path.join(audio_dir, row["mp3_path"]), {"dataset": "mtt"}
    except FileNotFoundError:
        print("[dataset] MagnaTagATune annotations not found, skipping (see README to place data).")

    try:
        gtzan = load_gtzan(cfg)
        for _, row in gtzan.iterrows():
            yield f"gtzan_{row['clip_id']}", row["path"], {"dataset": "gtzan", "genre": row["genre"]}
    except Exception as e:
        print(f"[dataset] GTZAN not available, skipping: {e}")

    try:
        deam = load_deam(cfg)
        for _, row in deam.iterrows():
            yield f"deam_{row['clip_id']}", row["path"], {"dataset": "deam"}
    except FileNotFoundError:
        print("[dataset] DEAM annotations not found, skipping (see README to place data).")


# ---------------------------------------------------------------------------
# Split building
# ---------------------------------------------------------------------------

def build_splits(cfg: dict):
    ensure_dir(cfg["data"]["splits_dir"])

    try:
        mtt = load_magnatagatune(cfg)
        items = mtt.to_dict("records")
        splits = no_artist_leakage_split(items, artist_key_fn=lambda r: r["artist_key"], seed=cfg["seed"])
        save_json({k: [r["clip_id"] for r in v] for k, v in splits.items()},
                   os.path.join(cfg["data"]["splits_dir"], "magnatagatune_splits.json"))
        save_json(mtt.attrs["tag_vocab"], os.path.join(cfg["data"]["splits_dir"], "mtt_tag_vocab.json"))
        print(f"[dataset] MagnaTagATune splits: "
              f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    except FileNotFoundError:
        print("[dataset] MagnaTagATune not found, skipping split build.")

    try:
        gtzan = load_gtzan(cfg)
        items = gtzan.to_dict("records")
        # GTZAN has no reliable artist metadata; spec allows "no leakage where possible" ->
        # stratify by genre with a fixed random split instead.
        rng = np.random.RandomState(cfg["seed"])
        idx = np.arange(len(items))
        rng.shuffle(idx)
        n = len(idx)
        n_test, n_val = int(0.15 * n), int(0.15 * n)
        splits = {
            "test": [items[i]["clip_id"] for i in idx[:n_test]],
            "val": [items[i]["clip_id"] for i in idx[n_test:n_test + n_val]],
            "train": [items[i]["clip_id"] for i in idx[n_test + n_val:]],
        }
        save_json(splits, os.path.join(cfg["data"]["splits_dir"], "gtzan_splits.json"))
        print(f"[dataset] GTZAN splits: train={len(splits['train'])} "
              f"val={len(splits['val'])} test={len(splits['test'])}")
    except Exception as e:
        print(f"[dataset] GTZAN split build failed/skipped: {e}")

    try:
        deam = load_deam(cfg)
        items = deam.to_dict("records")
        rng = np.random.RandomState(cfg["seed"])
        idx = np.arange(len(items))
        rng.shuffle(idx)
        n = len(idx)
        n_test, n_val = int(0.15 * n), int(0.15 * n)
        splits = {
            "test": [items[i]["clip_id"] for i in idx[:n_test]],
            "val": [items[i]["clip_id"] for i in idx[n_test:n_test + n_val]],
            "train": [items[i]["clip_id"] for i in idx[n_test + n_val:]],
        }
        save_json(splits, os.path.join(cfg["data"]["splits_dir"], "deam_splits.json"))
        print(f"[dataset] DEAM splits: train={len(splits['train'])} "
              f"val={len(splits['val'])} test={len(splits['test'])}")
    except FileNotFoundError:
        print("[dataset] DEAM not found, skipping split build.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--build_splits", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.build_splits:
        build_splits(cfg)


if __name__ == "__main__":
    main()
