"""Music structure graph construction (spec Section 3, step 3).

Two graph types, as specified:
  - Chord-transition graph: nodes = unique chords observed in the clip,
    edges = observed transitions weighted by count.
  - Segment graph: nodes = time segments, edges = temporal adjacency +
    cosine similarity of MFCC/chroma segment features > tau.

Both are exported as `torch_geometric.data.Data` objects and cached to
`data/processed/graphs/<clip_id>.pt`.
"""
import argparse
import os
from collections import Counter
from pathlib import Path

import numpy as np

from src.utils import load_config, ensure_dir


def build_chord_transition_graph(chord_sequence: list, chord_vocab: list):
    """Nodes = unique chords in `chord_vocab` (fixed size so graphs are
    batchable); edges = observed transitions weighted by count."""
    import torch
    from torch_geometric.data import Data

    vocab_index = {c: i for i, c in enumerate(chord_vocab)}
    num_nodes = len(chord_vocab)

    # node features: one-hot chord identity + normalized occurrence frequency
    occurrence = Counter(chord_sequence)
    total = max(sum(occurrence.values()), 1)
    x = np.eye(num_nodes, dtype=np.float32)
    freq = np.array([occurrence.get(c, 0) / total for c in chord_vocab], dtype=np.float32)
    x = np.concatenate([x, freq[:, None]], axis=1)

    edge_counter = Counter()
    for a, b in zip(chord_sequence[:-1], chord_sequence[1:]):
        if a in vocab_index and b in vocab_index:
            edge_counter[(vocab_index[a], vocab_index[b])] += 1

    if edge_counter:
        edges = np.array(list(edge_counter.keys()), dtype=np.int64).T
        weights = np.array(list(edge_counter.values()), dtype=np.float32)
        weights = weights / weights.max()
    else:
        edges = np.zeros((2, 0), dtype=np.int64)
        weights = np.zeros((0,), dtype=np.float32)

    return Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edges, dtype=torch.long),
        edge_attr=torch.tensor(weights, dtype=torch.float).unsqueeze(-1),
    )


def cosine_sim_matrix(feats: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
    normed = feats / norm
    return normed @ normed.T


def build_segment_graph(segment_features: np.ndarray, tau: float):
    """Nodes = time segments (features from audio_features.segment_features);
    edges = temporal adjacency + cosine similarity > tau (spec Section 3, step 3)."""
    import torch
    from torch_geometric.data import Data

    n = segment_features.shape[0]
    if n == 0:
        return Data(
            x=torch.zeros((1, segment_features.shape[-1] if segment_features.ndim == 2 else 1)),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 1)),
        )

    sim = cosine_sim_matrix(segment_features)
    src, dst, weights = [], [], []

    # temporal adjacency (always connect consecutive segments)
    for i in range(n - 1):
        for (a, b) in [(i, i + 1), (i + 1, i)]:
            src.append(a)
            dst.append(b)
            weights.append(float(sim[a, b]))

    # similarity edges beyond adjacency
    for i in range(n):
        for j in range(n):
            if i == j or abs(i - j) == 1:
                continue
            if sim[i, j] > tau:
                src.append(i)
                dst.append(j)
                weights.append(float(sim[i, j]))

    edge_index = np.array([src, dst], dtype=np.int64) if src else np.zeros((2, 0), dtype=np.int64)
    edge_weight = np.array(weights, dtype=np.float32) if weights else np.zeros((0,), dtype=np.float32)

    return Data(
        x=torch.tensor(segment_features, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_weight, dtype=torch.float).unsqueeze(-1),
    )


def build_and_cache_graph(clip_id: str, feat_npz_path: str, cfg: dict, out_dir: str):
    import torch

    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{clip_id}.pt")
    if os.path.exists(out_path):
        return out_path

    data = np.load(feat_npz_path, allow_pickle=True)
    chords = list(data["segment_chords"])
    seg_feats = data["segment_features"]

    chord_graph = build_chord_transition_graph(chords, cfg["graph"]["chord_vocab"])
    segment_graph = build_segment_graph(seg_feats, cfg["graph"]["segment_similarity_threshold"])

    torch.save({"chord_graph": chord_graph, "segment_graph": segment_graph}, out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    feat_dir = os.path.join(cfg["data"]["processed_dir"], "audio_features")
    out_dir = os.path.join(cfg["data"]["processed_dir"], "graphs")

    if not os.path.isdir(feat_dir):
        print(f"[graph_builder] {feat_dir} not found — run `python -m src.audio_features` first.")
        return

    n = 0
    for fname in sorted(os.listdir(feat_dir)):
        if not fname.endswith(".npz"):
            continue
        clip_id = fname[:-4]
        try:
            build_and_cache_graph(clip_id, os.path.join(feat_dir, fname), cfg, out_dir)
            n += 1
            if args.limit and n >= args.limit:
                break
        except Exception as e:
            print(f"[graph_builder] skipping {clip_id}: {e}")
    print(f"[graph_builder] built {n} graphs -> {out_dir}")


if __name__ == "__main__":
    main()
