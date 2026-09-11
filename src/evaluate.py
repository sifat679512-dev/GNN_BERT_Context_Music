"""Evaluation metrics matching spec Section 6, plus plotting helpers used by
the deliverables (F1 vs epoch curves, t-SNE of fused embeddings, PR curves).
"""
import argparse
import os

import numpy as np

from src.utils import load_config, load_json, ensure_dir


def macro_micro_f1(preds: np.ndarray, targets: np.ndarray, eps: float = 1e-8):
    """Per-tag Precision/Recall/F1 -> Macro-F1 (mean over tags) and Micro-F1
    (pooled globally), per spec Section 6."""
    tp = ((preds == 1) & (targets == 1)).sum(axis=0)
    fp = ((preds == 1) & (targets == 0)).sum(axis=0)
    fn = ((preds == 0) & (targets == 1)).sum(axis=0)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1_per_tag = 2 * precision * recall / (precision + recall + eps)
    macro_f1 = float(np.mean(f1_per_tag))

    tp_sum, fp_sum, fn_sum = tp.sum(), fp.sum(), fn.sum()
    micro_precision = tp_sum / (tp_sum + fp_sum + eps)
    micro_recall = tp_sum / (tp_sum + fn_sum + eps)
    micro_f1 = float(2 * micro_precision * micro_recall / (micro_precision + micro_recall + eps))

    return macro_f1, micro_f1


def auc_pr(scores: np.ndarray, targets: np.ndarray) -> float:
    """Mean AUC-PR over tags (spec Section 6)."""
    from sklearn.metrics import average_precision_score
    aps = []
    for k in range(targets.shape[1]):
        if targets[:, k].sum() == 0:
            continue
        aps.append(average_precision_score(targets[:, k], scores[:, k]))
    return float(np.mean(aps)) if aps else float("nan")


def emotion_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """MAE and R^2 for DEAM valence/arousal regression (spec Section 6)."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) + 1e-8
    r2 = float(1 - ss_res / ss_tot)
    return {"MAE": mae, "R2": r2}


def graph_coherence_score(node_embeddings: np.ndarray, edge_index: np.ndarray, tau: float = 0.75):
    """S_graph = (1/|E|) sum_(i,j) 1[cos(h_i, h_j) > tau] (spec Section 6, optional analysis)."""
    if edge_index.shape[1] == 0:
        return float("nan")
    norm = np.linalg.norm(node_embeddings, axis=1, keepdims=True) + 1e-8
    normed = node_embeddings / norm
    src, dst = edge_index[0], edge_index[1]
    cos_sim = np.sum(normed[src] * normed[dst], axis=1)
    return float(np.mean(cos_sim > tau))


def plot_f1_curve(history_path: str, out_path: str):
    import matplotlib.pyplot as plt
    hist = load_json(history_path)
    plt.figure(figsize=(6, 4))
    plt.plot(hist["epoch"], hist["macro_f1"], label="Macro-F1")
    plt.plot(hist["epoch"], hist["micro_f1"], label="Micro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("Tag classification F1 vs. epoch")
    plt.legend()
    ensure_dir(os.path.dirname(out_path))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_tsne(embeddings: np.ndarray, labels: list, out_path: str, perplexity: int = 30, title: str = "t-SNE of fused embeddings z"):
    """t-SNE of z coloured by genre/mood (spec Task 3 deliverable)."""
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    n = embeddings.shape[0]
    perplexity = min(perplexity, max(5, n // 3))
    proj = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=42).fit_transform(embeddings)

    plt.figure(figsize=(6, 6))
    unique_labels = sorted(set(labels))
    for lab in unique_labels:
        mask = [l == lab for l in labels]
        plt.scatter(proj[mask, 0], proj[mask, 1], label=str(lab), s=10, alpha=0.7)
    plt.legend(markerscale=2, fontsize=8, loc="best")
    plt.title(title)
    ensure_dir(os.path.dirname(out_path))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pr_curve(scores: np.ndarray, targets: np.ndarray, tag_names: list, out_path: str, top_n: int = 5):
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 5))
    for k in range(min(top_n, targets.shape[1])):
        if targets[:, k].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(targets[:, k], scores[:, k])
        plt.plot(recall, precision, label=tag_names[k])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curves (sample tags)")
    plt.legend(fontsize=8)
    ensure_dir(os.path.dirname(out_path))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    history_path = f"results/task{args.task}_history.json"
    if os.path.exists(history_path) and args.task in (1, 3):
        plot_f1_curve(history_path, f"results/plots/task{args.task}_f1_curve.png")
        print(f"[evaluate] wrote results/plots/task{args.task}_f1_curve.png")
    else:
        print(f"[evaluate] {history_path} not found — run src.train first.")


if __name__ == "__main__":
    main()
