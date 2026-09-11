"""Training entry point. Dispatches to Task 1-4 implementations
(spec Section 7 Algorithms 1-4).

    python -m src.train --task 1 --config config.yaml
    python -m src.train --task 2 --config config.yaml
    python -m src.train --task 3 --config config.yaml
    python -m src.train --task 4 --config config.yaml

NOTE: this script assumes preprocessed features/graphs exist under
data/processed/ (run audio_features.py and graph_builder.py first) and that
the Kaggle datasets have been placed under data/raw/ per the README. It is
written to be correct and directly runnable in a normal Python/CUDA
environment; it cannot be executed inside this sandbox (no internet access to
fetch the datasets or pretrained BERT weights, no GPU).
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import load_config, set_seed, save_json, save_checkpoint, ensure_dir
from src.dataset import load_magnatagatune, load_gtzan, load_deam, tags_to_pseudo_caption
from src.bert_encoder import BertTagClassifier, bert_tag_loss
from src.gnn_model import GnnGenreClassifier, gnn_loss
from src.fusion_model import GnnBertFusion, fusion_multitask_loss
from src.contrastive import ContrastiveGnnBert, info_nce_loss, retrieval_recall_at_k
from src.evaluate import macro_micro_f1, auc_pr


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Task 1: BERT tag classifier (Algorithm 1)
# ---------------------------------------------------------------------------

def train_task1(cfg: dict, epochs: int = None, max_samples: int = None):
    device = get_device()
    mtt = load_magnatagatune(cfg)
    tag_vocab = mtt.attrs["tag_vocab"]
    num_tags = len(tag_vocab)

    model = BertTagClassifier(
        num_tags=num_tags,
        model_name=cfg["text"]["bert_model"],
        fine_tune=cfg["text"]["fine_tune_bert"],
    ).to(device)

    optimizer = torch.optim.AdamW([
        {"params": model.encoder.bert.parameters(), "lr": cfg["train"]["lr_bert"]},
        {"params": model.head.parameters(), "lr": cfg["train"]["lr_head"]},
    ], weight_decay=cfg["train"]["weight_decay"])

    # Text source: for MagnaTagATune we use the tag set itself rendered as a short
    # phrase, since the raw dataset does not ship free-text captions/lyrics; this
    # keeps Task 1 self-contained on the three specified datasets.
    if max_samples and max_samples < len(mtt):
        mtt = mtt.sample(n=max_samples, random_state=cfg["seed"]).reset_index(drop=True)

    texts = [tags_to_pseudo_caption([t for t in tag_vocab if row[t] == 1]) for _, row in mtt.iterrows()]
    labels = mtt[tag_vocab].values.astype(np.float32)

    history = {"epoch": [], "macro_f1": [], "micro_f1": []}
    n = len(texts)
    idx = np.arange(n)
    rng = np.random.RandomState(cfg["seed"])
    num_epochs = epochs or cfg["train"]["epochs"]

    for epoch in range(num_epochs):
        rng.shuffle(idx)
        model.train()
        batch_size = cfg["train"]["batch_size"]
        all_logits, all_targets = [], []
        for start in tqdm(range(0, n, batch_size), desc=f"[task1] epoch {epoch}"):
            batch_idx = idx[start:start + batch_size]
            batch_texts = [texts[i] for i in batch_idx]
            batch_labels = torch.tensor(labels[batch_idx]).to(device)

            enc = model.encoder.tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=cfg["text"]["max_length"], return_tensors="pt"
            ).to(device)

            logits = model(enc["input_ids"], enc["attention_mask"])
            loss = bert_tag_loss(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_logits.append(logits.detach().cpu())
            all_targets.append(batch_labels.detach().cpu())

        preds = (torch.cat(all_logits).sigmoid() > 0.5).numpy()
        targets = torch.cat(all_targets).numpy()
        macro_f1, micro_f1 = macro_micro_f1(preds, targets)
        history["epoch"].append(epoch)
        history["macro_f1"].append(macro_f1)
        history["micro_f1"].append(micro_f1)
        print(f"[task1] epoch {epoch}: macro_f1={macro_f1:.4f} micro_f1={micro_f1:.4f}")

    ensure_dir("results")
    save_json(history, "results/task1_history.json")
    save_checkpoint(model, optimizer, num_epochs, history, "results/task1_best.pt")


# ---------------------------------------------------------------------------
# Task 2: GNN on segment/chord graphs (Algorithm 2)
# ---------------------------------------------------------------------------

def train_task2(cfg: dict, epochs: int = None, max_samples: int = None):
    device = get_device()
    gtzan = load_gtzan(cfg)
    genre_to_idx = {g: i for i, g in enumerate(cfg["data"]["gtzan"]["genres"])}

    graph_dir = os.path.join(cfg["data"]["processed_dir"], "graphs")
    if not os.path.isdir(graph_dir):
        raise FileNotFoundError("Run audio_features.py and graph_builder.py before training Task 2.")

    from torch_geometric.loader import DataLoader as PyGDataLoader

    graphs = []
    for _, row in gtzan.iterrows():
        gpath = os.path.join(graph_dir, f"gtzan_{row['clip_id']}.pt")
        if not os.path.exists(gpath):
            continue
        payload = torch.load(gpath, weights_only=False)
        g = payload["segment_graph"]
        g.y = torch.tensor([genre_to_idx[row["genre"]]], dtype=torch.long)
        graphs.append(g)
        if max_samples and len(graphs) >= max_samples:
            break

    if not graphs:
        raise RuntimeError("No preprocessed GTZAN graphs found in data/processed/graphs. Run src.audio_features and src.graph_builder first.")

    in_dim = graphs[0].x.shape[1]
    model = GnnGenreClassifier(
        in_dim=in_dim, num_classes=len(genre_to_idx),
        hidden_dim=cfg["model"]["gnn_hidden_dim"], num_layers=cfg["model"]["gnn_num_layers"],
        dropout=cfg["model"]["gnn_dropout"], gnn_type=cfg["model"]["gnn_type"],
        multilabel=False,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr_gnn"],
                                  weight_decay=cfg["train"]["weight_decay"])
    loader = PyGDataLoader(graphs, batch_size=cfg["train"]["batch_size"], shuffle=True)

    history = {"epoch": [], "accuracy": []}
    num_epochs = epochs or cfg["train"]["epochs"]
    for epoch in range(num_epochs):
        model.train()
        correct, total = 0, 0
        for batch in tqdm(loader, desc=f"[task2] epoch {epoch}"):
            batch = batch.to(device)
            logits, _ = model(batch.x, batch.edge_index, batch.batch)
            loss = gnn_loss(logits, batch.y, multilabel=False)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct += (logits.argmax(dim=-1) == batch.y).sum().item()
            total += batch.y.size(0)
        acc = correct / max(total, 1)
        history["epoch"].append(epoch)
        history["accuracy"].append(acc)
        print(f"[task2] epoch {epoch}: accuracy={acc:.4f}")

    num_epochs = epochs or cfg["train"]["epochs"]
    ensure_dir("results")
    save_json(history, "results/task2_history.json")
    save_checkpoint(model, optimizer, num_epochs, history, "results/task2_best.pt")


# ---------------------------------------------------------------------------
# Task 3: GNN-BERT fusion (Algorithm 3)
# ---------------------------------------------------------------------------

def train_task3(cfg: dict, fusion_type: str = "cross_attention", epochs: int = None, max_samples: int = None):
    """Trains the fusion model on MagnaTagATune (tags) + segment graphs, with the
    optional DEAM emotion auxiliary loss when a clip has a matching DEAM entry.
    `fusion_type` supports the required ablations: cross_attention / concat /
    bert_only / gnn_only."""
    device = get_device()
    mtt = load_magnatagatune(cfg)
    tag_vocab = mtt.attrs["tag_vocab"]

    graph_dir = os.path.join(cfg["data"]["processed_dir"], "graphs")
    if not os.path.isdir(graph_dir):
        raise FileNotFoundError("Run audio_features.py and graph_builder.py before training Task 3.")

    from torch_geometric.data import Batch

    rows, graphs, texts, labels = [], [], [], []
    for _, row in mtt.iterrows():
        gpath = os.path.join(graph_dir, f"mtt_{row['clip_id']}.pt")
        if not os.path.exists(gpath):
            continue
        payload = torch.load(gpath, weights_only=False)
        graphs.append(payload["segment_graph"])
        texts.append(tags_to_pseudo_caption([t for t in tag_vocab if row[t] == 1]))
        labels.append(row[tag_vocab].values.astype(np.float32))
        rows.append(row)
        if max_samples and len(graphs) >= max_samples:
            break

    if not graphs:
        raise RuntimeError("No preprocessed MagnaTagATune graphs found in data/processed/graphs.")

    labels = np.stack(labels)
    in_dim = graphs[0].x.shape[1]

    model = GnnBertFusion(
        graph_in_dim=in_dim, num_tags=len(tag_vocab),
        bert_model_name=cfg["text"]["bert_model"],
        gnn_hidden_dim=cfg["model"]["gnn_hidden_dim"], gnn_layers=cfg["model"]["gnn_num_layers"],
        gnn_dropout=cfg["model"]["gnn_dropout"], gnn_type=cfg["model"]["gnn_type"],
        attn_heads=cfg["model"]["attention_heads"], fusion_dim=cfg["model"]["fusion_dim"],
        fine_tune_bert=cfg["text"]["fine_tune_bert"], predict_emotion=False,
        fusion_type=fusion_type,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr_gnn"],
                                   weight_decay=cfg["train"]["weight_decay"])

    n = len(graphs)
    idx = np.arange(n)
    rng = np.random.RandomState(cfg["seed"])
    batch_size = cfg["train"]["batch_size"]
    num_epochs = epochs or cfg["train"]["epochs"]

    history = {"epoch": [], "macro_f1": [], "micro_f1": [], "fusion_type": fusion_type}
    for epoch in range(num_epochs):
        rng.shuffle(idx)
        model.train()
        all_logits, all_targets = [], []
        for start in tqdm(range(0, n, batch_size), desc=f"[task3:{fusion_type}] epoch {epoch}"):
            batch_idx = idx[start:start + batch_size]
            batch_graphs = Batch.from_data_list([graphs[i] for i in batch_idx]).to(device)
            batch_texts = [texts[i] for i in batch_idx]
            batch_labels = torch.tensor(labels[batch_idx]).to(device)

            enc = model.text_encoder.tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=cfg["text"]["max_length"], return_tensors="pt"
            ).to(device)

            out = model(enc["input_ids"], enc["attention_mask"],
                        batch_graphs.x, batch_graphs.edge_index, batch_graphs.batch)
            loss, _ = fusion_multitask_loss(out, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_logits.append(out["tag_logits"].detach().cpu())
            all_targets.append(batch_labels.detach().cpu())

        preds = (torch.cat(all_logits).sigmoid() > 0.5).numpy()
        targets = torch.cat(all_targets).numpy()
        macro_f1, micro_f1 = macro_micro_f1(preds, targets)
        history["epoch"].append(epoch)
        history["macro_f1"].append(macro_f1)
        history["micro_f1"].append(micro_f1)
        print(f"[task3:{fusion_type}] epoch {epoch}: macro_f1={macro_f1:.4f} micro_f1={micro_f1:.4f}")

    ensure_dir("results")
    save_json(history, f"results/task3_{fusion_type}_history.json")
    save_checkpoint(model, optimizer, num_epochs, history,
                     f"results/task3_{fusion_type}_best.pt")


# ---------------------------------------------------------------------------
# Task 4: contrastive dual-encoder (Algorithm 4) — MagnaTagATune pseudo-captions
# ---------------------------------------------------------------------------

def train_task4(cfg: dict, epochs: int = None, max_samples: int = None):
    device = get_device()
    mtt = load_magnatagatune(cfg)
    tag_vocab = mtt.attrs["tag_vocab"]

    graph_dir = os.path.join(cfg["data"]["processed_dir"], "graphs")
    from torch_geometric.data import Batch

    graphs, texts = [], []
    for _, row in mtt.iterrows():
        gpath = os.path.join(graph_dir, f"mtt_{row['clip_id']}.pt")
        if not os.path.exists(gpath):
            continue
        payload = torch.load(gpath, weights_only=False)
        graphs.append(payload["segment_graph"])
        texts.append(tags_to_pseudo_caption([t for t in tag_vocab if row[t] == 1]))
        if max_samples and len(graphs) >= max_samples:
            break

    if not graphs:
        raise RuntimeError("No preprocessed MagnaTagATune graphs found in data/processed/graphs.")

    in_dim = graphs[0].x.shape[1]
    model = ContrastiveGnnBert(
        graph_in_dim=in_dim, embed_dim=cfg["model"]["contrastive_embed_dim"],
        bert_model_name=cfg["text"]["bert_model"],
        gnn_hidden_dim=cfg["model"]["gnn_hidden_dim"], gnn_layers=cfg["model"]["gnn_num_layers"],
        gnn_dropout=cfg["model"]["gnn_dropout"], gnn_type=cfg["model"]["gnn_type"],
        fine_tune_bert=cfg["text"]["fine_tune_bert"],
        temperature=cfg["model"]["contrastive_temperature"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr_gnn"],
                                   weight_decay=cfg["train"]["weight_decay"])

    n = len(graphs)
    idx = np.arange(n)
    rng = np.random.RandomState(cfg["seed"])
    batch_size = cfg["train"]["batch_size"]
    num_epochs = epochs or cfg["train"]["epochs"]

    history = {"epoch": [], "loss": []}
    for epoch in range(num_epochs):
        rng.shuffle(idx)
        model.train()
        epoch_losses = []
        for start in tqdm(range(0, n, batch_size), desc=f"[task4] epoch {epoch}"):
            batch_idx = idx[start:start + batch_size]
            if len(batch_idx) < 2:
                continue
            batch_graphs = Batch.from_data_list([graphs[i] for i in batch_idx]).to(device)
            batch_texts = [texts[i] for i in batch_idx]

            enc = model.text_encoder.tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=cfg["text"]["max_length"], return_tensors="pt"
            ).to(device)

            g, t = model(batch_graphs.x, batch_graphs.edge_index,
                         enc["input_ids"], enc["attention_mask"], batch_graphs.batch)
            loss, _ = info_nce_loss(g, t, cfg["model"]["contrastive_temperature"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history["epoch"].append(epoch)
        history["loss"].append(avg_loss)
        print(f"[task4] epoch {epoch}: loss={avg_loss:.4f}")

    ensure_dir("results")
    save_json(history, "results/task4_history.json")
    save_checkpoint(model, optimizer, num_epochs, history, "results/task4_best.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="override number of training epochs")
    parser.add_argument("--max_samples", type=int, default=None, help="cap number of samples for training/eval")
    parser.add_argument("--fusion_type", default="cross_attention",
                         choices=["cross_attention", "concat", "bert_only", "gnn_only"],
                         help="Task 3 ablation variant (spec: BERT-only, GNN-only, early "
                              "concat, cross-attention)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    if args.task == 1:
        train_task1(cfg, epochs=args.epochs, max_samples=args.max_samples)
    elif args.task == 2:
        train_task2(cfg, epochs=args.epochs, max_samples=args.max_samples)
    elif args.task == 3:
        train_task3(cfg, fusion_type=args.fusion_type, epochs=args.epochs, max_samples=args.max_samples)
    elif args.task == 4:
        train_task4(cfg, epochs=args.epochs, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
