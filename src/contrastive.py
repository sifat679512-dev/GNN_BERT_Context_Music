"""Task 4 (Advanced): cross-modal alignment via contrastive learning (spec Section 4.4).

ADAPTATION NOTE (see README): the spec's Task 4 is designed around MusicCaps
natural-language captions, which is not one of the three datasets used in this
project (MagnaTagATune, GTZAN, DEAM). This module is dataset-agnostic — it
consumes any (graph, text) pairs — and in `train.py --task 4` it is driven by
MagnaTagATune clips paired with `dataset.tags_to_pseudo_caption()` pseudo-
captions instead of real MusicCaps captions. Swap in real MusicCaps pairs with
zero code changes if/when that dataset becomes available.

InfoNCE:
    L_NCE = -log( exp(sim(g_i, t_i)/tau) / sum_j exp(sim(g_i, t_j)/tau) )
    sim(u, v) = u^T v / (||u|| ||v||)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bert_encoder import BertTextEncoder
from src.gnn_model import GraphEncoder


class ContrastiveGnnBert(nn.Module):
    """Dual encoder: graph tower + text tower, projected to a shared embedding
    space, L2-normalized, trained with InfoNCE (spec Algorithm 4)."""

    def __init__(self, graph_in_dim: int, embed_dim: int = 128,
                 bert_model_name: str = "distilbert-base-uncased",
                 gnn_hidden_dim: int = 128, gnn_layers: int = 3, gnn_dropout: float = 0.3,
                 gnn_type: str = "graphsage", fine_tune_bert: bool = True,
                 temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

        self.text_encoder = BertTextEncoder(bert_model_name, fine_tune_bert)
        self.graph_encoder = GraphEncoder(graph_in_dim, gnn_hidden_dim, gnn_layers,
                                           gnn_dropout, gnn_type)

        self.graph_proj = nn.Linear(gnn_hidden_dim, embed_dim)
        self.text_proj = nn.Linear(self.text_encoder.hidden_size, embed_dim)

    def encode_graph(self, graph_x, graph_edge_index, graph_batch=None):
        _, g = self.graph_encoder(graph_x, graph_edge_index, graph_batch)
        g = self.graph_proj(g)
        return F.normalize(g, dim=-1)  # g_i = Normalize(GNN(G_i))

    def encode_text(self, input_ids, attention_mask):
        t = self.text_encoder(input_ids, attention_mask)
        t = self.text_proj(t)
        return F.normalize(t, dim=-1)  # t_i = Normalize(BERT_CLS(caption_i))

    def forward(self, graph_x, graph_edge_index, input_ids, attention_mask, graph_batch=None):
        g = self.encode_graph(graph_x, graph_edge_index, graph_batch)
        t = self.encode_text(input_ids, attention_mask)
        return g, t


def info_nce_loss(g: torch.Tensor, t: torch.Tensor, temperature: float = 0.07):
    """Symmetric InfoNCE over a batch treated as in-batch negatives
    (spec Algorithm 4: S_ij = g_i^T t_j / tau, average of both directions)."""
    logits = g @ t.t() / temperature   # (B, B) similarity matrix S_ij
    labels = torch.arange(g.size(0), device=g.device)

    loss_g2t = F.cross_entropy(logits, labels)         # caption retrieval given audio graph
    loss_t2g = F.cross_entropy(logits.t(), labels)      # audio retrieval given caption
    loss = (loss_g2t + loss_t2g) / 2
    return loss, logits


@torch.no_grad()
def retrieval_recall_at_k(sim_matrix: torch.Tensor, ks=(1, 5, 10)):
    """Computes Caption->Audio and Audio->Caption R@K from a similarity matrix
    where sim_matrix[i, j] = sim(graph_i, text_j) and the correct match is the
    diagonal (spec: 'Retrieval metrics: Caption -> Audio R@1, R@5, R@10; Audio -> Caption R@K')."""
    n = sim_matrix.size(0)
    results = {}

    # Audio -> Caption: for each graph row, is the true caption in top-K columns?
    ranked_cols = sim_matrix.argsort(dim=1, descending=True)
    for k in ks:
        hits = sum(i in ranked_cols[i, :k].tolist() for i in range(n))
        results[f"audio_to_caption_R@{k}"] = hits / n

    # Caption -> Audio: for each text column, is the true graph in top-K rows?
    ranked_rows = sim_matrix.t().argsort(dim=1, descending=True)
    for k in ks:
        hits = sum(i in ranked_rows[i, :k].tolist() for i in range(n))
        results[f"caption_to_audio_R@{k}"] = hits / n

    return results
