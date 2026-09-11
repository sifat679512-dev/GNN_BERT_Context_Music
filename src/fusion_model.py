"""Task 3 (Hard): GNN-BERT fusion for multi-context understanding (spec Section 4.3).

Cross-attention fusion:
    A = softmax(Q K^T / sqrt(d)),   Q = g W_Q,   K = H_text W_K
    z = CONCAT(g, A H_text),        y_hat = sigma(W z)

Multi-task loss:
    L = L_tags + alpha * ||v - v_hat||_2^2 + beta * ||a - a_hat||_2^2
(v, a) are DEAM valence/arousal targets when available; ablations also cover
BERT-only, GNN-only, and early-concat fusion (spec deliverables).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bert_encoder import BertTextEncoder
from src.gnn_model import GraphEncoder


class CrossAttentionFusion(nn.Module):
    """A = softmax(QK^T / sqrt(d)); z = CONCAT(g, A H_text)."""

    def __init__(self, graph_dim: int, text_dim: int, attn_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.attn_dim = attn_dim
        self.w_q = nn.Linear(graph_dim, attn_dim)
        self.w_k = nn.Linear(text_dim, attn_dim)
        self.w_v = nn.Linear(text_dim, attn_dim)
        self.out_dim = graph_dim + attn_dim

    def forward(self, g: torch.Tensor, h_text: torch.Tensor):
        """g: (B, graph_dim) graph-level embedding (one query token per sample).
        h_text: (B, L, text_dim) per-token BERT hidden states."""
        q = self.w_q(g).unsqueeze(1)             # (B, 1, attn_dim)
        k = self.w_k(h_text)                      # (B, L, attn_dim)
        v = self.w_v(h_text)                      # (B, L, attn_dim)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.attn_dim ** 0.5)  # (B, 1, L)
        attn = F.softmax(scores, dim=-1)
        attended = torch.matmul(attn, v).squeeze(1)  # (B, attn_dim)  == A H_text

        z = torch.cat([g, attended], dim=-1)
        return z, attn.squeeze(1)


class GnnBertFusion(nn.Module):
    """Full Task 3 model. `fusion_type` selects the ablation variant:
    'cross_attention' (recommended), 'concat' (early concat baseline),
    'bert_only', or 'gnn_only'."""

    def __init__(self, graph_in_dim: int, num_tags: int,
                 bert_model_name: str = "distilbert-base-uncased",
                 gnn_hidden_dim: int = 128, gnn_layers: int = 3, gnn_dropout: float = 0.3,
                 gnn_type: str = "graphsage", attn_heads: int = 4, fusion_dim: int = 256,
                 fine_tune_bert: bool = True, predict_emotion: bool = False,
                 fusion_type: str = "cross_attention"):
        super().__init__()
        self.fusion_type = fusion_type
        self.predict_emotion = predict_emotion

        self.text_encoder = BertTextEncoder(bert_model_name, fine_tune_bert)
        self.graph_encoder = GraphEncoder(graph_in_dim, gnn_hidden_dim, gnn_layers,
                                           gnn_dropout, gnn_type)

        text_dim = self.text_encoder.hidden_size
        graph_dim = gnn_hidden_dim

        if fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion(graph_dim, text_dim, fusion_dim, attn_heads)
            z_dim = self.fusion.out_dim
        elif fusion_type == "concat":
            self.fusion = None
            z_dim = graph_dim + text_dim
        elif fusion_type == "bert_only":
            self.fusion = None
            z_dim = text_dim
        elif fusion_type == "gnn_only":
            self.fusion = None
            z_dim = graph_dim
        else:
            raise ValueError(f"unknown fusion_type {fusion_type}")

        self.tag_head = nn.Linear(z_dim, num_tags)
        if predict_emotion:
            self.emotion_head = nn.Linear(z_dim, 2)  # (valence, arousal)

    def forward(self, input_ids, attention_mask, graph_x, graph_edge_index, graph_batch=None):
        bert_out = self.text_encoder.bert(input_ids=input_ids, attention_mask=attention_mask)
        h_text = bert_out.last_hidden_state           # (B, L, text_dim)
        t = h_text[:, 0, :]                            # CLS

        node_emb, g = self.graph_encoder(graph_x, graph_edge_index, graph_batch)  # g: (B, graph_dim)

        attn_weights = None
        if self.fusion_type == "cross_attention":
            z, attn_weights = self.fusion(g, h_text)
        elif self.fusion_type == "concat":
            z = torch.cat([g, t], dim=-1)
        elif self.fusion_type == "bert_only":
            z = t
        else:  # gnn_only
            z = g

        tag_logits = self.tag_head(z)
        emotion_pred = self.emotion_head(z) if self.predict_emotion else None
        return {
            "tag_logits": tag_logits,
            "emotion_pred": emotion_pred,
            "z": z,
            "graph_embedding": g,
            "text_cls": t,
            "attn_weights": attn_weights,
        }


def fusion_multitask_loss(outputs: dict, tag_targets: torch.Tensor,
                           emotion_targets: torch.Tensor = None,
                           alpha: float = 1.0, beta: float = 1.0):
    """L = L_tags + alpha * ||v - v_hat||^2 + beta * ||a - a_hat||^2 (spec Eq., Section 4.3)."""
    l_tags = F.binary_cross_entropy_with_logits(outputs["tag_logits"], tag_targets)
    total = l_tags
    parts = {"tags": l_tags.item()}

    if emotion_targets is not None and outputs["emotion_pred"] is not None:
        v_hat, a_hat = outputs["emotion_pred"][:, 0], outputs["emotion_pred"][:, 1]
        v, a = emotion_targets[:, 0], emotion_targets[:, 1]
        l_v = alpha * F.mse_loss(v_hat, v)
        l_a = beta * F.mse_loss(a_hat, a)
        total = total + l_v + l_a
        parts["valence_mse"] = l_v.item()
        parts["arousal_mse"] = l_a.item()

    return total, parts
