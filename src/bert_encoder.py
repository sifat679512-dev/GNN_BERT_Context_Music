"""Task 1 (Easy): BERT-based multi-label classifier on textual music context
(tags / captions / lyrics), no graph structure (spec Section 4.1).

    t = BERT_CLS(X_text),  y_hat_k = sigma(w_k^T t + b_k)
    L_BERT = -(1/K) * sum_k [ y_k log y_hat_k + (1-y_k) log(1-y_hat_k) ]
"""
import torch
import torch.nn as nn


class BertTextEncoder(nn.Module):
    """Wraps a HuggingFace BERT/DistilBERT model and returns the CLS embedding."""

    def __init__(self, model_name: str = "distilbert-base-uncased", fine_tune: bool = True):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        if not fine_tune:
            for p in self.bert.parameters():
                p.requires_grad = False

    def tokenize(self, texts: list, max_length: int = 128):
        return self.tokenizer(
            texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token == first token of last_hidden_state; works for BERT and DistilBERT alike
        cls = out.last_hidden_state[:, 0, :]
        return cls  # (batch, hidden_size)


class BertTagClassifier(nn.Module):
    """Task 1 model: BERT_CLS -> linear head -> per-tag sigmoid."""

    def __init__(self, num_tags: int, model_name: str = "distilbert-base-uncased",
                 fine_tune: bool = True, dropout: float = 0.2):
        super().__init__()
        self.encoder = BertTextEncoder(model_name, fine_tune)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.hidden_size, num_tags)

    def forward(self, input_ids, attention_mask):
        t = self.encoder(input_ids, attention_mask)
        logits = self.head(self.dropout(t))
        return logits  # BCEWithLogitsLoss expects raw logits, not sigmoid output

    def embed(self, input_ids, attention_mask):
        return self.encoder(input_ids, attention_mask)


def bert_tag_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """L_BERT = mean over tags of per-tag BCE (spec Eq. in Section 4.1)."""
    return nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
