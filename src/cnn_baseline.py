"""Baseline B2 (spec Section 8): CNN on mel-spectrogram, no graph, no text.
Used as the comparison point for Task 2's GNN model."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MelCNNBaseline(nn.Module):
    """Small 4-block CNN over a fixed-size log-mel spectrogram (spec B2)."""

    def __init__(self, num_classes: int, n_mels: int = 128, multilabel: bool = False):
        super().__init__()
        self.multilabel = multilabel
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, mel: torch.Tensor):
        # mel: (B, 1, n_mels, T)
        feat = self.conv(mel)
        return self.head(feat)


def cnn_loss(logits: torch.Tensor, targets: torch.Tensor, multilabel: bool) -> torch.Tensor:
    if multilabel:
        return F.binary_cross_entropy_with_logits(logits, targets)
    return F.cross_entropy(logits, targets)


class RandomTagBaseline:
    """Baseline B1 (spec Section 8): majority-class / random tag predictor."""

    def __init__(self, tag_prior: torch.Tensor):
        self.tag_prior = tag_prior  # per-tag base rate from the training set

    def predict(self, n: int):
        return self.tag_prior.unsqueeze(0).repeat(n, 1)
