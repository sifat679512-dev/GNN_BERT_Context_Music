"""Task 2 (Medium): GraphSAGE or GAT encoder on segment/chord graphs, audio-only
node features, mean-pool readout, tag/genre prediction (spec Section 4.2).

GraphSAGE update:
    h_i^(l+1) = sigma( W^(l) . CONCAT(h_i^(l), MEAN_{j in N(i)} h_j^(l)) )
Readout:
    g = (1/|V|) sum_i h_i^(L),   y_hat = sigma(W g + b)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_conv_stack(gnn_type: str, in_dim: int, hidden_dim: int, num_layers: int):
    from torch_geometric.nn import SAGEConv, GATConv

    convs = nn.ModuleList()
    for layer in range(num_layers):
        d_in = in_dim if layer == 0 else hidden_dim
        if gnn_type == "gat":
            heads = 4 if layer < num_layers - 1 else 1
            concat = layer < num_layers - 1
            convs.append(GATConv(d_in, hidden_dim // heads if concat else hidden_dim,
                                  heads=heads, concat=concat, dropout=0.1))
        else:  # graphsage (default per spec Section 4.2)
            convs.append(SAGEConv(d_in, hidden_dim))
    return convs


class GraphEncoder(nn.Module):
    """L-layer GraphSAGE/GAT encoder + mean-pool graph-level readout."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 3,
                 dropout: float = 0.3, gnn_type: str = "graphsage"):
        super().__init__()
        self.gnn_type = gnn_type
        self.convs = _build_conv_stack(gnn_type, in_dim, hidden_dim, num_layers)
        self.dropout = dropout
        self.out_dim = hidden_dim

    def forward(self, x, edge_index, batch=None):
        from torch_geometric.nn import global_mean_pool

        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        node_embeddings = h
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        graph_embedding = global_mean_pool(h, batch)  # g = mean_i h_i^(L)
        return node_embeddings, graph_embedding


class GnnGenreClassifier(nn.Module):
    """Task 2 model: GraphEncoder -> linear head -> per-class/tag sigmoid or softmax."""

    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 128,
                 num_layers: int = 3, dropout: float = 0.3, gnn_type: str = "graphsage",
                 multilabel: bool = False):
        super().__init__()
        self.encoder = GraphEncoder(in_dim, hidden_dim, num_layers, dropout, gnn_type)
        self.head = nn.Linear(hidden_dim, num_classes)
        self.multilabel = multilabel

    def forward(self, x, edge_index, batch=None):
        _, g = self.encoder(x, edge_index, batch)
        logits = self.head(g)
        return logits, g


def gnn_loss(logits: torch.Tensor, targets: torch.Tensor, multilabel: bool) -> torch.Tensor:
    if multilabel:
        return F.binary_cross_entropy_with_logits(logits, targets)
    return F.cross_entropy(logits, targets)
