import torch
import torch.nn as nn

class SimpleGraphEncoder(nn.Module):
    """
    A simple Graph Neural Network encoder that computes a graph-level embedding from node features and adjacency.
    Uses one round of message passing and pooling.
    """
    def __init__(self, input_dim, hidden_dim=128):
        super(SimpleGraphEncoder, self).__init__()
        # Linear transform for input features
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, node_feats, adj):
        # node_feats: tensor (N, input_dim), adj: tensor (N, N) adjacency (0/1)
        h = torch.relu(self.lin1(node_feats))  # (N, hidden_dim)
        # One round of message passing: aggregate neighbor features
        if adj is not None:
            # adj is (N, N), do adjacency matrix multiplication
            neigh_sum = adj @ h  # (N, hidden_dim)
            h = h + neigh_sum    # simple aggregation: self + neighbors
        h = torch.relu(self.lin2(h))
        # Graph-level embedding: sum pooling of node features
        g_emb = torch.sum(h, dim=0)  # (hidden_dim,)
        return g_emb
