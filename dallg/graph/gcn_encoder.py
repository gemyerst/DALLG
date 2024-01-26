
import torch
import torch.nn as nn
from .graph_transformer import GraphTransformer


class GCNEncoder(nn.Module):
    def __init__(self, n_node_features: int, n_out_features: int, n_layers: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=n_node_features, out_features=n_out_features),
            nn.ReLU())
        self.gat = GraphTransformer(
            dim=n_out_features,
            depth=n_layers,
            dim_head=16,
            heads=4)

    def forward(self, node_features: torch.FloatTensor, adj_matrix: torch.FloatTensor):
        """
        Args:
            node_features: Tensor of node features, having shape (b, n_nodes, n_node_features)
            adj_matrix: Batch of adjacency matrices of the graph. If there is an edge from i to j,
                         adj_matrix[b,i,j]=1 else 0. Supports directed edges by non-symmetric matrices.
                         Assumes to already have added the identity connections.
                         Shape is (b, n_nodes, n_nodes)

        Returns:
            torch.FloatTensor
        """
        node_features = self.fc(node_features)
        node_logits, _ = self.gat(nodes=node_features, edges=None, adj_mat=adj_matrix, mask=None)
        return node_logits
