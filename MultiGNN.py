import torch
from torch_geometric.utils import dropout_adj
import torch.nn as nn
from typing import List, Dict, Union, Callable
from torch_geometric.nn import global_mean_pool


class MultiHeadGNN(nn.Module):

    def __init__(self, hidden_dim: int, output_dim: int, mlp_dim: int, dropout: int, heads: List[Dict[str, Union[Callable, Dict]]]):

        super(MultiHeadGNN, self).__init__()
        n_head = len(heads)
        head_output_dim = hidden_dim//n_head
        heads_list = []
        # heads = [{'layer': GCN, 'params': ?}, {'layer': GIN, 'params': ?}, ...]
        for layer in heads:
            # layers should have in_channels and out_channels
            heads_list.append(layer['layer'](
                hidden_dim, head_output_dim, **layer['params']))
        self.GNN_heads = nn.ModuleList(heads_list)
        self.post_GNN_heads = nn.Sequential(nn.BatchNorm1d(head_output_dim*n_head),
                                            nn.Linear(
                                                head_output_dim*n_head, mlp_dim),
                                            nn.Dropout(p=dropout),
                                            nn.GELU(),
                                            nn.Linear(mlp_dim, output_dim))

    def forward(self, x, adj_t):
        identity = x
        heads_output = []
        for head in self.GNN_heads:
            # edge dropout
            adj_t = dropout_adj(adj_t, p=0.2, training=self.train)[
                0]  # select edge_index
            heads_output.append(head(x, adj_t))
        x = torch.cat(heads_output, 1)
        x = self.post_GNN_heads(x)
        return x + identity


class GNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, mlp_dim: int, dropout: int, num_layers: int, heads: List):

        super(GNN, self).__init__()
        # pre procesing
        self.pre_procesing = nn.Sequential(nn.Linear(input_dim, mlp_dim),
                                           nn.Dropout(p=dropout),
                                           nn.BatchNorm1d(mlp_dim),
                                           nn.ReLU(),
                                           nn.Linear(mlp_dim, hidden_dim))

        # GGN layers
        GNN_layers = []
        for i in range(num_layers):
            GNN_layers.append(MultiHeadGNN(hidden_dim, hidden_dim,
                                           mlp_dim, dropout, heads))

        self.GNN_layers = nn.ModuleList(GNN_layers)

        # post procesing
        self.post_procesing = nn.Sequential(nn.Dropout(p=dropout),
                                            nn.BatchNorm1d(hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, output_dim))

    def forward(self, x, adj_t):
        x = self.pre_procesing(x)
        for layer in self.GNN_layers:
            x = layer(x, adj_t)
        x = self.post_procesing(x)
        return x
