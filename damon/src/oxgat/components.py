import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj


class GATLayer(torch.nn.Module):
    """Represents a single multi-head attention layer.

    Parameters
    ----------
    in_features : int
        Length of each of the input node features.
    out_features : int
        Length of each of the output node features *per attention head*.
    leaky_relu_slope : int, optional
        Negative slope of the LeakyReLU activation, defaults to `0.2`.
    num_heads : int, optional
        How many independent attention heads to apply to the input.
        Defaults to `1`.
    is_final_layer : bool, optional
        Whether this module is the final attention layer in a model. If `True`
        the output is an average of the individual heads' output, otherwise it
        is a concatenation. Defaults to `False`.
    attention_dropout : float, optional
        Dropout probability to be applied to normalized attention coefficients
        during training. Defaults to `0` (no dropout).
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 leaky_relu_slope: float = 0.2,
                 num_heads: int = 1,
                 is_final_layer: bool = False,
                 attention_dropout: float = 0):
        super().__init__()
        self.is_final_layer = is_final_layer
        self.heads = torch.nn.ModuleList([AttentionHead(in_features,
                                                        out_features,
                                                        leaky_relu_slope,
                                                        attention_dropout)
                                          for _ in range(num_heads)])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """Applies this module forwards on an input graph.
        
        Parameters
        ----------
        x : torch.Tensor
            The node feature tensor of the input graph.
        edge_index : torch.Tensor
            The COO-formatted edge index tensor of the input graph (*not* an
            adjacency matrix).

        Returns
        -------
        torch.Tensor
            The new node feature tensor after applying this module.
        """
        outputs = [head(x, edge_index) for head in self.heads]
        output = torch.mean(torch.stack(outputs), dim=0) \
                 if self.is_final_layer \
                 else torch.cat(outputs, dim=1)
        return output


class AttentionHead(torch.nn.Module):
    """Represents a single attention head.
    
    Parameters
    ----------
    in_features : int
        Length of each of the input node features.
    out_features : int
        Length of each of the output node features.
    leaky_relu_slope : int, optional
        Negative slope of the LeakyReLU activation, defaults to `0.2`.
    neighbourhood_depth : int, optional
        Calculate attention only between nodes up to this many edges apart.
        Defaults to 1.
    attention_dropout : float, optional
        Dropout probability to be applied to normalized attention coefficients
        during training. Defaults to `0` (no dropout).
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 leaky_relu_slope: bool = 0.2,
                 attention_dropout: float = 0,
                 neighbourhood_depth: int = 1):
        super().__init__()

        self.W = torch.nn.Linear(in_features, out_features, bias=False)
        self.a1 = torch.nn.Linear(out_features, 1, bias=False)
        self.a2 = torch.nn.Linear(out_features, 1, bias=False)

        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=leaky_relu_slope)
        self.neighbourhood_depth = neighbourhood_depth
        self.attention_dropout = attention_dropout

        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W.weight, gain=np.sqrt(2))
        torch.nn.init.xavier_uniform_(self.a1.weight, gain=np.sqrt(2))
        torch.nn.init.xavier_uniform_(self.a2.weight, gain=np.sqrt(2))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """Applies this module forwards on an input graph.
        
        Parameters
        ----------
        x : torch.Tensor
            The node feature tensor of the input graph.
        edge_index : torch.Tensor
            The COO-formatted edge index tensor of the input graph (*not* an
            adjacency matrix).

        Returns
        -------
        torch.Tensor
            The new node feature tensor after applying this module.
        """
        x = self.W(x)
        attention = F.dropout(self._attention(x, edge_index),
                              p=self.attention_dropout,
                              training=self.training)
        return attention @ x

    # Calculates the attention matrix for a graph
    def _attention(self, x, edge_index):
        n = x.shape[0]
        adj = torch.squeeze(to_dense_adj(edge_index, max_num_nodes=n))
        if self.neighbourhood_depth > 1:
            # Calculate higher-order adjacency matrix
            adj = torch.matrix_power(adj, self.neighbourhood_depth)

        zero = -9e15 * torch.ones(n,n).type_as(x)
        e = torch.where(adj > 0, self.a1(x) + self.a2(x).T, zero)
        return F.softmax(self.leaky_relu(e), dim=1)
