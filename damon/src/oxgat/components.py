import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from . import utils


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
    strict_neighbourhoods : bool, optional
        If `True`, only allow a node to pay attention to other nodes connected by
        a path of length *exactly* `neighbourhood_depth`; if `False` allow
        connections of length *up to* `neighbourhood_depth`. In particular, when
        `neighbourhood_depth=1` this controls whether nodes can pay attention to
        themselves or not. Defaults to `False`.
    sparse : bool, optional
        Whether to use sparse matrix operations. Must be `False` if
        `neighbourhood_depth > 1`. Defaults to `True`.
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 leaky_relu_slope: float = 0.2,
                 num_heads: int = 1,
                 is_final_layer: bool = False,
                 attention_dropout: float = 0,
                 strict_neighbourhoods: bool = False,
                 sparse: bool = True):
        super().__init__()
        self.is_final_layer = is_final_layer
        self.heads = torch.nn.ModuleList([AttentionHead(in_features,
                                                        out_features,
                                                        leaky_relu_slope,
                                                        attention_dropout,
                                                        strict_neighbourhoods=strict_neighbourhoods,
                                                        sparse=sparse)
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
    attention_dropout : float, optional
        Dropout probability to be applied to normalized attention coefficients
        during training. Defaults to `0` (no dropout).
    neighbourhood_depth : int, optional
        Calculate attention only between nodes up to this many edges apart.
        Defaults to 1.
    strict_neighbourhoods : bool, optional
        If `True`, only allow a node to pay attention to other nodes connected by
        a path of length *exactly* `neighbourhood_depth`; if `False` allow
        connections of length *up to* `neighbourhood_depth`. In particular, when
        `neighbourhood_depth=1` this controls whether nodes can pay attention to
        themselves or not. Defaults to `False`.
    sparse : bool, optional
        Whether to use sparse matrix operations. Must be `False` if
        `neighbourhood_depth > 1`. Defaults to `True`.
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 leaky_relu_slope: bool = 0.2,
                 attention_dropout: float = 0,
                 neighbourhood_depth: int = 1,
                 strict_neighbourhoods: bool = False,
                 sparse: bool = True):
        super().__init__()
        assert neighbourhood_depth == 1 or not sparse, \
            "Sparse computation only supported for `neighbourhood_depth=1`."

        self.W = torch.nn.Linear(in_features, out_features, bias=False)
        self.a1 = torch.nn.Linear(out_features, 1, bias=False)
        self.a2 = torch.nn.Linear(out_features, 1, bias=False)

        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=leaky_relu_slope)
        self.neighbourhood_depth = neighbourhood_depth
        self.attention_dropout = attention_dropout
        self.strict_neighbourhoods = strict_neighbourhoods
        self.sparse = sparse

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
        if self.sparse:
            sparse_attention = self._sparse_attention(x, edge_index)
            sparse_attention = utils.sparse_dropout(sparse_attention,
                                                    p=self.attention_dropout,
                                                    training=self.training)
            return torch.sparse.mm(sparse_attention, x)
        else:
            attention = self._sparse_attention(x, edge_index)
            attention = F.dropout(attention,
                                  p=self.attention_dropout,
                                  training=self.training)
            return attention @ x

    # Calculates the attention matrix for a graph
    def _attention(self, x, edge_index):
        n = x.shape[0]
        adj = torch.squeeze(to_dense_adj(edge_index, max_num_nodes=n))
        if not self.strict_neighbourhoods: # Add self-loops
            adj += torch.diag(torch.ones(n, dtype=torch.long)).type_as(x)
        if self.neighbourhood_depth > 1:
            # Calculate higher-order adjacency matrix
            adj = torch.matrix_power(adj, self.neighbourhood_depth)

        zero = -9e15 * torch.ones(n,n).type_as(x)
        e = torch.where(adj > 0, self.a1(x) + self.a2(x).T, zero)
        return F.softmax(self.leaky_relu(e), dim=1)

    # Calculates the sparse attention matrix for a graph
    def _sparse_attention(self, x, edge_index):
        n = x.shape[0]
        if not self.strict_neighbourhoods: # Add self-loops
            self_loops = torch.arange(n, dtype=torch.long).type_as(edge_index).expand(2,n)
            edge_index = torch.cat([edge_index, self_loops], dim=1)
        attention_vals = self.leaky_relu(self.a1(x[edge_index[0,:],:]) + \
                                         self.a2(x[edge_index[1,:],:])).flatten()
        e = torch.sparse_coo_tensor(edge_index, attention_vals, size=(n,n))
        return torch.sparse.softmax(e, dim=1)