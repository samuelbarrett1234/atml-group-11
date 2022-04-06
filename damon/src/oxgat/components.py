"""Provides individual model components (layers, attention heads) for use
in various model architectures.
"""
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch_geometric.transforms import OneHotDegree
from torch_geometric.data import Data
from typing import Type

from . import utils


class AbstractAttentionHead(torch.nn.Module, ABC):
    @abstractmethod
    def __init__(self):
        """Initialization"""

    @abstractmethod
    def reset_parameters(self):
        """(Re)-initialize parameters"""

    @abstractmethod
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """Forward pass"""


class GATAttentionHead(AbstractAttentionHead):
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
        torch.nn.Module.__init__(self)
        assert neighbourhood_depth == 1 or not sparse, \
            "Sparse computation only supported for `neighbourhood_depth=1`."

        self.W = torch.nn.Linear(in_features, out_features, bias=False)
        self.a1 = torch.nn.Linear(out_features, 1, bias=False)
        self.a2 = torch.nn.Linear(out_features, 1, bias=False)

        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=leaky_relu_slope)
        self.leaky_relu_slope = leaky_relu_slope
        self.neighbourhood_depth = neighbourhood_depth
        self.attention_dropout = attention_dropout
        self.strict_neighbourhoods = strict_neighbourhoods
        self.sparse = sparse
        self.out_features = out_features

        self.reset_parameters()
        
    def reset_parameters(self):
        W_gain = np.sqrt(1.55/self.W.in_features) # Optimized for ELU
        a_gain = torch.nn.init.calculate_gain("leaky_relu",
                                              self.leaky_relu_slope)
        a_correction = np.sqrt((self.out_features+1)/(2*self.out_features+1))
        torch.nn.init.xavier_uniform_(self.W.weight, gain=W_gain)
        torch.nn.init.xavier_uniform_(self.a1.weight, gain=a_correction*a_gain)
        torch.nn.init.xavier_uniform_(self.a2.weight, gain=a_correction*a_gain)

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
            attention = self._attention(x, edge_index)
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


class GATv2AttentionHead(AbstractAttentionHead):
    """Represents a single GATv2 attention head.
    
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
    weight_sharing : bool, optional
        Whether to require W_1=W_2. Defaults to True.
    bias : bool, optional
        Whether to add a bias term (to the learnt weight matrix/matrices).
        Defaults to True.
    feature_update_matrix : str, optional
        String in ["source", "target", "separate"]. Defaults to "source".
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 leaky_relu_slope: bool = 0.2,
                 attention_dropout: float = 0,
                 weight_sharing: bool = True,
                 bias: bool = True,
                 feature_update_matrix: str = "source"):
        torch.nn.Module.__init__(self)
        assert feature_update_matrix in ["source", "target", "separate"]
        
        self.W1 = torch.nn.Linear(in_features, out_features, bias=bias)
        self.W2 = (self.W1 if weight_sharing 
                   else torch.nn.Linear(in_features, out_features, bias=bias))
        if feature_update_matrix == "separate":
            self.W3 = torch.nn.Linear(in_features, out_features, bias=bias)
        self.a1 = torch.nn.Linear(out_features, 1, bias=False)
        self.a2 = torch.nn.Linear(out_features, 1, bias=False)

        # Reference for feature update matrix
        self.W = self.W1 if feature_update_matrix == "target" else (
                 self.W2 if feature_update_matrix == "source" else
                 self.W3)
        
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=leaky_relu_slope)
        self.leaky_relu_slope = leaky_relu_slope
        self.attention_dropout = attention_dropout
        self.out_features = out_features
        self.weight_sharing = weight_sharing
        self.feature_update_matrix = feature_update_matrix
        self.separate = (feature_update_matrix == "separate")

        self.reset_parameters()

    def reset_parameters(self):
        # TODO: add corrections
        W_gain = torch.nn.init.calculate_gain("leaky_relu",
                                              self.leaky_relu_slope)
        torch.nn.init.xavier_uniform_(self.W1.weight, gain=W_gain)
        if not self.weight_sharing:
            torch.nn.init.xavier_uniform_(self.W2.weight, gain=W_gain)
        if self.separate:
            torch.nn.init.xavier_uniform_(self.W3.weight, gain=W_gain)
        torch.nn.init.xavier_uniform_(self.a1.weight, gain=1)
        torch.nn.init.xavier_uniform_(self.a2.weight, gain=1)

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
        sparse_attention = self._sparse_attention(x, edge_index)
        sparse_attention = utils.sparse_dropout(sparse_attention,
                                                p=self.attention_dropout,
                                                training=self.training)
        return torch.sparse.mm(sparse_attention, self.W(x))

    # Calculates the sparse attention matrix for a graph
    def _sparse_attention(self, x, edge_index):
        n = x.shape[0]

        # Add self-loops
        self_loops = torch.arange(n, dtype=torch.long).type_as(edge_index).expand(2,n)
        edge_index = torch.cat([edge_index, self_loops], dim=1)

        # Calculate attention
        x1, x2 = self.leaky_relu(self.W1(x)), self.leaky_relu(self.W2(x))
        attention_vals = (self.a1(x1[edge_index[0,:],:]) + 
                          self.a2(x2[edge_index[1,:],:])).flatten()
        e = torch.sparse_coo_tensor(edge_index, attention_vals, size=(n,n))
        return torch.sparse.softmax(e, dim=1)


class MultiHeadAttentionLayer(torch.nn.Module):
    """Represents a single multi-head attention layer.

    Parameters
    ----------
    attention_type : Type[AbstractAttentionHead]
        The class of attention head to use.
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
    **kwargs
        Keyword arguments to be passed to the attention heads themselves.
    """
    def __init__(self,
                 attention_type: Type[AbstractAttentionHead],
                 in_features: int,
                 out_features: int,
                 num_heads: int = 1,
                 is_final_layer: bool = False,
                 **kwargs):
        super().__init__()
        self.is_final_layer = is_final_layer
        self.in_features = in_features
        self.heads = torch.nn.ModuleList([attention_type(in_features,
                                                         out_features,
                                                         **kwargs)
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


class MultiHeadAttentionLayerWithDegrees(MultiHeadAttentionLayer): # TODO: generalise to weighted degree
    def __init__(self, *args, **kwargs):
        max_degree = kwargs.pop("max_degree") # Required
        self.add_degrees = OneHotDegree(max_degree)
        if "in_features" in kwargs:
            kwargs["in_features"] += max_degree+1
        else:
            args[1] += max_degree+1
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = self.add_degrees(Data(x, edge_index)).x
        return super().forward(x, edge_index)
