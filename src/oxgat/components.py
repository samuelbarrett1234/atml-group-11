import torch
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
        super(GATLayer, self).__init__()
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
                 else torch.concat(outputs, dim=1)
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
        super(AttentionHead, self).__init__()
        self.W = torch.nn.Linear(in_features, out_features, bias=False)
        self.a = torch.nn.Linear(out_features*2, 1, bias=False)
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=leaky_relu_slope)
        self.neighbourhood_depth = neighbourhood_depth
        self.attention_dropout = attention_dropout

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
        x = self.W(x) # Shared linear map
        attention = self._attention(x, torch.squeeze(to_dense_adj(edge_index)))
        attention = F.dropout(attention, p=self.attention_dropout)
        return attention @ x

    # Calculates the attention matrix for a graph
    def _attention(self, x, adj):
        n = x.shape[0]

        # Get attention mechanism input for each pair of nodes
        x_repeated = x.repeat(n,1,1) # Shape (n,n,F')
        feature_pairs = torch.cat([x_repeated,
                                   x_repeated.transpose(0,1)],
                                  dim=2) # Shape (n,n,2F')

        # Calculate higher-order adjacency matrix if necessary
        if self.neighbourhood_depth > 1:
            adj = torch.matrix_power(adj, self.neighbourhood_depth)

        # Calculate attention for each edge
        e = torch.where(adj > 0,
                        torch.squeeze(self.a(feature_pairs)),
                        torch.zeros(n,n))
        attention = F.softmax(self.leaky_relu(e), dim=1)

        return attention