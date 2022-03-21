from sys import intern
import torch
import torch.nn as nn

from gat.layers import (Layer_Attention_MultiHead_GAT,
                        Layer_VanillaTransformer)


class GAT_Transductive(nn.Module):
    
    def __init__(self, input_dim, num_classes):
        super(GAT_Transductive, self).__init__()

        self.dropout_1 = nn.Dropout(p=0.6)
        self.attention_layer_1 = Layer_Attention_MultiHead_GAT(input_dim=input_dim,
                                                               repr_dim=8,
                                                               n_heads=8,
                                                               alpha=0.2,
                                                               attention_aggr='concat',
                                                               dropout=0.6)
        self.activation_1 = nn.ELU()

        self.dropout_2 = nn.Dropout(p=0.6)
        self.attention_layer_2 = Layer_Attention_MultiHead_GAT(input_dim=64,
                                                               repr_dim=num_classes,
                                                               n_heads=1,
                                                               alpha=0.2,
                                                               attention_aggr='concat',
                                                               dropout=0.6)

    """ Params:
        node_matrix: a `N x F` matrix of node features 

        adjacency_matrix: the `N x N` matrix giving the graph structure

        Returns:
            raw, unnormalised scores for each class, i.e. as specified 
            here `https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html`
            to be used in a CrossEntropyLoss
    """
    def forward(self, node_matrix, adj_matrix):
        node_matrix_dropout = self.dropout_1(node_matrix)

        z_1 = self.attention_layer_1(node_matrix_dropout, adj_matrix)
        a_1 = self.activation_1(z_1)

        a_1_dropout = self.dropout_2(a_1)
        
        z_2 = self.attention_layer_2(a_1_dropout, adj_matrix)

        return z_2


class GAT_Inductive(nn.Module):
    # NOTE: Here num_classes is fixed has default value 121 as per the formulation 
    # in the paper which says that their final layer has exactly 121-dimensional 
    # output
    def __init__(self, input_dim, num_classes=121):
        super(GAT_Inductive, self).__init__()

        self.attention_layer_1 = Layer_Attention_MultiHead_GAT(input_dim=input_dim,
                                                               repr_dim=256,
                                                               n_heads=4,
                                                               alpha=0.2,
                                                               attention_aggr='concat',
                                                               dropout=None)
        self.activation_1 = nn.ELU()

        self.attention_layer_2 = Layer_Attention_MultiHead_GAT(input_dim=1024,
                                                               repr_dim=256,
                                                               n_heads=4,
                                                               alpha=0.2,
                                                               attention_aggr='concat',
                                                               dropout=None)
        self.activation_2 = nn.ELU()

        self.attention_layer_3 = Layer_Attention_MultiHead_GAT(input_dim=1024,
                                                               repr_dim=num_classes,
                                                               n_heads=6,
                                                               alpha=0.2,
                                                               attention_aggr='mean',
                                                               dropout=None)
        self.activation_3 = nn.Sigmoid()

    """ Params:
        node_matrix: a `N x F` matrix of node features 

        adjacency_matrix: the `N x N` matrix giving the graph structure

        Returns:
            raw, unnormalised scores for each class, i.e. as specified 
            here `https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html`
            to be used in a CrossEntropyLoss
    """
    def forward(self, node_matrix, adj_matrix):
        z_1 = self.attention_layer_1(node_matrix, adj_matrix)
        a_1 = self.activation_1(z_1)

        z_2 = self.attention_layer_2(a_1, adj_matrix)
        a_2 = self.activation_2(z_2)

        # implementing the skip connection accross the intermediate 
        # attention layer
        a_2_skip = a_2 + a_1

        z_3 = self.attention_layer_3(a_2_skip, adj_matrix)
        a_3 = self.activation_3(z_3)

        return a_3


class VanillaTransformer_Transductive(nn.Module):
    def __init__(self, input_dim, num_classes, internal_dim,
                 num_layers, num_heads, dropout=None,
                 nonlinear_internal_dim=None,
                 identity_bias=0.01):
        super(VanillaTransformer_Transductive, self).__init__()
        assert(num_layers >= 1)
        key_dim = internal_dim // num_heads
        assert(key_dim > 0)

        # sequence of node vector dimensions:
        dims = [input_dim] + [internal_dim] * (num_layers - 1) + [num_classes]

        # default to a multiple of the internal dim
        if nonlinear_internal_dim is None:
            nonlinear_internal_dim = 2 * internal_dim

        # construct transformer layers based on this sequence of dimensions:
        self.layers = nn.ModuleList([
            Layer_VanillaTransformer(input_dim=in_dim, out_dim=out_dim, n_heads=num_heads,
                                     key_dim=key_dim,
                                     hidden_dim=nonlinear_internal_dim,
                                     dropout=dropout,
                                     identity_bias=identity_bias)
            for in_dim, out_dim in zip(dims[:-1], dims[1:])
        ])

    """ Params:
        node_matrix: a `N x input_dim` matrix of node features 

        adjacency_matrix: the `N x N` matrix giving the graph structure

        Returns:
            raw, unnormalised scores for each class, i.e. as specified 
            here `https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html`
            to be used in a CrossEntropyLoss
    """
    def forward(self, node_matrix, adj_matrix):
        for L in self.layers:
            node_matrix = L(node_matrix, adj_matrix)
        return node_matrix
