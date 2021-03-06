import torch
import torch.nn as nn

from gat.layers import (Layer_Attention_MultiHead_GAT,
                        Layer_VanillaTransformer,
                        Layer_Attention_MultiHead_GATv2)


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

        return z_3


class VanillaTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, internal_dim,
                 num_layers, num_heads, pos_emb_dim,
                 nonlinear_internal_dim=None,
                 identity_bias=0.01,
                 dropout_att=None,
                 dropout_hidden=None,
                 skip_conn=False):
        super(VanillaTransformer, self).__init__()
        assert(num_layers >= 1)
        key_dim = internal_dim // num_heads
        assert(key_dim > 0)
        self.skip_conn = skip_conn

        # default to a multiple of the internal dim
        if nonlinear_internal_dim is None:
            nonlinear_internal_dim = 2 * internal_dim
            
        self.pre = nn.Linear(input_dim, internal_dim, bias=False)
        self.post = nn.Linear(internal_dim, num_classes)

        # construct transformer layers based on this sequence of dimensions:
        self.layers = nn.ModuleList([
            Layer_VanillaTransformer(input_dim=internal_dim, out_dim=internal_dim,
                                     n_heads=num_heads,
                                     key_dim=key_dim,
                                     hidden_dim=nonlinear_internal_dim,
                                     dropout_att=dropout_att,
                                     dropout_hidden=dropout_hidden,
                                     identity_bias=identity_bias)
            for _ in range(num_layers)
        ])

        self.pos_emb_dim = pos_emb_dim
        self.pos_emb_linear = nn.Linear(pos_emb_dim, internal_dim)

    def anneal_attention_dropout(self, drop):
        """Anneals attention dropout on the transformer sublayer.
        """
        for L in self.layers:
            L.anneal_attention_dropout(drop)

    def anneal_hidden_dropout(self, drop):
        """Anneals hidden dropout on the transformer sublayer.
        """
        for L in self.layers:
            L.anneal_attention_dropout(drop)

    """ Params:
        node_matrix: a `N x input_dim` matrix of node features 

        adjacency_matrix: the `N x N` matrix giving the graph structure

        `pos_embs`: Nx`pos_emb_dim` matrix of positional embeddings

        Returns:
            raw, unnormalised scores for each class, i.e. as specified 
            here `https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html`
            to be used in a CrossEntropyLoss
    """
    def forward(self, node_matrix, adj_matrix, pos_embs):
        node_matrix = self.pre(node_matrix)
        node_matrix = node_matrix + self.pos_emb_linear(pos_embs)
        if not self.skip_conn:
            for L in self.layers:
                node_matrix = L(node_matrix, adj_matrix)
        else:
            node_matrix = self.layers[0](node_matrix, adj_matrix)
            for L in self.layers[1:-1]:
                node_matrix = node_matrix + L(node_matrix, adj_matrix)
            node_matrix = self.layers[-1](node_matrix, adj_matrix)
        return self.post(node_matrix)


class UniversalTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, internal_dim,
                 num_layers, num_heads, pos_emb_dim,
                 nonlinear_internal_dim=None,
                 identity_bias=0.01,
                 dropout_att=None,
                 dropout_hidden=None,
                 skip_conn=False):
        super(UniversalTransformer, self).__init__()
        key_dim = internal_dim // num_heads
        assert(key_dim > 0)
        self.skip_conn = skip_conn

        # default to a multiple of the internal dim
        if nonlinear_internal_dim is None:
            nonlinear_internal_dim = 2 * internal_dim

        # a universal transformer is basically a single
        # transformer layer iterated a number of times,
        # wrapped in two linear layers to convert the
        # input and output sizes:
        self.pre = nn.Linear(input_dim, internal_dim, bias=False)
        self.transformer = Layer_VanillaTransformer(
            input_dim=internal_dim, out_dim=internal_dim, n_heads=num_heads,
            key_dim=key_dim,
            hidden_dim=nonlinear_internal_dim,
            dropout_att=dropout_att,
            dropout_hidden=dropout_hidden,
            identity_bias=identity_bias)
        self.post = nn.Linear(internal_dim, num_classes)
        self.num_layers = num_layers

        self.pos_emb_dim = pos_emb_dim
        self.pos_emb_linear = nn.Linear(pos_emb_dim, internal_dim)

    def anneal_attention_dropout(self, drop):
        """Anneals attention dropout on the transformer sublayer.
        """
        self.transformer.anneal_attention_dropout(drop)

    def anneal_hidden_dropout(self, drop):
        """Anneals hidden dropout on the transformer sublayer.
        """
        self.transformer.anneal_hidden_dropout(drop)

    """ Params:
        node_matrix: a `N x input_dim` matrix of node features 

        adjacency_matrix: the `N x N` matrix giving the graph structure

        `pos_embs`: Nx`pos_emb_dim` matrix of positional embeddings

        Returns:
            raw, unnormalised scores for each class, i.e. as specified 
            here `https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html`
            to be used in a CrossEntropyLoss
    """
    def forward(self, node_matrix, adj_matrix, pos_embs):
        node_matrix = self.pre(node_matrix)
        node_matrix = node_matrix + self.pos_emb_linear(pos_embs)
        for _ in range(self.num_layers):
            next_matrix = self.transformer(node_matrix, adj_matrix)
            if not self.skip_conn:
                node_matrix = next_matrix
            else:
                node_matrix = node_matrix + next_matrix
        return self.post(node_matrix)


class GATv2(nn.Module):
    def __init__(self, input_dim, num_classes, internal_dim,
                 num_layers, num_heads,
                 dropout=None,
                 skip_conn=False,
                 alpha=0.2,
                 attention_aggr='concat'):
        super(GATv2, self).__init__()
        assert(num_layers >= 1)
        key_dim = (internal_dim // num_heads if attention_aggr == 'concat' else internal_dim)
        assert(key_dim > 0)
        self.skip_conn = skip_conn

        # sequence of node vector dimensions (for inputs and outputs):
        in_dims = [input_dim] + [internal_dim] * (num_layers - 1)
        out_dims = [key_dim] * (num_layers - 1) + [num_classes]
        attention_aggrs = [attention_aggr] * (num_layers - 1) + ['mean']

        # construct transformer layers based on this sequence of dimensions:
        self.layers = nn.ModuleList([
            Layer_Attention_MultiHead_GATv2(
                input_dim=in_dim, repr_dim=out_dim, n_heads=num_heads,
                alpha=alpha, attention_aggr=att_agg, dropout=dropout)
            for in_dim, out_dim, att_agg in zip(in_dims, out_dims, attention_aggrs)
        ])
        self.activations = nn.ModuleList([
            nn.ELU() for _ in range(num_layers - 1)
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
        if not self.skip_conn:
            for L, act in zip(self.layers[:-1], self.activations):
                node_matrix = act(L(node_matrix, adj_matrix))
        else:
            node_matrix = self.activations[0](self.layers[0](node_matrix, adj_matrix))
            for L, act in zip(self.layers[1:-1], self.activations[1:]):
                node_matrix = node_matrix + act(L(node_matrix, adj_matrix))
        return self.layers[-1](node_matrix, adj_matrix)
