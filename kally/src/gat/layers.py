import torch 
import torch.nn as nn
import numpy as np


class Layer_Attention_MultiHead_GAT(nn.Module):
    """ Params:
        input_dim: The dimensionality of the input nodes, this is `F`
        in the paper

        repr_dim: The embedding dimensionality of the layer, this is `F'`
        in the  paper

        n_heads: The number of attention heads, this is K in the paper

        alpha: The leaky ReLU slope parameter, `alpha` in the paper

        attention_aggr: The aggregation scheme for the multihead attention
        must be in ['concat', 'mean'] as per the paper

        dropout: If dropout != None, will use it to this as the probability
        to apply dropout to the normalised attention weights. The authors
        of GAT do this in the transductive setting.

        NOTE: We don't apply a nonlinearity in the end - we don't consider the
        activation sigma to be part of the attention layer itself. Similarly, 
        dropout on input is handled outside this layer
    """
    def __init__(self,
                 input_dim,
                 repr_dim,
                 n_heads,
                 alpha=0.2,
                 attention_aggr='concat',
                 dropout=0.6):
        super(Layer_Attention_MultiHead_GAT, self).__init__()

        self.repr_dim = repr_dim
        self.n_heads = n_heads

        if attention_aggr not in ['concat', 'mean']:
            raise ValueError('Unexpected value for attention_aggr: Attention aggregation scheme must either be `concat` or `mean`')
        self.attention_aggr = attention_aggr

        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)

        self.W = nn.Parameter(torch.empty(n_heads, input_dim, repr_dim))
        self.shared_attention = nn.Parameter(torch.empty(n_heads, 2*repr_dim, 1))

        # see torch documentation for recommended values for certain activations
        nn.init.xavier_uniform_(self.W.data, gain=np.sqrt(2 / (1 + alpha**2)))
        nn.init.xavier_uniform_(self.shared_attention.data, gain=np.sqrt(2 / (1 + alpha**2)))
        
        self.attention_activation = nn.LeakyReLU(alpha)
        self.softmax = nn.Softmax(dim=-1)

    """ Params:
        node_matrix: a `N x F` matrix of node features 

        adjacency_matrix: the `N x N` matrix giving the graph structure

        Returns:
            if self.attention_aggr == 'concat':
                An `N x K*F'` matrix where F' is the representation
                dimension and K is the number of heads
            
            if self.attention_aggr == 'mean':
                An `N x F'` matrix where F' is the representation
                dimension

            The results correspond to equations (4) and (6) in the paper 
            without the nonlinearity sigma
    """
    def forward(self,
                node_matrix,
                adjacency_matrix):
        
        # the initial linear transformation W_k*h resulting in a shape `K x N x F'`
        nodes_stacked = torch.stack([node_matrix for _ in range(self.n_heads)])
        hidden_repr = torch.bmm(nodes_stacked, self.W) # resulting shape n_heads x N x repr_dim

        # implementing the linear attention function making use of broadcasting
        first_half_full_attn = torch.bmm(hidden_repr, self.shared_attention[:, :self.repr_dim, :]).view(self.n_heads, 1, -1)
        second_half_full_attn = torch.bmm(hidden_repr, self.shared_attention[:, self.repr_dim:, :]).view(self.n_heads, -1, 1)
        full_attn = self.attention_activation(first_half_full_attn + second_half_full_attn)

        # masking out non-neighbourhood regions and summing using the attention weights
        adjacency_stacked = torch.stack([adjacency_matrix for _ in range(self.n_heads)])
        mask = -1e16 * torch.ones_like(adjacency_stacked)
        neighbourhood_attention = torch.where(adjacency_stacked > 0, full_attn, mask)
        neighbourhood_attention = self.softmax(neighbourhood_attention)
        if hasattr(self, 'dropout'):
            neighbourhood_attention = self.dropout(neighbourhood_attention)
        repr = torch.bmm(neighbourhood_attention, hidden_repr)
        
        # final aggregation, resulting in shape `N x K*F'` if concat 
        # and shape `N x F'` if mean (the latter is usually done if
        # the layer is final)
        if self.attention_aggr == 'concat':
            repr = torch.cat(torch.unbind(repr, dim=0), dim=1)

        else: # self.attention_aggr == 'mean'
            repr = torch.mean(repr, dim=0)

        return repr


class Layer_VanillaMHA(nn.Module):
    """ Params:
        input_dim: The dimensionality of the input nodes.

        repr_dim: The embedding dimensionality of the layer.

        n_heads: The number of attention heads.
    """
    def __init__(self,
                 input_dim,
                 repr_dim,
                 n_heads):
        super(Layer_VanillaMHA, self).__init__()

        self.input_dim = input_dim
        self.out_dim = repr_dim
        self.num_heads = n_heads
        self.key_dim = self.out_dim // self.num_heads

        self.Wks = nn.Parameter(torch.empty(
            self.num_heads, self.input_dim, self.key_dim))  # keys
        self.Wqs = nn.Parameter(torch.empty(
            self.num_heads, self.input_dim, self.key_dim))  # queries
        self.Wvs = nn.Parameter(torch.empty(
            self.num_heads, self.input_dim, self.key_dim))  # values
        self.Wo = nn.Parameter(torch.empty(
            self.num_heads, self.key_dim, self.out_dim))  # output

        nn.init.normal_(self.Wks.data, std=input_dim ** -0.5)
        nn.init.normal_(self.Wqs.data, std=input_dim ** -0.5)
        nn.init.normal_(self.Wvs.data, std=input_dim ** -0.5)
        nn.init.normal_(self.Wo.data, std=input_dim ** -0.5)

        self.softmax = nn.Softmax(dim=-1)

    """ Params:
        node_matrix: a `N x self.input_dim` matrix of node features 

        adjacency_matrix: the `N x N` matrix giving the graph structure

        Returns:
            An `N x self.repr_dim` matrix of new node features.
    """
    def forward(self,
                node_matrix,
                adjacency_matrix):
        keys = torch.einsum('ij,mkn->min', node_matrix, self.Wks)
        queries = torch.einsum('ij,mkn->min', node_matrix, self.Wqs)
        values = torch.einsum('ij,mkn->min', node_matrix, self.Wvs)

        # atts[i, j, k] represents the attention amount from query
        # `j` to key `k` in attention head `i`
        atts = torch.einsum('ikn,imn->ikm', queries, keys)
        atts /= self.key_dim ** 0.5
        atts -= 1.0e16 * torch.unsqueeze(1 - adjacency_matrix, 0)  # restrict vision to neighbourhoods
        atts = self.softmax(atts)

        return torch.einsum('ikj,ilm,imn->kn', atts, values, self.Wo)


class Layer_VanillaTransformer(nn.Module):
    """ Params:
        input_dim: The dimensionality of the input nodes.

        repr_dim: The embedding dimensionality of the layer.

        n_heads: The number of attention heads.

        hidden_dim: The dimensionality of the internal hidden layer of the nonlinearity.
    """
    def __init__(self,
                 input_dim,
                 repr_dim,
                 n_heads,
                 hidden_dim,
                 dropout=None):
        super(Layer_VanillaTransformer, self).__init__()
        # attention
        self.mha = Layer_VanillaMHA(input_dim, repr_dim, n_heads)
        # nonlinearity
        self.nonlinearity = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, repr_dim)
        )
        # layer normalisation
        self.ln1, self.ln2 = nn.LayerNorm(repr_dim), nn.LayerNorm(repr_dim)
        # dropout
        if dropout is not None:
            self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout)
        else:
            self.drop1, self.drop2 = None, None

    """ Params:
        node_matrix: a `N x input_dim` matrix of node features 

        adjacency_matrix: the `N x N` matrix giving the graph structure

        Returns:
        a `N x repr_dim` matrix of new node features.
    """
    def forward(self,
                node_matrix,
                adjacency_matrix):
        # attention
        node_matrix = self.mha(node_matrix, adjacency_matrix)

        # dropout
        if self.drop1 is not None:
            node_matrix = self.drop1(node_matrix)

        # layer norm then nonlinearity
        node_matrix = self.nonlinearity(self.ln1(node_matrix))

        # dropout again
        if self.drop2 is not None:
            node_matrix = self.drop2(node_matrix)

        # layer norm, then return result
        return self.ln2(node_matrix)
