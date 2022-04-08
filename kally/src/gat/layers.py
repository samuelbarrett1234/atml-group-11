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
        self.shared_attention = nn.Parameter(torch.empty(n_heads, 2*repr_dim))

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
        
        hidden_repr = torch.einsum('jk,ikl->ijl', node_matrix, self.W) # resulting shape n_heads x N x repr_dim

        first_half_full_attn = torch.einsum(
            'ijl,il->ij', hidden_repr, self.shared_attention[:, :self.repr_dim]) # result shape n_heads x N
        second_half_full_attn = torch.einsum(
            'ijl,il->ij', hidden_repr, self.shared_attention[:, self.repr_dim:]) # result shape n_heads x N
        full_attn = self.attention_activation(
            torch.unsqueeze(first_half_full_attn, 2)
            + torch.unsqueeze(second_half_full_attn, 1)) # result shape n_heads x N x N

        # masking out non-neighbourhood regions and summing using the attention weights
        neighbourhood_attention = self.softmax(
            full_attn + torch.unsqueeze(-1.0e16 * (1-adjacency_matrix), 0))
        if hasattr(self, 'dropout'):
            neighbourhood_attention = self.dropout(neighbourhood_attention)
        repr = torch.einsum('ijk,ikl->ijl', neighbourhood_attention, hidden_repr)  # result shape n_heads x N x repr_dim
        
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

        key_dim: The dimensionality of the keys and queries. Typically much
                 smaller than input/output dims, e.g. by dividing by `n_heads`.

        out_dim: The dimensionality of the output.

        n_heads: The number of attention heads.

        identity_bias: A small number to add to each attention value from
                       a node to itself, to promote nodes attending to themselves
                       and allowing a node to distinguish itself from its
                       neighbours. This is applied just before the softmax.
    """
    def __init__(self,
                 input_dim,
                 key_dim,
                 out_dim,
                 n_heads,
                 identity_bias=0.01):
        super(Layer_VanillaMHA, self).__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.num_heads = n_heads
        self.key_dim = key_dim
        self.identity_bias = identity_bias

        self.Wks = nn.Parameter(torch.empty(
            self.num_heads, self.input_dim, self.key_dim))  # keys
        self.Wqs = nn.Parameter(torch.empty(
            self.num_heads, self.input_dim, self.key_dim))  # queries
        self.Wvs = nn.Parameter(torch.empty(
            self.num_heads, self.input_dim, self.key_dim))  # values
        self.Wo = nn.Parameter(torch.empty(
            self.num_heads, self.key_dim, self.out_dim))  # output

        nn.init.xavier_normal_(self.Wks.data, gain= 2.0 ** 0.5)
        nn.init.xavier_normal_(self.Wqs.data, gain= 2.0 ** 0.5)
        nn.init.xavier_normal_(self.Wvs.data, gain= 2.0 ** 0.5)
        nn.init.xavier_normal_(self.Wo.data, gain= 2.0 ** 0.5)

        self.softmax = nn.Softmax(dim=-1)

    """ Params:
        node_matrix: a `N x input_dim` matrix of node features 

        adjacency_matrix: the `N x N` matrix giving the graph structure

        Returns:
            An `N x out_dim` matrix of new node features.
    """
    def forward(self,
                node_matrix,
                adjacency_matrix):
        # enforce self-loops:
        I_n = torch.eye(adjacency_matrix.shape[0], device=adjacency_matrix.device)
        adjacency_matrix = torch.maximum(adjacency_matrix, I_n)

        keys = torch.einsum('ij,mjn->min', node_matrix, self.Wks)
        queries = torch.einsum('ij,mjn->min', node_matrix, self.Wqs)
        values = torch.einsum('ij,mjn->min', node_matrix, self.Wvs)

        # atts[i, j, k] represents the attention amount from query
        # `j` to key `k` in attention head `i`
        atts = torch.einsum('ikn,imn->ikm', queries, keys)
        atts /= self.key_dim ** 0.5
        atts += torch.unsqueeze(self.identity_bias * I_n, 0)  # bias each node towards attending to itself
        atts -= 1.0e16 * torch.unsqueeze(1 - adjacency_matrix, 0)  # restrict vision to neighbourhoods
        atts = self.softmax(atts)

        return torch.einsum('ikl,ilm,imn->kn', atts, values, self.Wo)


class Layer_VanillaTransformer(nn.Module):
    """ Params:
        input_dim: The dimensionality of the input nodes.

        key_dim: The dimensionality of the keys and queries. Typically much
                 smaller than input/output dims, e.g. by dividing by `n_heads`.

        out_dim: The dimensionality of the output.

        n_heads: The number of attention heads.

        hidden_dim: The dimensionality of the internal hidden layer of the nonlinearity.

        identity_bias: See `Layer_VanillaMHA`.
    """
    def __init__(self,
                 input_dim,
                 key_dim,
                 out_dim,
                 n_heads,
                 hidden_dim,
                 identity_bias=0.0,
                 dropout_att=None,
                 dropout_hidden=None):
        super(Layer_VanillaTransformer, self).__init__()
        # attention
        self.mha = Layer_VanillaMHA(input_dim, key_dim, out_dim, n_heads,
                                    identity_bias=identity_bias)

        # want a separate bias than that offered by nn.Linear
        # so that we can force it to initialise to a small constant
        self.bias = nn.Parameter(torch.empty(
            hidden_dim
        ))
        nn.init.constant_(self.bias, 0.1)
        self.linear1 = nn.Linear(out_dim, hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_dim, out_dim, bias=True)
        self.relu = nn.ReLU()

        # layer normalisation
        self.ln1, self.ln2 = nn.LayerNorm(out_dim), nn.LayerNorm(out_dim)
        # dropout
        if dropout_att is not None:
            self.drop_att = nn.Dropout(dropout_att)
        else:
            self.drop_att = None
        if dropout_hidden is not None:
            self.drop_hidden = nn.Dropout(dropout_hidden)
        else:
            self.drop_hidden = None

    def anneal_attention_dropout(self, drop):
        """ Sets the dropout of the *attenion layer* (only) to `drop`.
        """
        if drop is None:
            self.drop_att = None
        elif self.drop_att is not None:
            self.drop_att.p = drop
        else:
            self.drop_att = nn.Dropout(drop)

    def anneal_hidden_dropout(self, drop):
        """ Sets the dropout of the *hidden layer* (only) to `drop`.
        """
        if drop is None:
            self.drop_hidden = None
        elif self.drop_hidden is not None:
            self.drop_hidden.p = drop
        else:
            self.drop_hidden = nn.Dropout(drop)

    """ Params:
        node_matrix: a `N x input_dim` matrix of node features 

        adjacency_matrix: the `N x N` matrix giving the graph structure

        Returns:
        a `N x out_dim` matrix of new node features.
    """
    def forward(self,
                node_matrix,
                adjacency_matrix):
        # attention
        node_matrix = self.mha(node_matrix, adjacency_matrix)

        # dropout
        if self.drop_att is not None:
            node_matrix = self.drop_att(node_matrix)

        # layer norm then nonlinearity
        node_matrix = self.ln1(node_matrix)
        node_matrix = self.relu(self.linear1(node_matrix) + self.bias)
        if self.drop_hidden is not None:  # dropout in the middle
            node_matrix = self.drop_hidden(node_matrix)
        node_matrix = self.linear2(node_matrix)

        # layer norm, then return result
        return self.ln2(node_matrix)


class Layer_Attention_Dynamic_GATWithBias(nn.Module):
    """ This is a modification of the original GAT attention
        (as in Layer_Attention_MultiHead_GAT) to add a trainable bias
        for each pair of nodes before computing the LeakyReLU,
        effectively turning the attention to a dynamic attention 

        Note: This method requires knowledge of the number of 
        nodes in advance and tehrefore is onlu suitable for 
        transductive tasks

        Params:
            All params in the plus

            n_nodes: the number of nodes that the input graphs will have

            epsilon_bias: the additional regulariser for the attentional biases
    """
    def __init__(self,
                 input_dim,
                 repr_dim,
                 n_heads,
                 n_nodes,
                 epsilon_bias=0.01,
                 alpha=0.2,
                 attention_aggr='concat',
                 dropout=0.6):
        super(Layer_Attention_Dynamic_GATWithBias, self).__init__()

        self.repr_dim = repr_dim
        self.n_heads = n_heads
        self.epsilon_bias = epsilon_bias

        if attention_aggr not in ['concat', 'mean']:
            raise ValueError('Unexpected value for attention_aggr: Attention aggregation scheme must either be `concat` or `mean`')
        self.attention_aggr = attention_aggr

        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)

        self.W = nn.Parameter(torch.empty(n_heads, input_dim, repr_dim))
        self.shared_attention = nn.Parameter(torch.empty(n_heads, 2*repr_dim))
        self.attention_biases = nn.Parameter(torch.empty(n_nodes, n_nodes))

        # see torch documentation for recommended values for certain activations
        nn.init.xavier_uniform_(self.W.data, gain=np.sqrt(2 / (1 + alpha**2)))
        nn.init.xavier_uniform_(self.shared_attention.data, gain=np.sqrt(2 / (1 + alpha**2)))
        nn.init.xavier_uniform_(self.attention_biases.data, gain=np.sqrt(2 / (1 + alpha**2)))
        
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
        
        hidden_repr = torch.einsum('jk,ikl->ijl', node_matrix, self.W) # resulting shape n_heads x N x repr_dim

        first_half_full_attn = torch.einsum(
            'ijl,il->ij', hidden_repr, self.shared_attention[:, :self.repr_dim]) # result shape n_heads x N
        second_half_full_attn = torch.einsum(
            'ijl,il->ij', hidden_repr, self.shared_attention[:, self.repr_dim:]) # result shape n_heads x N
        static_attn_preactivation = torch.unsqueeze(first_half_full_attn, 2) + torch.unsqueeze(second_half_full_attn, 1)
        dynamic_attn_preactivation = static_attn_preactivation + self.epsilon_bias*self.attention_biases
        full_attn = self.attention_activation(dynamic_attn_preactivation) # result shape n_heads x N x N

        neighbourhood_attention = self.softmax(
            full_attn + torch.unsqueeze(-1.0e16 * (1-adjacency_matrix), 0))
        
        if hasattr(self, 'dropout'):
            neighbourhood_attention = self.dropout(neighbourhood_attention)
        repr = torch.einsum('ijk,ikl->ijl', neighbourhood_attention, hidden_repr)  # result shape n_heads x N x repr_dim
        
        # final aggregation, resulting in shape `N x K*F'` if concat 
        # and shape `N x F'` if mean (the latter is usually done if
        # the layer is final)
        if self.attention_aggr == 'concat':
            repr = torch.cat(torch.unbind(repr, dim=0), dim=1)

        else: # self.attention_aggr == 'mean'
            repr = torch.mean(repr, dim=0)

        return repr


class Layer_Attention_MultiHead_GATv2(nn.Module):
    """ This is just the GATv2 version of the Layer_Attention_MultiHead_GAT layer
        The only difference with the original GAT is that the order of the second 
        linear transformation and the LeakyReLU are switched
    """
    def __init__(self,
                 input_dim,
                 repr_dim,
                 n_heads,
                 alpha=0.2,
                 attention_aggr='concat',
                 dropout=0.6):
        super(Layer_Attention_MultiHead_GATv2, self).__init__()

        self.repr_dim = repr_dim
        self.n_heads = n_heads

        if attention_aggr not in ['concat', 'mean']:
            raise ValueError('Unexpected value for attention_aggr: Attention aggregation scheme must either be `concat` or `mean`')
        self.attention_aggr = attention_aggr

        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)

        self.W = nn.Parameter(torch.empty(n_heads, input_dim, repr_dim))
        self.shared_attention = nn.Parameter(torch.empty(n_heads, 2*repr_dim))

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

        hidden_repr = torch.einsum('jk,ikl->ijl', node_matrix, self.W)
        activated_hidden_repr = self.attention_activation(hidden_repr)

        first_half_full_attn = torch.einsum(
            'ijl,il->ij', activated_hidden_repr, self.shared_attention[:, :self.repr_dim]) # result shape n_heads x N
        second_half_full_attn = torch.einsum(
            'ijl,il->ij', activated_hidden_repr, self.shared_attention[:, self.repr_dim:]) # result shape n_heads x N
        full_attn = torch.unsqueeze(first_half_full_attn, 2) \
            + torch.unsqueeze(second_half_full_attn, 1) # result shape n_heads x N x N

        # masking out non-neighbourhood regions and summing using the attention weights
        neighbourhood_attention = self.softmax(
            full_attn + torch.unsqueeze(-1.0e16 * (1-adjacency_matrix), 0))

        if hasattr(self, 'dropout'):
            neighbourhood_attention = self.dropout(neighbourhood_attention)
        repr = torch.einsum('ijk,ikl->ijl', neighbourhood_attention, hidden_repr)  # result shape n_heads x N x repr_dim
        
        # final aggregation, resulting in shape `N x K*F'` if concat 
        # and shape `N x F'` if mean (the latter is usually done if
        # the layer is final)
        if self.attention_aggr == 'concat':
            repr = torch.cat(torch.unbind(repr, dim=0), dim=1)

        else: # self.attention_aggr == 'mean'
            repr = torch.mean(repr, dim=0)

        return repr
