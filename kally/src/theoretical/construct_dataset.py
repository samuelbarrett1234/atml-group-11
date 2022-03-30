""" This script contains procedures for constructing artificial
    problems which allows us to compare and benchmark the capacity 
    of different kinds of attention
"""
import itertools as it 
import random

import torch
import torch_geometric as tg 


""" This is the original one from the paper but it has a few downsides, including:

    - the proof that GAT has static attention relies on an implicit assumption about the graph 
    being complete, i.e. fully connected, but this dataset does not comprise of complete 
    graphs 

    - the feature structure is weird, i.e. they have integer indexes (as in a CBOW fashion).
    The need for such an obscure representation becomes apparent when one looks at their code -
    instead of using just a single attention layer as they claim in the paper, they in fact 
    make use of not just one but two embedding layers and do an aggregation 
    between them before making use of attention. On top of that, they even employ a third layer -
    a dense layer after the attention. This is not mentioned anywhere in the paper and we resort to 
    constructing our own dataset to compare expressivity
"""
def original_from_paper(k, train_test_split=0.8):
    dataset = []
    edge_list = list(it.product(range(k, 2*k), range(k)))
    common_edge_index = torch.tensor(edge_list, dtype=torch.long, requires_grad=False).transpose(0, 1)

    for permutation in it.permutations(range(k)):
        part_A_node_list = list(zip(range(k), k*[k]))
        part_B_node_list = list(zip(range(k), permutation))
        node_list = part_A_node_list + part_B_node_list

        node_features = torch.tensor(node_list, dtype=torch.long, requires_grad=False)
        targets = torch.tensor(permutation, dtype=torch.long, requires_grad=False)
        target_mask = torch.tensor(k*[True] + k*[False], dtype=torch.bool, requires_grad=False)

        data_point = tg.data.Data(x=node_features,
                                  edge_index=common_edge_index,
                                  y=targets,
                                  target_mask=target_mask)
        dataset.append(data_point)

    random.shuffle(dataset)
    k_factoriel = len(dataset)
    train_size = int(train_test_split*k_factoriel)

    train_set = dataset[:train_size]
    test_set = dataset[train_size:]

    return train_set, test_set


""" Our artificial dataset - a fully-connected graph with 2*k nodes.

    Each node has one of k possible labels and each label is present
    in exactly 2 nodes.The nodes are divided in two groups - those with
    indices 0 to k-1 and those between k and 2k-1. The feature vectors 
    are 2k-vectors with the first half one-encoding the exact index of
    the corresponding node (i.e. its identity, mod k). The second half is
    all zeros for the nodes in the first half and for the ones in the
    second half, it one-hot encodes their label. Each node has exactly
    one of k possible labels. Each node of the first group is paired
    with the corresponding node with the same identity in the second
    group and they have the same labels so that each label is present 
    in exactly 2 nodes.
    
    The idea of the task is for each node of the first group to learn 
    to attend exclusively to its corresponding 'twin' in the second 
    group, which holds the information needed for finding the correct 
    label. 

    This is a complete graph so all statements about staticity and
    dynamicity of attention mechanisms holds true. In our experiments 
    we run a network with exactly one layer - the corresponding attention 
    layer. We expect to find that models which are capable of 
    dynamic attention perform better on this task.
    
    Notice that the proof that GATv2 is dynamic actually relies on an 
    assumption that the representation dimension can be arbitrarily large,
    which is not the case here because we want to do classification with 
    this single layer so we must set the representation dimension to k
"""
def kn_artificial(k, train_test_split=0.8):
    dataset = []

    edge_list = list(it.product(range(2*k), range(2*k)))
    common_edge_index = torch.tensor(edge_list, dtype=torch.long, requires_grad=False).transpose(0, 1)

    common_eye = torch.eye(k)
    common_zeros = torch.zeros(k,k)

    for permutation in it.permutations(range(k)):
        perm = list(permutation)    
        part_A_features = torch.cat([common_eye, common_zeros], dim=1)
        part_B_features = torch.cat([common_eye, common_eye[perm]], dim=1)
        node_features = torch.cat([part_A_features, part_B_features], dim=0)

        targets = torch.tensor(permutation, dtype=torch.long, requires_grad=False)
        targets = torch.cat([targets, targets], dim=0)

        data_point = tg.data.Data(x=node_features,
                                  edge_index=common_edge_index,
                                  y=targets)
        dataset.append(data_point)

    random.shuffle(dataset)
    k_factoriel = len(dataset)
    train_size = int(train_test_split*k_factoriel)

    train_set = dataset[:train_size]
    test_set = dataset[train_size:]

    return train_set, test_set