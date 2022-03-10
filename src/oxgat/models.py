import torch
import torch.nn.functional as F
from . import components


class TransductiveGATModel(torch.nn.Module):
    """Implementation of the transductive model defined in the original GAT paper.

    Parameters
    ----------
    in_features : int
        The number of features per node in the input data.
    num_classes : int
        The number of classes for node classification.
    """
    def __init__(self, in_features: int, num_classes: int):
        super(TransductiveGATModel, self).__init__()
        self.gat_layer_1 = components.GATLayer(in_features=in_features,
                                             out_features=8,
                                             num_heads=8,
                                             attention_dropout=0.6)
        self.gat_layer_2 = components.GATLayer(in_features=64,
                                             out_features=num_classes,
                                             is_final_layer=True,
                                             attention_dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6)
        x = self.gat_layer_1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6)
        x = self.gat_layer_2(x, edge_index)
        x = F.softmax(x, dim=1)
        return x