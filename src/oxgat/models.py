import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from . import components


class TransductiveGATModel(pl.LightningModule):
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
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat_layer_1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat_layer_2(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=0.5,
                                     weight_decay=0.0005)
        return optimizer

    def training_step(self, data, batch_idx):
        out = self(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, data, batch_idx):
        out = self(data)
        loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
        self.log("val_loss", loss)
        pred = out.argmax(dim=1)
        correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())
        self.log("val_acc", acc)

    def test_step(self, data, batch_idx):
        out = self(data)
        pred = out.argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())
        self.log("test_acc", acc)
