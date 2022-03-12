from abc import ABC, abstractmethod

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
import torch.nn.functional as F
import torch_geometric.loader

from . import components, utils


class AbstractModel(pl.LightningModule, ABC):
    """Abstract interface for models defined in this package.
    
    All models are subclasses of `pl.LightningModule` so can be trained using the
    standard PyTorch Lightning framework, but models here also provide methods to
    automate the training and testing in some standard way.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, data):
        """As required by PyTorch Lightning."""
    
    @abstractmethod
    def configure_optimizers(self):
        """As required by PyTorch Lightning."""

    @abstractmethod
    def training_step(self, data, batch_idx):
        """As required by PyTorch Lightning."""

    @abstractmethod
    def validation_step(self, data, batch_idx):
        """As required by PyTorch Lightning."""

    @abstractmethod
    def test_step(self, data, batch_idx):
        """As required by PyTorch Lightning."""

    @abstractmethod
    def standard_train(self, dataset, use_gpu=False):
        """Automated training interface.
        
        Parameters
        ----------
        dataset : torch_geometric.data.Dataset
            Dataset to train on.
        use_gpu : bool, optional
            Whether to use (single) GPU or not. Defaults to False.
        """

    @abstractmethod
    def standard_test(self, dataset):
        """Automated testing interface for after having run `self.standard_train()`.
        
        Parameters
        ----------
        dataset : torch_geometric.data.Dataset
            Dataset to test on.
        """


class TransductiveGATModel(AbstractModel):
    """Implementation of the transductive model defined in the original GAT paper.

    Parameters
    ----------
    in_features : int
        The number of features per node in the input data.
    num_classes : int
        The number of classes for node classification.
    """
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.gat_layer_1 = components.GATLayer(in_features=in_features,
                                             out_features=8,
                                             num_heads=8,
                                             attention_dropout=0.6)
        self.gat_layer_2 = components.GATLayer(in_features=64,
                                             out_features=num_classes,
                                             is_final_layer=True,
                                             attention_dropout=0.6)
        self.trainer = None

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
                                     lr=0.005,
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

    def standard_train(self, dataset, use_gpu=False):
        """Automated training of this model exactly as was done in
        the original paper.
        """
        self._init_trainer(use_gpu)
        dataloader = torch_geometric.loader.DataLoader(dataset)
        self.trainer.fit(self, train_dataloaders=dataloader, val_dataloaders=dataloader)

    def standard_test(self, dataset):
        """Method to test this model after having run `self.standard_train()`.
        """
        assert self.trainer is not None, "Must run `self.standard_train()` first."
        dataloader = torch_geometric.loader.DataLoader(dataset)
        self.trainer.test(self, dataloader)

    # Initialize the trainer for use in self.train()
    def _init_trainer(self, use_gpu):
        # Stop training only if neither validation loss nor accuracy has
        # improved in last 100 epochs.
        early_stopping = utils.MultipleEarlyStopping(
            monitors=["val_acc","val_loss"],
            modes=["max","min"],
            patience=100,
            verbose=False
        )
        trainer_args = {"max_epochs": 100000,
                        "log_every_n_steps": 1,
                        "callbacks": [early_stopping]}
        if use_gpu:
            trainer_args["gpus"] = 1
        self.trainer = pl.Trainer(**trainer_args)