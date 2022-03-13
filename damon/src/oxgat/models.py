from abc import ABC, abstractmethod

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
import torch.nn.functional as F
import torch_geometric.loader
from sklearn.metrics import f1_score

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


class BaseGATModel(AbstractModel):
    """Semi-abstract base class for inductive and transductive models defined in
    the original GAT paper. See superclass and subclasses for documentation.
    """
    def __init__(self, lr, regularisation=0, train_batch_size=1):
        super().__init__()
        self.lr = lr
        self.regularisation = regularisation
        self.train_batch_size = train_batch_size
        self.trainer = None

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.regularisation)

    def standard_train(self, train_dataset, val_dataset=None, use_gpu=False):
        """Automated training of this model exactly as was done in
        the original paper.
        """
        self._init_trainer(use_gpu)
        if val_dataset is None:
            val_dataset = train_dataset
        train_loader = torch_geometric.loader.DataLoader(train_dataset,
                                                         batch_size=self.train_batch_size,
                                                         shuffle=True)
        val_loader = torch_geometric.loader.DataLoader(val_dataset)
        self.trainer.fit(self, train_dataloaders=train_loader, val_dataloaders=val_loader)

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


class TransductiveGATModel(BaseGATModel):
    """Implementation of the transductive model defined in the original GAT paper.

    Parameters
    ----------
    in_features : int
        The number of features per node in the input data.
    num_classes : int
        The number of classes for node classification.
    pubmed : bool, optional
        Whether to apply the architecture changes made for the PubMed dataset
        in the original paper. Defaults to False.
    """
    def __init__(self, in_features: int, num_classes: int, pubmed: bool = False):
        super().__init__(lr=0.01 if pubmed else 0.005,
                         regularisation=0.001 if pubmed else 0.0005)
        self.gat_layer_1 = components.GATLayer(in_features=in_features,
                                               out_features=8,
                                               num_heads=8,
                                               attention_dropout=0.6)
        self.gat_layer_2 = components.GATLayer(in_features=64,
                                               out_features=num_classes,
                                               num_heads=8 if pubmed else 1,
                                               is_final_layer=True,
                                               attention_dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat_layer_1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat_layer_2(x, edge_index)
        return F.log_softmax(x, dim=1)

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


class InductiveGATModel(BaseGATModel):
    """Implementation of the inductive model defined in the original GAT paper.

    Parameters
    ----------
    in_features : int
        The number of features per node in the input data.
    num_classes : int
        The number of classes for node classification.
    """
    def __init__(self, in_features: int, num_classes: int):
        super().__init__(lr=0.005, train_batch_size=2)
        self.gat_layer_1 = components.GATLayer(in_features=in_features,
                                               out_features=256,
                                               num_heads=4)
        self.gat_layer_2 = components.GATLayer(in_features=1024,
                                               out_features=256,
                                               num_heads=4)
        self.gat_layer_3 = components.GATLayer(in_features=1024,
                                               out_features=num_classes,
                                               num_heads=6,
                                               is_final_layer=True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat_layer_1(x, edge_index)
        x = F.elu(x)
        x1 = self.gat_layer_2(x, edge_index)
        x = F.elu(x1) + x # Skip connection
        x = self.gat_layer_3(x, edge_index)
        return torch.sigmoid(x)

    def training_step(self, data, batch_idx):
        out = self(data)
        loss = F.binary_cross_entropy(out, data.y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, data, batch_idx):
        out = self(data)
        loss = F.binary_cross_entropy(out, data.y)
        self.log("val_loss", loss)
        pred = torch.round(out)
        acc = f1_score(data.y.cpu(), pred.cpu(), average="micro")
        self.log("val_acc", acc)

    def test_step(self, data, batch_idx):
        out = self(data)
        pred = torch.round(out)
        acc = f1_score(data.y.cpu(), pred.cpu(), average="micro")
        self.log("test_acc", acc)
