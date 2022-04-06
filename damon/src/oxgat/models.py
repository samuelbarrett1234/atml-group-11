"""Defines various model architectures, all of which provide built-in training
(implementing an `AbstractModel` interface which extends the standard
PyTorch Lightning module).
"""
from abc import ABC, abstractmethod

import pytorch_lightning as pl
from sklearn.metrics import f1_score, roc_auc_score
import torch
import torch.nn.functional as F
import torch_geometric.loader
from typing import Type, Union, List, Optional

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


class _BaseGATModel(AbstractModel):
    """Semi-abstract base class for inductive and transductive models defined in
    the original GAT paper.
    
    See superclass and subclasses for documentation; this
    class is for internal use only to minimise code duplication.
    """
    def __init__(self, lr, regularisation=0, train_batch_size=1):
        super().__init__()
        self.lr = lr
        self.regularisation = regularisation
        self.train_batch_size = train_batch_size
        self.trainer = None
        self.checkpointer: pl.callbacks.ModelCheckpoint = None

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
        train_loader = torch_geometric.loader.DataLoader(
            train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True)
        val_loader = torch_geometric.loader.DataLoader(val_dataset)
        self.trainer.fit(self, train_dataloaders=train_loader,
                         val_dataloaders=val_loader)
        # Restore best weights and validate
        self.trainer = pl.Trainer(**self.trainer_args)
        self.trainer.validate(self, val_loader, ckpt_path="best")

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
        progress_bar = pl.callbacks.RichProgressBar()
        early_stopping = utils.MultipleEarlyStopping(
            monitors=["val_acc","val_loss"],
            modes=["max","min"],
            patience=100,
            verbose=False
        )
        trainer_args = {"max_epochs": 100000,
                        "log_every_n_steps": 1,
                        "callbacks": [early_stopping,
                                      self.checkpointer,
                                      progress_bar]}
        if use_gpu:
            trainer_args["gpus"] = 1
        self.trainer_args = trainer_args
        self.trainer = pl.Trainer(**trainer_args)


class TransductiveGATModel(_BaseGATModel):
    """Implementation of the transductive model defined in the original GAT paper.

    Parameters
    ----------
    in_features : int
        The number of features per node in the input data.
    num_classes : int
        The number of classes for node classification.
    pubmed : bool, optional
        Whether to apply the architecture/training changes made for the PubMed
        dataset in the original paper. Defaults to False.
    citeseer : bool, optional
        Whether to apply the architecture/training changes made for the PubMed
        dataset in the original paper. Defaults to False. Mutually exclusive
        with `pubmed`.
    **kwargs
        Keyword arguments to be supplied to the attention layers.
    """
    def __init__(self, in_features: int, num_classes: int,
                 pubmed: bool = False, citeseer: bool = False,
                 **kwargs):
        assert not (pubmed and citeseer)
        super().__init__(lr=0.01 if pubmed else 0.005,
                         regularisation=0.001 if pubmed else 0.0005)
        self.gat_layer_1 = components.MultiHeadAttentionLayer(
            attention_type=components.GATAttentionHead,
            in_features=in_features,
            out_features=8,
            num_heads=8,
            attention_dropout=0.6,
            **kwargs)
        self.gat_layer_2 = components.MultiHeadAttentionLayer(
            attention_type=components.GATAttentionHead,
            in_features=64,
            out_features=num_classes,
            num_heads=8 if pubmed else 1,
            is_final_layer=True,
            attention_dropout=0.6,
            **kwargs)
        self.checkpointer = pl.callbacks.ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            save_weights_only=True
        ) if citeseer else pl.callbacks.ModelCheckpoint(
            monitor="val_loss", # Included acc in original paper for Cora
            mode="min",
            save_weights_only=True)

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
        acc = int(correct) / int(data.val_mask.sum())
        self.log("val_acc", acc)

    def test_step(self, data, batch_idx):
        out = self(data)
        pred = out.argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())
        self.log("test_acc", acc)


class InductiveGATModel(_BaseGATModel):
    """Implementation of the inductive model defined in the original GAT paper.

    Parameters
    ----------
    in_features : int
        The number of features per node in the input data.
    num_classes : int
        The number of classes for node classification.
    restore_loss : bool = True
        Whether to restore to the best loss (as opposed to score).
    **kwargs
        Keyword arguments to be supplied to the attention layers.
    """
    def __init__(self, in_features: int, num_classes: int, restore_loss: bool = True, **kwargs):
        super().__init__(lr=0.005, train_batch_size=2)
        self.gat_layer_1 = components.MultiHeadAttentionLayer(
            attention_type=components.GATAttentionHead,
            in_features=in_features,
            out_features=256,
            num_heads=4,
            **kwargs)
        self.gat_layer_2 = components.MultiHeadAttentionLayer(
            attention_type=components.GATAttentionHead,
            in_features=1024,
            out_features=256,
            num_heads=4,
            **kwargs)
        self.gat_layer_3 = components.MultiHeadAttentionLayer(
            attention_type=components.GATAttentionHead,
            in_features=1024,
            out_features=num_classes,
            num_heads=6,
            is_final_layer=True,
            **kwargs)
        self.checkpointer = pl.callbacks.ModelCheckpoint(monitor="val_loss" if restore_loss else "val_acc",
                                                         mode="min" if restore_loss else "max",
                                                         save_weights_only=True)

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


class CustomNodeClassifier(AbstractModel):
    def __init__(
            self,
            in_features: int,
            num_classes: int,
            num_layers: int = 2,
            heads_per_layer: Union[int, List[int]] = [8,1],
            hidden_feature_dims: Union[int, List[int]] = 8,
            layer_type: Type[torch.nn.Module] = components.MultiHeadAttentionLayer,
            attention_type: Type[components.AbstractAttentionHead] = components.GATAttentionHead,
            dropout: float = 0.6,
            learning_rate: float = 0.005,
            regularisation: float = 0.0005,
            restore_best: str = "loss",
            batch_size: Optional[int] = None,
            mode: str = "transductive",
            sampling: bool = False,
            sampling_neighbors: int = -1, # All
            loader_num_workers: int = 0,
            early_stopping_patience: int = 100,
            max_epochs: int = 2000,
            **kwargs):
        super().__init__()
        if isinstance(heads_per_layer, int):
            heads_per_layer = [heads_per_layer]*num_layers
        if isinstance(hidden_feature_dims, int):
            hidden_feature_dims = [hidden_feature_dims]*(num_layers-1)
        if batch_size is None:
            batch_size = 128 if sampling else 1
        assert len(heads_per_layer) == num_layers
        assert len(hidden_feature_dims) == num_layers-1
        assert restore_best in ["loss", "acc"]
        assert mode in ["transductive", "inductive"]
        if sampling:
            assert mode == "transductive"

        self.layers = torch.nn.ModuleList([
            layer_type(attention_type=attention_type,
                       in_features=(in_features if i==0
                                    else heads_per_layer[i-1]*
                                        hidden_feature_dims[i-1]),
                       out_features=(num_classes if i==num_layers-1
                                     else hidden_feature_dims[i]),
                       num_heads=heads_per_layer[i],
                       is_final_layer=(i==num_layers-1),
                       attention_dropout=dropout,
                       **kwargs)
            for i in range(num_layers)])

        self.checkpointer = pl.callbacks.ModelCheckpoint(
            monitor=f"val_{restore_best}",
            mode=("min" if restore_best=="loss" else "max"))
        self.learning_rate = learning_rate
        self.regularisation = regularisation
        self.transductive = (mode == "transductive")
        self.sampling = sampling
        self.sampling_neighbors = sampling_neighbors
        self.batch_size = batch_size
        self.dropout=dropout
        self.loader_num_workers = loader_num_workers
        self.early_stopping_patience = early_stopping_patience
        self.max_epochs = max_epochs

        self.final_metrics = {}
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, layer in enumerate(self.layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x, edge_index)
            if i < len(self.layers)-1:
                x = F.elu(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, data, batch_idx):
        mask = (data.train_mask if self.transductive
                else torch.ones(data.num_nodes, dtype=torch.bool))
        out = self(data)
        loss = F.nll_loss(out[mask], data.y[mask])
        self.log("train_loss", loss, batch_size=int(mask.sum()))
        return loss

    def validation_step(self, data, batch_idx):
        mask = (data.val_mask if self.transductive
                else torch.ones(data.num_nodes, dtype=torch.bool))
        out = self(data)
        loss = F.nll_loss(out[mask], data.y[mask])
        self.log("val_loss", loss, batch_size=int(mask.sum()))
        pred = out.argmax(dim=1)
        correct = (pred[mask] == data.y[mask]).sum()
        acc = int(correct) / int(mask.sum())
        self.log("val_acc", acc, batch_size=int(mask.sum()))

    def test_step(self, data, batch_idx):
        mask = (data.test_mask if self.transductive
                else torch.ones(data.num_nodes, dtype=torch.bool))
        out = self(data)
        pred = out.argmax(dim=1)
        correct = (pred[mask] == data.y[mask]).sum()
        acc = int(correct) / int(mask.sum())
        self.log("test_acc", acc, batch_size=int(mask.sum()))
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.learning_rate,
                                weight_decay=self.regularisation)

    def standard_train(self, train_dataset, val_dataset=None, use_gpu=False):
        """Automated training of this model."
        """
        self._init_trainer(use_gpu)
        if val_dataset is None:
            assert self.transductive
            val_dataset = train_dataset
        if self.transductive:
            assert len(train_dataset) == 1
        train_loader = (torch_geometric.loader.NeighborLoader(
                train_dataset[0],
                num_neighbors=[self.sampling_neighbors]*len(self.layers),
                batch_size=self.batch_size,
                num_workers=self.loader_num_workers) 
            if self.sampling
            else torch_geometric.loader.DataLoader(train_dataset,
                                                   batch_size=self.batch_size,
                                                   num_workers=self.loader_num_workers))
        val_loader = torch_geometric.loader.DataLoader(val_dataset,
                                                       batch_size=self.batch_size,
                                                       num_workers=self.loader_num_workers)
        self.trainer.fit(self,
                         train_dataloaders=train_loader,
                         val_dataloaders=val_loader)
        # Restore best weights and validate
        self.final_metrics["end_train_loss"] = float(self.trainer.callback_metrics["train_loss"])
        self.final_metrics["end_val_loss"] = float(self.trainer.callback_metrics["val_loss"])
        self.final_metrics["end_val_acc"] = float(self.trainer.callback_metrics["val_acc"])
        self.final_metrics["epochs_trained"] = self.current_epoch+1
        self.trainer = pl.Trainer(**self.trainer_args)
        self.trainer.validate(self, val_loader, ckpt_path="best")
        self.final_metrics["restored_val_loss"] = float(self.trainer.callback_metrics["val_loss"])
        self.final_metrics["restored_val_acc"] = float(self.trainer.callback_metrics["val_acc"])
        self.final_metrics["restored_epoch_number"] = self.current_epoch

    def standard_test(self, dataset):
        """Method to test this model after having run `self.standard_train()`.
        """
        assert self.trainer is not None, "Must run `self.standard_train()` first."
        dataloader = torch_geometric.loader.DataLoader(dataset)
        self.trainer.test(self, dataloader)
        self.final_metrics["test_acc"] = float(self.trainer.callback_metrics["test_acc"])

    # Initialize the trainer for use in self.train()
    def _init_trainer(self, use_gpu):
        # Stop training only if neither validation loss nor accuracy has
        # improved in last 100 epochs.
        progress_bar = pl.callbacks.RichProgressBar()
        early_stopping = utils.MultipleEarlyStopping(
            monitors=["val_acc","val_loss"],
            modes=["max","min"],
            patience=self.early_stopping_patience,
            verbose=False
        )
        trainer_args = {"max_epochs": self.max_epochs,
                        "log_every_n_steps": 1,
                        "callbacks": [early_stopping,
                                      self.checkpointer,
                                      progress_bar]}
        if use_gpu:
            trainer_args["gpus"] = 1
        self.trainer_args = trainer_args
        self.trainer = pl.Trainer(**trainer_args)


class CustomGraphClassifier(AbstractModel):
    def __init__(
            self,
            in_features: int,
            num_classes: int,
            num_attention_layers: int = 4,
            heads_per_layer: Union[int, List[int]] = 2,
            hidden_feature_dims: Union[int, List[int]] = 16,
            layer_type: Type[torch.nn.Module] = components.MultiHeadAttentionLayer,
            attention_type: Type[components.AbstractAttentionHead] = components.GATAttentionHead,
            num_mlp_hidden_layers: int = 1,
            mlp_hidden_layer_dims: Optional[Union[int, List[int]]] = None,
            dropout: float = 0.6,
            learning_rate: float = 0.005,
            regularisation: float = 0.0005,
            restore_best: str = "loss",
            batch_size: int = 128,
            pooling: str = "mean",
            loader_num_workers: int = 4,
            cast_to_float = False,
            early_stopping_patience: int = 100,
            metric: str = "accuracy",
            add_pos: bool = False,
            **kwargs):
        super().__init__()
        if isinstance(heads_per_layer, int):
            heads_per_layer = [heads_per_layer]*num_attention_layers
        if isinstance(hidden_feature_dims, int):
            hidden_feature_dims = [hidden_feature_dims]*num_attention_layers
        if mlp_hidden_layer_dims is None:
            mlp_hidden_layer_dims = hidden_feature_dims[-1]
        if isinstance(mlp_hidden_layer_dims, int):
            mlp_hidden_layer_dims = [mlp_hidden_layer_dims]*num_mlp_hidden_layers
        assert len(heads_per_layer) == num_attention_layers
        assert len(hidden_feature_dims) == num_attention_layers
        assert len(mlp_hidden_layer_dims) == num_mlp_hidden_layers
        assert restore_best in ["loss", "acc"]
        assert pooling in ["mean", "max", "sum"]
        assert metric in ["accuracy", "roc_auc_score"]
        if metric == "roc_auc_score":
            assert num_classes == 2

        self.attention_layers = torch.nn.ModuleList([
            layer_type(attention_type=attention_type,
                       in_features=(in_features if i==0
                                    else heads_per_layer[i-1]*
                                        hidden_feature_dims[i-1]),
                       out_features=hidden_feature_dims[i],
                       num_heads=heads_per_layer[i],
                       is_final_layer=(i==num_attention_layers-1),
                       attention_dropout=dropout,
                       **kwargs)
            for i in range(num_attention_layers)])

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_feature_dims[-1],
                            mlp_hidden_layer_dims[0]),
            *[layer for i in range(num_mlp_hidden_layers)
                    for layer in (torch.nn.ReLU(),
                                  torch.nn.Linear(mlp_hidden_layer_dims[i],
                                                  (num_classes
                                                   if i == num_mlp_hidden_layers-1
                                                   else mlp_hidden_layer_dims[i+1])))
             ])
        self.pool = {
            "mean": torch_geometric.nn.global_mean_pool,
            "max": torch_geometric.nn.global_max_pool,
            "sum": torch_geometric.nn.global_add_pool,
        }[pooling]

        self.checkpointer = pl.callbacks.ModelCheckpoint(
            monitor=f"val_{restore_best}",
            mode=("min" if restore_best=="loss" else "max"),
            save_weights_only=True)
        self.learning_rate = learning_rate
        self.regularisation = regularisation
        self.batch_size = batch_size
        self.dropout=dropout
        self.loader_num_workers = loader_num_workers
        self.cast_to_float = cast_to_float
        self.early_stopping_patience = early_stopping_patience
        self.use_roc = (metric == "roc_auc_score")
        self.add_pos = add_pos
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.add_pos: # For MNIST/CIFAR
            x = torch.cat([x, data.pos], dim=-1)
        if self.cast_to_float:
            x = x.float()
        for layer in self.attention_layers:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x, edge_index)
            x = F.elu(x)
        out = self.pool(x, batch=data.batch)
        out = self.mlp(out)
        return F.log_softmax(out, dim=1)

    def training_step(self, data, batch_idx):
        out = self(data)
        loss = F.nll_loss(out, data.y.flatten())
        self.log("train_loss", loss, batch_size=data.num_graphs)
        return loss

    def validation_step(self, data, batch_idx):
        out = self(data)
        loss = F.nll_loss(out, data.y.flatten())
        self.log("val_loss", loss, batch_size=data.num_graphs)
        if self.use_roc:
            acc = roc_auc_score(data.y.flatten().cpu(), out[...,1].cpu())
        else:
            pred = out.argmax(dim=1)
            correct = (pred == data.y.flatten()).sum()
            acc = int(correct) / data.num_graphs
        self.log("val_acc", acc, batch_size=data.num_graphs)

    def test_step(self, data, batch_idx):
        out = self(data)
        if self.use_roc:
            acc = roc_auc_score(data.y.flatten().cpu(), out[...,1].cpu())
        else:
            pred = out.argmax(dim=1)
            correct = (pred == data.y.flatten()).sum()
            acc = int(correct) / data.num_graphs
        self.log("test_acc", acc, batch_size=data.num_graphs)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.learning_rate,
                                weight_decay=self.regularisation)

    def standard_train(self, train_dataset, val_dataset, use_gpu=False):
        """Automated training of this model."
        """
        self._init_trainer(use_gpu)
        train_loader = torch_geometric.loader.DataLoader(train_dataset,
                                                         batch_size=self.batch_size,
                                                         num_workers=self.loader_num_workers,
                                                         shuffle=True)
        val_loader = torch_geometric.loader.DataLoader(val_dataset,
                                                       num_workers=self.loader_num_workers,
                                                       batch_size=val_dataset.len())
        self.trainer.fit(self,
                         train_dataloaders=train_loader,
                         val_dataloaders=val_loader)
        # Restore best weights and validate
        self.trainer = pl.Trainer(**self.trainer_args)
        self.trainer.validate(self, val_loader, ckpt_path="best")

    def standard_test(self, dataset):
        """Method to test this model after having run `self.standard_train()`.
        """
        assert self.trainer is not None, "Must run `self.standard_train()` first."
        dataloader = torch_geometric.loader.DataLoader(dataset, batch_size=dataset.len())
        self.trainer.test(self, dataloader)

    # Initialize the trainer for use in self.train()
    def _init_trainer(self, use_gpu):
        # Stop training only if neither validation loss nor accuracy has
        # improved in last 100 epochs.
        progress_bar = pl.callbacks.RichProgressBar()
        early_stopping = utils.MultipleEarlyStopping(
            monitors=["val_acc","val_loss"],
            modes=["max","min"],
            patience=self.early_stopping_patience,
            verbose=False
        )
        trainer_args = {"max_epochs": 100000,
                        "log_every_n_steps": 1,
                        "callbacks": [early_stopping,
                                      self.checkpointer,
                                      progress_bar]}
        if use_gpu:
            trainer_args["gpus"] = 1
        self.trainer_args = trainer_args
        self.trainer = pl.Trainer(**trainer_args)
