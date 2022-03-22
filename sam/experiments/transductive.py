import os
import sys
import csv
import json
import torch
import torch.nn as nn
import torch.optim as optim 
import torch_geometric as tg
from experiments.utils import extract_train_configs
sys.path.append(os.environ['PATH_TO_GAT'])
from gat import VanillaTransformer_Transductive, UniversalTransformer_Transductive


def load_model(cfg, input_dim, n_classes):
    """Loads model from given configuration.
    See `configs` folder for example model configurations.
    """
    assert("type" in cfg)
    kwargs = cfg.get("model_kwargs", {})
    kwargs["input_dim"] = input_dim
    kwargs["num_classes"] = n_classes
    if cfg["type"] == "vanilla":
        model = VanillaTransformer_Transductive(**kwargs)
    elif cfg["type"] == "universal":
        model = UniversalTransformer_Transductive(**kwargs)
    else:
        raise NotImplementedError(f"Model type {cfg['type']} not supported.")

    train_cfgs = extract_train_configs(cfg)
    return model, train_cfgs


def load_dataset(dsname):
    """Load dataset with given name.
    Returns a big tuple of values.
    """
    if dsname == 'cora':
        dataset = tg.datasets.Planetoid(root='data', name='Cora', split='full')
        cora_dataloader = tg.loader.DataLoader(dataset)
        cora_graph = next(iter(cora_dataloader))

        nodes = cora_graph.x
        labels = cora_graph.y
        adjacency_matrix = tg.utils.to_dense_adj(cora_graph.edge_index).squeeze(dim=0)

        train_mask = cora_graph.train_mask
        val_mask = cora_graph.val_mask
        test_mask = cora_graph.test_mask

        input_dim, num_classes = 1433, 7

        return (input_dim, num_classes,
            nodes, labels, adjacency_matrix,
            train_mask, val_mask, test_mask)

    else:
        raise NotImplementedError(f"Dataset {dsname} is not implemented.")


def train_model(nodes, adjacency_matrix, labels,
                train_mask, val_mask, test_mask,
                model, train_cfgs):
    """Train given model on given dataset, according to some
    settings as specified in `train_cfgs`. Yields a tuple of
    of statistics about training, should you wish to log
    them.
    """
    # set up training
    criterion = nn.CrossEntropyLoss()

    optimiser = optim.Adam(
        model.parameters(), 
        lr=train_cfgs["learning_rate"],
        weight_decay=train_cfgs["weight_decay"])

    sched = optim.lr_scheduler.StepLR(
        optimiser, train_cfgs["step_lr"]["step"],
        gamma=train_cfgs["step_lr"]["gamma"])

    # begin training
    for epoch in range(train_cfgs["max_epoch"]):
        # if annealing dropout, update it
        if not isinstance(train_cfgs.get("dropout_attention", float()), float):
            p_att = train_cfgs["dropout_attention"](epoch)
            model.anneal_attention_dropout(p_att)
        else:
            p_att = train_cfgs.get("dropout_attention", None)
        if not isinstance(train_cfgs.get("dropout_hidden", float()), float):
            p_hid = train_cfgs["dropout_hidden"](epoch)
            model.anneal_hidden_dropout(p_hid)
        else:
            p_hid = train_cfgs.get("dropout_hidden", None)

        # train model for one step
        model.train()
        output = model(nodes, adjacency_matrix)
        loss = criterion(output[train_mask], labels[train_mask])
        
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        sched.step()

        # compute validation and testing accuracies
        model.eval()
        with torch.no_grad():
            output = model(nodes, adjacency_matrix)
            val_acc = (output[val_mask].argmax(dim=1) ==
                labels[val_mask]).sum().item() / val_mask.sum().item()
            test_acc = (output[test_mask].argmax(dim=1) ==
                labels[test_mask]).sum().item() / test_mask.sum().item()

        yield (epoch, float(loss), float(val_acc),
               float(test_acc), p_att, p_hid)


class TransductiveExperiment:
    """Perform a transductive experiment on the given dataset,
    using the model specified in the given config, outputting
    results to the given log file, and saving the best version
    of the model to the given filename. The device to train on
    is specified using `device`.

    Use `run` to launch an instance of training. You can call
    this multiple times if you want to train several independent
    models and keep only the best one.
    """
    def __init__(self, device, dataset, config, log_file, out_model_filename):
        self.dsname = dataset
        self.device = device
        self.cfg = config
        # firstly, log the config to the log file:
        json.dump(self.cfg, log_file)
        log_file.write('\n')
        self.logger = csv.writer(log_file)
        self.model_filename = out_model_filename
        self.best_val_acc = 0.0
        self.cur_model_idx = 0
        self._tag = config.get("tag", "<NO-TAG>")

    def tag(self):
        """Return the experiment's tag.
        """
        return self._tag

    def best_val(self):
        """Return best performance on validation set.
        """
        return self.best_val_acc

    def run(self):
        """Train fresh model on the given dataset, return
        some statistics. For more statistics, consult the
        log file, which will be written to during this
        function. The best version of the model will be
        saved, also.
        """
        # load dataset
        (input_dim, n_classes,
         nodes, labels, adjacency_matrix,
         train_mask, val_mask, test_mask) = load_dataset(self.dsname)
        # move all data to the right device
        nodes = nodes.to(self.device)
        labels = labels.to(self.device)
        adjacency_matrix = adjacency_matrix.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        test_mask = test_mask.to(self.device)
        # construct model
        model, train_cfgs = load_model(
            self.cfg, input_dim, n_classes)
        model.to(self.device)
        # begin training
        model_train_stats = train_model(
            nodes, adjacency_matrix, labels, train_mask, val_mask,
            test_mask, model, train_cfgs)
        # examine results
        for epoch, loss, val_acc, test_acc, p_att, p_hid in model_train_stats:
            # is it an improvement?
            is_improvement = (val_acc > self.best_val_acc)

            self.logger.writerow([
                self.cur_model_idx,
                epoch,
                loss,
                val_acc,
                test_acc,
                is_improvement,
                p_att,
                p_hid
            ])

            if is_improvement:
                # save model and update new best
                self.best_val_acc = val_acc
                torch.save(model, self.model_filename + ".pt")

        self.cur_model_idx += 1
