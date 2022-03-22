import os
import sys
import csv
import glob
import json
import math
import pathlib
import collections
import argparse as ap
import torch
import torch.nn as nn
import torch_geometric as tg
import torch.optim as optim 
from tqdm import tqdm


sys.path.append(os.environ['PATH_TO_GAT'])
from gat import VanillaTransformer_Transductive, UniversalTransformer_Transductive


def anneal_dropout(start, time, init, inc):
    return lambda epoch: (
        math.sin(min(max(epoch - start, 0.0), time) / time
                 * math.pi / 2.0) * inc + init)


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

    train_cfgs = cfg.get("train_cfg", {})

    # dropout annealing
    if not isinstance(train_cfgs.get("dropout_hidden", float()), float):
        train_cfgs["dropout_hidden"] = anneal_dropout(
            train_cfgs["dropout_hidden"]["start"],
            train_cfgs["dropout_hidden"]["time"],
            train_cfgs["dropout_hidden"]["initial"],
            train_cfgs["dropout_hidden"]["inc"]
        )
    if not isinstance(train_cfgs.get("dropout_attention", float()), float):
        train_cfgs["dropout_attention"] = anneal_dropout(
            train_cfgs["dropout_attention"]["start"],
            train_cfgs["dropout_attention"]["time"],
            train_cfgs["dropout_attention"]["initial"],
            train_cfgs["dropout_attention"]["inc"]
        )

    return model, train_cfgs, cfg["out_name"]


def train_model(nodes, adjacency_matrix, labels,
                train_mask, val_mask, test_mask,
                model, train_cfgs):
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


def load_dataset(dsname):
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


class MultipleModelRunner:
    """This is a simple class which trains several models
    on a single dataset. Its main functionality is that
    it knows how long it has left, i.e. is friendly to
    progress bars. It also keeps track of the best model
    version so far for each config `out_name`.
    """
    def __init__(self, dsname, configs):
        self.dsname = dsname
        self.configs = configs
        self.generator = self._generate()
        # a mapping from model output filenames to best accuracies
        # (this map allows us to only maintain the best model over
        # different configuration files; useful if doing a grid
        # search!)
        self.best_val_accs = collections.defaultdict(lambda: 0.0)
        # compute length
        self._len = 0
        for fname in self.configs:
            with open(fname, 'r') as f:
                self._len += json.load(f)["train_cfg"]["max_epoch"]

    def __len__(self):
        return self._len

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.generator)

    def _generate(self):
        (input_dim, n_classes,
         nodes, labels, adjacency_matrix,
         train_mask, val_mask, test_mask) = load_dataset(self.dsname)

        for fname in self.configs:
            # experiment name defined to be the stem of the config name
            expr_name = pathlib.Path(fname).stem

            # load model and other training configurations
            with open(fname, 'r') as f:
                model, train_cfgs, out_model_name = load_model(
                    json.load(f), input_dim, n_classes)

            # begin model training:
            model_train_stats = train_model(
                nodes, adjacency_matrix, labels, train_mask, val_mask,
                test_mask, model, train_cfgs)

            for epoch, loss, val_acc, test_acc, p_att, p_hid in model_train_stats:
                # is it an improvement?
                is_improvement = (val_acc >
                    self.best_val_accs[out_model_name])

                yield (expr_name,
                    out_model_name,
                    epoch,
                    loss,
                    val_acc,
                    test_acc,
                    is_improvement,
                    p_att,
                    p_hid)

                if is_improvement:
                    # save model and update new best
                    self.best_val_accs[out_model_name] = val_acc
                    torch.save(model, out_model_name + ".pt")


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("dataset", type=str,
                        help="Name of dataset, e.g. 'cora', to train on.")
    parser.add_argument("out_log", type=str,
                        help="The file to output training logs to.")
    parser.add_argument("config", type=str, nargs='+',
                        help="Path to config file. Can supply many configs.")
    args = parser.parse_args()

    # allow recursive glob
    args.config = [file for pat in args.config for file in glob.glob(pat, recursive=True)]

    # check they all exist
    if not all(map(os.path.isfile, args.config)):
        print("Cannot find some or all config files.")
        exit(1)

    if len(args.config) == 0:
        print("Error: No config files detected.")
        exit(1)

    print("Running on", len(args.config), "config files...")

    # this is for logging training info to a file (for appending)
    log_f = open(args.out_log, "a", newline='')
    logger = csv.writer(log_f)

    # now start training the models:
    runner = MultipleModelRunner(args.dataset, args.config)
    for (expr_name, out_model_name, epoch, loss,
         val_acc, test_acc, is_improvement,
         p_att, p_hid) in tqdm(runner):

        # log training info:
        logger.writerow([
            expr_name,
            out_model_name,
            epoch,
            loss,
            val_acc,
            test_acc,
            is_improvement,
            p_att,
            p_hid
        ])


    # print results
    print("Best validation accuracies achieved:")
    for name, val_acc in runner.best_val_accs.items():
        print(name, '\t', val_acc)
