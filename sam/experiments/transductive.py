import torch
import torch.nn as nn
import torch.optim as optim 
import torch_geometric as tg
from experiments.utils import laplacian_pos_emb, flip_pos_embs


def load_dataset(dsname):
    """Load dataset with given name.
    Returns a big tuple of values.
    """
    if dsname == 'cora':
        dataset = tg.datasets.Planetoid(root='data', name='Cora', split='public',
                                        transform=tg.transforms.NormalizeFeatures())
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


def train_model(nodes, labels, adjacency_matrix,
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

    if "step_lr" in train_cfgs:
        sched = optim.lr_scheduler.StepLR(
            optimiser, train_cfgs["step_lr"]["step"],
            gamma=train_cfgs["step_lr"]["gamma"])
    else:
        sched = None

    if hasattr(model, 'pos_emb_dim'):
        # compute positional embeddings for whole graph upfront
        # (do it on the CPU - torch bug)
        pos_embs = laplacian_pos_emb(adjacency_matrix.to(
            torch.device('cpu')),
            model.pos_emb_dim).to(adjacency_matrix.device)

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
        if hasattr(model, 'pos_emb_dim'):
            output = model(nodes, adjacency_matrix, flip_pos_embs(pos_embs))
        else:
            output = model(nodes, adjacency_matrix)
        loss = criterion(output[train_mask], labels[train_mask])
        
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if sched is not None:
            sched.step()

        # compute validation and testing accuracies
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'pos_emb_dim'):
                output = model(nodes, adjacency_matrix, flip_pos_embs(pos_embs))
            else:
                output = model(nodes, adjacency_matrix)
            val_acc = (output[val_mask].argmax(dim=1) ==
                labels[val_mask]).sum().item() / val_mask.sum().item()
            test_acc = (output[test_mask].argmax(dim=1) ==
                labels[test_mask]).sum().item() / test_mask.sum().item()

        yield (epoch, float(loss), float(val_acc),
               float(test_acc), p_att, p_hid)
