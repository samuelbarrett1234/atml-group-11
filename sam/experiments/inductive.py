import torch
import torch.nn as nn
import torch.optim as optim 
import torch_geometric as tg
import sklearn.metrics as metrics


def load_dataset(dsname):
    """Load dataset with given name.
    Returns a big tuple of values.
    """
    if dsname == 'ppi':
        train_dataset = tg.datasets.PPI(root='data', split='train')
        val_dataset = tg.datasets.PPI(root='data', split='val')
        test_dataset = tg.datasets.PPI(root='data', split='test')

        train_loader = tg.loader.DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = tg.loader.DataLoader(val_dataset, batch_size=2)
        test_loader = tg.loader.DataLoader(test_dataset, batch_size=2)

        # for simplicity later on we just cache those
        val_graph_pair = next(iter(val_loader))
        test_graph_pair = next(iter(test_loader))

        nodes_val = val_graph_pair.x
        y_val = val_graph_pair.y
        adjacency_val = tg.utils.to_dense_adj(val_graph_pair.edge_index).squeeze(dim=0)

        nodes_test = test_graph_pair.x
        y_test = test_graph_pair.y
        adjacency_test = tg.utils.to_dense_adj(test_graph_pair.edge_index).squeeze(dim=0)

        # 50: input dimension; 121: number of classes
        return (50, 121, train_loader,
            nodes_val, y_val, adjacency_val,
            nodes_test, y_test, adjacency_test)

    else:
        raise NotImplementedError(f"Dataset {dsname} is not implemented.")


def train_model(train_loader,
                nodes_val, y_val, adjacency_val,
                nodes_test, y_test, adjacency_test,
                model, train_cfgs):
    """Train given model on given dataset, according to some
    settings as specified in `train_cfgs`. Yields a tuple of
    of statistics about training, should you wish to log
    them.
    """
    # required to move data back to cpu (if applicable)
    # before invoking sklearn
    cpu_device = torch.device('cpu')
    # we're not going to need the following items on the GPU
    # so transfer them right away
    y_val = y_val.to(cpu_device)
    y_test = y_test.to(cpu_device)

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

        # train model for one epoch
        model.train()
        for train_graph_pair in train_loader:
            nodes_train = train_graph_pair.x
            y_train = train_graph_pair.y
            adjacency_train = tg.utils.to_dense_adj(
                train_graph_pair.edge_index,
                max_num_nodes=nodes_train.shape[0]).squeeze(dim=0)

            output = model(nodes_train, adjacency_train)
            loss = criterion(output, y_train)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        sched.step()

        # compute validation and testing accuracies
        model.eval()
        with torch.no_grad():
            # val
            output = model(nodes_val, adjacency_val)
            output_labelled = torch.where(output > 0.5, 1.0, 0.0).to(cpu_device)
            val_f1 = metrics.f1_score(output_labelled, y_val, average='micro')
            # test
            output = model(nodes_test, adjacency_test)
            output_labelled = torch.where(output > 0.5, 1.0, 0.0).to(cpu_device)
            test_f1 = metrics.f1_score(output_labelled, y_test, average='micro')

        yield (epoch, float(loss), float(val_f1),
               float(test_f1), p_att, p_hid)
