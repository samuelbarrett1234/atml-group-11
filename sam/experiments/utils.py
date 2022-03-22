import math


def anneal_dropout(start, time, init, inc):
    """Compute the value of dropout, if you are starting
    at epoch `start`, you want the dropout to increase
    for `time` epochs, starting at a value of `init`,
    increasing by a total of `inc`.
    """
    assert(init + inc <= 1.0)
    return lambda epoch: (
        math.sin(min(max(epoch - start, 0.0), time) / time
                 * math.pi / 2.0) * inc + init)


def extract_train_configs(cfg):
    """Extract the section of the model config to do with training.
    """
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

    return train_cfgs
