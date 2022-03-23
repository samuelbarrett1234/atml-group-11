import os
import sys
import math
sys.path.append(os.environ['PATH_TO_GAT'])
from gat import VanillaTransformer, UniversalTransformer


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


def load_model(cfg, input_dim, n_classes):
    """Loads model from given configuration.
    See `configs` folder for example model configurations.
    """
    assert("type" in cfg)
    kwargs = cfg.get("model_kwargs", {})
    kwargs["input_dim"] = input_dim
    kwargs["num_classes"] = n_classes
    if cfg["type"] == "vanilla":
        model = VanillaTransformer(**kwargs)
    elif cfg["type"] == "universal":
        model = UniversalTransformer(**kwargs)
    else:
        raise NotImplementedError(f"Model type {cfg['type']} not supported.")

    train_cfgs = extract_train_configs(cfg)
    return model, train_cfgs
