import os
import csv
import glob
import json
import argparse as ap
from functools import partial
import torch
import torch.multiprocessing
from tqdm import tqdm
import experiments


# a mapping from dataset names to the class
# which purports to execute the experiment
# (note: this mapping will not necessarily be
# injective.)
EXPERIMENT_CLASS = {
    'cora' : (experiments.utils.load_model,
              experiments.transductive.load_dataset,
              experiments.transductive.train_model),
    'ppi' : (experiments.utils.load_model,
             experiments.inductive.load_dataset,
             experiments.inductive.train_model)
}


def move_to_device(device, x):
    """This function is to help move data, particularly
    that which is yielded from a DataLoader, to the right
    device.

    This function allows `x` to be a range of argument
    types, feel free to extend if necessary, but will
    raise a NotImplementedError if it doesn't know what
    to do.
    """
    if hasattr(x, 'to'):
        return x.to(device)

    elif isinstance(x, tuple):
        return tuple(map(partial(move_to_device, device), x))

    elif isinstance(x, list):
        return list(map(partial(move_to_device, device), x))

    elif isinstance(x, torch.utils.data.DataLoader):
        # automatically transfer data to the right device as it is yielded
        # from the data loader
        class AutoTransferringDataLoader:
            def __init__(self, device, dl):
                self.dl = dl
                self.device = device

            def __iter__(self):
                return map(partial(move_to_device, self.device), iter(self.dl))

        return AutoTransferringDataLoader(device, x)

    else:
        raise NotImplementedError(f"Don't know how to move data to device: {x}")


class Experiment:
    """Perform a transductive/inductive experiment on the given
    dataset, using the model specified in the given config, outputting
    results to the given log file, and saving the best version
    of the model to the given filename. The device to train on
    is specified using `device`.

    Use `run` to launch an instance of training. You can call
    this multiple times if you want to train several independent
    models and keep only the best one.

    `model_loader_function` should take three arguments as input:
    the configuration, the input dimension, and the number of
    classes. It should return a 2-tuple (torch model, training config).

    `dataset_loader_function` should take the dataset name as input
    and return a tuple containing (i) the input dimension, (ii)
    the number of classes, (iii) any number of tensors.

    `model_trainer_function` should take as input (i) an argument
    `model` passed as a kwarg; (ii) an argument `train_cfgs` also
    passed as a kwarg containing the training configuration; and
    (iii) all of the *tensor* outputs of `load_dataset`, in the
    same order. It should yield the parameters for logging.
    """
    def __init__(self, device, dataset_name,
                 config, log_filename, out_model_filename,
                 model_loader_function,
                 dataset_loader_function,
                 model_trainer_function):
        self.dsname = dataset_name
        self.device = device
        self.cfg = config
        self.log_filename = log_filename
        self.model_filename = out_model_filename
        self.best_val_score = 0.0
        self.cur_model_idx = 0
        self._tag = config.get("tag", "<NO-TAG>")
        self.load_model = model_loader_function
        self.load_dataset = dataset_loader_function
        self.train_model = model_trainer_function

    def __len__(self):
        # assume that `self.train_model` respects the configuration
        # file's chosen number of training epochs. This will throw
        # if not.
        return self.cfg["train_cfg"]["max_epoch"]

    def tag(self):
        """Return the experiment's tag.
        """
        return self._tag

    def best_val(self):
        """Return best performance on validation set.
        """
        return self.best_val_score

    def run(self):
        """Train fresh model on the given dataset, return
        some statistics. For more statistics, consult the
        log file, which will be written to during this
        function. The best version of the model will be
        saved, also.
        Yields `None` per epoch (this just allows you to
        count epochs as they go by.)
        """
        with open(self.log_filename, 'w', newline='') as log_file:
            # firstly, log the config to the log file:
            json.dump(self.cfg, log_file)
            log_file.write('\n')
            logger = csv.writer(log_file)

            # load dataset
            dataset = self.load_dataset(self.dsname)
            input_dim = dataset[0]
            n_classes = dataset[1]
            # move all data to the right device if applicable
            dataset = move_to_device(self.device, dataset[2:])
            # construct model
            model, train_cfgs = self.load_model(
                self.cfg, input_dim, n_classes)
            model.to(self.device)
            # begin training
            model_train_stats = self.train_model(
                *dataset,
                model=model, train_cfgs=train_cfgs)
            # examine results
            for epoch, loss, val_score, test_score, p_att, p_hid in model_train_stats:
                # is it an improvement?
                is_improvement = (val_score > self.best_val_score)

                logger.writerow([
                    self.cur_model_idx,
                    epoch,
                    loss,
                    val_score,
                    test_score,
                    is_improvement,
                    p_att,
                    p_hid
                ])

                if is_improvement:
                    # save model and update new best
                    self.best_val_score = val_score
                    torch.save(model, self.model_filename)

                yield None

            self.cur_model_idx += 1


class MultipleExperimentRunner:
    """Runs multiple instances of `Experiment`, done
    by iterating over the runner. Also has a length
    estimator, so you can gauge how long it will take.
    """
    def __init__(self, experiments):
        if not isinstance(experiments, list):  # materialise list
            experiments = list(experiments)
        self.exprs = experiments
        self.iterable = iter(self.exprs[0].run())  # current experiment

    def __len__(self):
        return sum(map(len, self.exprs))

    def __iter__(self):
        return self

    def __next__(self):
        try:
            # try to carry on with current experiment
            return next(self.iterable)
        except StopIteration:  # if current experiment finishes
            del self.exprs[0]
            if len(self.exprs) == 0:  # if there are no more experiments
                raise StopIteration()
            else:  # else begin next experiment
                self.iterable = iter(self.exprs[0].run())
                return next(self.iterable)


def load_experiments(config_filenames, device, dataset):
    for fname in config_filenames:
        # place log and saved model next to config
        base = os.path.splitext(fname)[0]
        log_fname = base + ".log"
        model_fname = base + ".pt"

        with open(fname, "r") as cfg_f:
            cfg = json.load(cfg_f)

        yield Experiment(
            device, dataset, cfg, log_fname, model_fname,
            *EXPERIMENT_CLASS[dataset]
        )


def run(i, args):
    # device `i + 1` will run on configs in range
    # [ config_step * i, config_step * (i + 1) )
    # and device 0 will run on the remainder
    config_step = len(args.config) // len(args.device)

    # determine device and configs for this run process
    device = torch.device(args.device[i])
    if i > 0:
            args.config = args.config[
                config_step * (i - 1) : config_step * i
            ]
    else:
        args.config = args.config[config_step * (len(args.device) - 1):]

    if not args.quiet:
        print("Running on", len(args.config), "config files...")

    if len(args.config) == 0:
        print("Warning: worker", i, "was assigned no jobs.")
        return None

    # load experiments, then pass them to the MultipleExperimentRunner
    runner = MultipleExperimentRunner(load_experiments(args.config, device, args.dataset))
    if not args.quiet:
        runner = tqdm(runner)
    # iterate through the runner to run all of the experiments
    for _ in runner:
        pass
    # done!


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("dataset", type=str,
                        help="Name of dataset, e.g. 'cora' or 'ppi', to train on.")
    parser.add_argument("config", type=str, nargs='+',
                        help="Path to config file. Can supply many configs.")
    parser.add_argument("--quiet", action="store_true",
                        help="Set this to produce no output.")
    parser.add_argument("--device", type=str, action='append',
                        help="Set torch device, e.g. cuda:0 or cpu.")
    args = parser.parse_args()

    if args.dataset not in EXPERIMENT_CLASS:
        print("Unrecognised dataset. Allowed values are:",
              ", ".join(EXPERIMENT_CLASS.keys()))
        exit(1)

    # allow recursive glob
    args.config = [file for pat in args.config for file in glob.glob(pat, recursive=True)]

    # check they all exist
    if not all(map(os.path.isfile, args.config)):
        print("Cannot find some or all config files.")
        exit(1)

    if len(args.config) == 0:
        print("Error: No config files detected.")
        exit(1)

    if args.device is None:
        args.device = ['cpu']  # default device

    # for multiple devices, parallelise across them
    if len(args.device) > 1:
        torch.multiprocessing.spawn(run, (args,), nprocs=len(args.device))
    else:
        run(0, args)
