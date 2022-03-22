import os
import glob
import json
import math
import collections
import argparse as ap
from tqdm import tqdm
from experiments import TransductiveExperiment


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("dataset", type=str,
                        help="Name of dataset, e.g. 'cora', to train on.")
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

    # store the best performance seen for each tag
    # (the keys are tags; the values are pairs (validation-acc, config-name).)
    best_for_tag = collections.defaultdict(lambda: (-math.inf, None))

    for fname in tqdm(args.config):
        # place log and saved model next to config
        base = os.path.splitext(fname)[0]
        log_fname = base + ".log"
        model_fname = base + ".pt"

        with open(fname, "r") as cfg_f:
            cfg = json.load(cfg_f)
        
        with open(log_fname, "a", newline='') as log_f:
            expr = TransductiveExperiment(args.dataset, cfg, log_f, model_fname)
            expr.run()
            if best_for_tag[expr.tag()][0] < expr.best_val():
                best_for_tag[expr.tag()] = (expr.best_val(), fname)

    # print results
    print("Best validation accuracies achieved:")
    for name, val_acc_model_fname in best_for_tag.items():
        print(name, '\t', val_acc_model_fname[0], '\t', val_acc_model_fname[1])
