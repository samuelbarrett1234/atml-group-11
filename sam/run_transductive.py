import os
import glob
import json
import argparse as ap
import torch
from tqdm import tqdm
from experiments import TransductiveExperiment


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("dataset", type=str,
                        help="Name of dataset, e.g. 'cora', to train on.")
    parser.add_argument("config", type=str, nargs='+',
                        help="Path to config file. Can supply many configs.")
    parser.add_argument("--quiet", action="store_true",
                        help="Set this to produce no output.")
    parser.add_argument("--device", type=str, default='cpu',
                        help="Set torch device, e.g. cuda:0 or cpu.")
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

    device = torch.device(args.device)

    if not args.quiet:
        print("Running on", len(args.config), "config files...")

    for fname in (args.config if args.quiet else tqdm(args.config)):
        # place log and saved model next to config
        base = os.path.splitext(fname)[0]
        log_fname = base + ".log"
        model_fname = base + ".pt"

        with open(fname, "r") as cfg_f:
            cfg = json.load(cfg_f)

        with open(log_fname, "w", newline='') as log_f:
            expr = TransductiveExperiment(
                device, args.dataset, cfg, log_f, model_fname)
            expr.run()
