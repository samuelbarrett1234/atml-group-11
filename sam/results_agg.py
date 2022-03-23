import csv
import json
import glob
import math
import collections
import argparse as ap


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--filenames", action='store_true',
                     help="Print only best model log filenames.")
    grp.add_argument("--configs", action='store_true',
                     help="Print only best model configurations.")
    parser.add_argument("log_filenames", type=str,
                        help="GLOB string to match all training logs.")
    args = parser.parse_args()

    # mapping from tag to a 3-tuple containing
    # (i) the best validation performance seen so far,
    # and (ii) which configuration achieved it
    # and (iii) the log filename of that configuration
    best_for_tag = collections.defaultdict(lambda: (-math.inf, None, None))

    for fname in glob.glob(args.log_filenames, recursive=True):
        with open(fname, "r") as f:
            # firstly read the config:
            cfg = json.loads(f.readline())
            # now read the training logs in CSV format
            for row in csv.reader(f):
                # if the validation accuracy is better, save it
                val_acc = float(row[3])
                if val_acc > best_for_tag[cfg["tag"]][0]:
                    best_for_tag[cfg["tag"]] = (val_acc, cfg, fname)

    if args.filenames:
        for values in best_for_tag.values():
            print(values[2])
    elif args.configs:
        for values in best_for_tag.values():
            pretty_cfg = json.dumps(values[1], indent=4, sort_keys=True)
            print(pretty_cfg)
    else:
        print("The bests found were:")
        for tag, values in best_for_tag.items():
            print("-----------------------")
            print(f"Tag '{tag}' achieved validation accuracy {values[0]} using configuration:")
            pretty_cfg = json.dumps(values[1], indent=4, sort_keys=True)
            print(pretty_cfg)
