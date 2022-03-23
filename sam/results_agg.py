import csv
import json
import glob
import math
import collections
import argparse as ap


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("log_filenames", type=str,
                        help="GLOB string to match all training logs.")
    args = parser.parse_args()

    # mapping from tag to a 2-tuple containing
    # (i) the best validation performance seen so far,
    # and (ii) which configuration achieved it
    best_for_tag = collections.defaultdict(lambda: (-math.inf, None))

    for fname in glob.glob(args.log_filenames, recursive=True):
        with open(fname, "r") as f:
            # firstly read the config:
            cfg = json.loads(f.readline())
            # now read the training logs in CSV format
            for row in csv.reader(f):
                # if the validation accuracy is better, save it
                val_acc = float(row[3])
                if val_acc > best_for_tag[cfg["tag"]][0]:
                    best_for_tag[cfg["tag"]] = (val_acc, cfg)

    print("The bests found were:")
    for tag, acc_cfg in best_for_tag.items():
        print("-----------------------")
        print(f"Tag '{tag}' achieved validation accuracy {acc_cfg[0]} using configuration:")
        pretty_cfg = json.dumps(acc_cfg[1], indent=4, sort_keys=True)
        print(pretty_cfg)
