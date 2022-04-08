import os
import json
import copy
import pathlib
import itertools
import argparse as ap
import tqdm


"""EXAMPLE SWEEP
If your JSON file is
{
    "x" : [1, 2, 3]
    "y" : {
        "z" : [4, 5, 6],
        "w" : 7
    }
}
and you want to iterate over x and z, you would supply arguments
x y.z
to the command line.
You cannot specify y or y.w because they do not point to lists.
"""


def check_path(in_cfg, path):
    for name in path.split('.'):
        try:
            in_cfg = in_cfg[name]
        except KeyError:
            raise ValueError(f"Could not find {name} in path {path}.")

    if not isinstance(in_cfg, list):
        raise ValueError(f"Path {path} did not map to a list.")


def get_path_size(in_cfg, path):
    for name in path.split('.'):
        in_cfg = in_cfg[name]
    return len(in_cfg)


def get_elem(in_cfg, path, index):
    for name in path.split('.'):
            in_cfg = in_cfg[name]
    return in_cfg[index]


def set_elem(in_cfg, path, val):
    for name in path.split('.'):
        if isinstance(in_cfg[name], list):
            in_cfg[name] = val
        else:
            in_cfg = in_cfg[name]


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("input", type=str,
                        help="Input config file.")
    parser.add_argument("out_folder", type=str,
                        help="Folder to write the new config files to.")
    parser.add_argument("cross_product_paths", nargs='+', type=str,
                        help=("A sequence of full-stop-delimited-sequences "
                              "specifying paths in the JSON file pointing "
                              "to lists of parameter values to grid search. "
                              "See `sweep.py` for an example."))
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print("Cannot find input file.")
        exit(1)

    if not os.path.isdir(args.out_folder):
        print("Cannot find output folder.")
        exit(1)

    # load input config
    with open(args.input, "r") as f:
        cfg = json.load(f)

    # check all paths are valid
    for path in args.cross_product_paths:
        check_path(cfg, path)

    # now iterate through them:
    ranges = [range(get_path_size(cfg, path)) for path in args.cross_product_paths]
    for file_counter, idxs in tqdm.tqdm(enumerate(itertools.product(*ranges))):
        # construct new config
        new_cfg = copy.deepcopy(cfg)
        for j, i in enumerate(idxs):
            path = args.cross_product_paths[j]
            set_elem(new_cfg, path, get_elem(new_cfg, path, i))

        # get its filename
        fname = str(pathlib.Path(args.out_folder)
            / pathlib.Path(args.input).stem) + str(file_counter) + ".json"

        # save it
        with open(fname, "w") as f:
            json.dump(new_cfg, f)
