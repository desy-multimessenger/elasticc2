import argparse
import logging
import os
import socket
from pathlib import Path
from pprint import pprint

from elasticc2.taxonomy import var as tax
from elasticc2.utils import load_config

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

if socket.gethostname() in ["wgs33.zeuthen.desy.de"]:
    n_threads = 32
elif socket.gethostname() in ["wgs18.zeuthen.desy.de"]:
    n_threads = 16
else:
    n_threads = None

config = load_config()
# Path to extracted features
# These will already include subselection based on ndet, restrictions on alerts per
# object and train/validation split
basedir = os.environ.get("ELASTICCDATA")


setups_binary = {
    1: {
        "key": "galactic",
        "pos_tax": config["galactic"],
        "neg_tax": tax.get_ids(exclude=config["galactic"]),
        "neg_name": "non_galactic",
    },
    2: {
        "key": "varstar_ulens",
        "neg_name": "mdwarf_nova",
        "pos_tax": [*tax.rec.periodic.get_ids(), *tax.ids_from_keys("ulens")],
        "neg_tax": [tax.nrec.fast.mdwarf.id, tax.nrec.fast.nova.id],
    },
    3: {
        "key": "nova",
        "neg_name": "mdwarf",
        "pos_tax": tax.ids_from_keys("nova"),
        "neg_tax": tax.ids_from_keys("mdwarf"),
    },
    4: {
        "key": "agn",
        "neg_name": "kn_parsnip",
        "pos_tax": tax.ids_from_keys("agn"),
        "neg_tax": [
            *tax.ids_from_keys("kn"),
            *tax.nrec.sn.get_ids(),
            *tax.nrec.long.get_ids(),
        ],
    },
    5: {
        "key": "kn",
        "neg_name": "parsnip",
        "pos_tax": tax.ids_from_keys("kn"),
        "neg_tax": [
            *tax.nrec.sn.get_ids(),
            *tax.nrec.long.get_ids(),
        ],
    },
    6: {
        "key": "sn",
        "neg_name": "long",
        "pos_tax": tax.nrec.sn.get_ids(),
        "neg_tax": tax.nrec.long.get_ids(),
    },
    7: {
        "key": "snia",
        "neg_name": "snlong",
        "pos_tax": tax.ids_from_keys("snia"),
        "neg_tax": [
            tax.ids_from_keys("snibc"),
            tax.ids_from_keys("snii"),
            tax.ids_from_keys("sniin"),
            tax.ids_from_keys("sniax"),
            tax.ids_from_keys("sn91bg"),
            tax.nrec.long.get_ids(),
        ],
    },
    8: {
        "key": "snia",
        "neg_name": "snibc",
        "pos_tax": tax.ids_from_keys("snia"),
        "neg_tax": [
            tax.ids_from_keys("snibc"),
        ],
    },
    9: {
        "key": "snii",
        "neg_name": "snibc",
        "pos_tax": tax.ids_from_keys("snii"),
        "neg_tax": [
            tax.ids_from_keys("snibc"),
        ],
    },
}

setups_multivar = {
    1: {"name": "stars", "tax": tax.rec.periodic.get_ids()},
    2: {
        "name": "stars_ulens",
        "tax": [*tax.rec.periodic.get_ids(), *tax.ids_from_keys("ulens")],
    },
    3: {
        "name": "stars_ulens_mdwarf_nova",
        "tax": [
            *tax.rec.periodic.get_ids(),
            *tax.ids_from_keys(["ulens", "mdwarf", "nova"]),
        ],
    },
    4: {"name": "all", "tax": tax.get_ids()},
    5: {"name": "nrec", "tax": tax.nrec.get_ids()},
    6: {"name": "sn_long", "tax": [*tax.nrec.sn.get_ids(), *tax.nrec.long.get_ids()]},
    7: {
        "name": "agn_kn_sn_long",
        "tax": [
            *tax.ids_from_keys(["kn", "agn"]),
            *tax.nrec.sn.get_ids(),
            *tax.nrec.long.get_ids(),
        ],
    },
}


def run_setup_binary(num: int, max_taxlength: int, path_to_featurefiles: str):
    pos_tax = setups_binary[num]["pos_tax"]
    neg_tax = setups_binary[num]["neg_tax"]
    key = setups_binary[num]["key"]
    neg_name = setups_binary[num]["neg_name"]
    if isinstance(pos_tax[0], list):
        pos_tax = [
            x 
            for xs in pos_tax
            for x in xs
        ]
    if isinstance(neg_tax[0], list):
        neg_tax = [
            x 
            for xs in neg_tax
            for x in xs
        ]

    from elasticc2.train_binary_model import XgbModel

    if num == 3:
        grid_search_sample_size = 1000
    else:
        grid_search_sample_size = 10000

    model = XgbModel(
        pos_tax=pos_tax,
        neg_tax=neg_tax,
        pos_name=key,
        n_iter=10,
        neg_name=neg_name,
        path_to_featurefiles=path_to_featurefiles,
        grid_search_sample_size=grid_search_sample_size,
        max_taxlength=max_taxlength,
        n_threads=n_threads,
    )

    model.train()

    model.evaluate()


def run_setup_multivar(num: int, max_taxlength: int, path_to_featurefiles: str):
    tax = setups_multivar[num]["tax"]
    name = setups_multivar[num]["name"]

    from elasticc2.train_multivar_model import XgbModel

    model = XgbModel(
        tax=tax,
        name=name,
        n_iter=10,
        path_to_featurefiles=path_to_featurefiles,
        max_taxlength=max_taxlength,
        n_threads=n_threads,
        # objective=None,
        objective="multi:softprob",
    )

    model.train()

    model.evaluate()


def run_xgb() -> None:
    """
    Executes to model training
    """
    # Argument parsing
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="XGB classifier training for ELAsTiCC 2",
    )
    parser.add_argument(
        "mode",
        metavar="mode",
        choices=["bin", "multi"],
        help=(
            "Specifies the XGB configurarion. "
            "Either binary ('bin') or multivariate ('multi')."
        ),
    )
    parser.add_argument(
        "-s",
        "--setups",
        type=int,
        default=None,
        nargs="+",
        help=(
            "Specifies a list of preconfigured training setups. Training is performed "
            "on setups in the same order as supplied in the list. "
            "Default is to run all setups. "
            "Example usage: -s 1 2 3"
        ),
    )
    parser.add_argument(
        "-l",
        "--max-taxlength",
        default=-1,
        type=int,
        help=(
            "Limits the number of samples per taxonomy class included in the training. "
            "Default is to use the complete data set (-1). "
        ),
    )
    parser.add_argument(
        "-d",
        "--dirpath",
        default=None,
        type=str,
        help=(
            "Path to folder containing training and validation feature files. "
            "Default is to use the environment variable 'ELASTICCDATA'. "
            "Feature files assumed to be called features*[train|validate].parquet."
        ),
    )

    parser.add_argument(
        "-L",
        "--list",
        action="store_true",
        help="Prints the list of available setups for given mode and quits.",
    )

    args: argparse.Namespace = parser.parse_args()

    # Prints list of available setups and quits
    if args.list:
        if args.mode == "bin":
            pprint(setups_binary, sort_dicts=False)
        else:
            pprint(setups_multivar, sort_dicts=False)
        quit()

    print(f"Running XGB with config: {args}")

    # Selects run method and setup dictionary from corresponding XGB mode
    if args.mode == "bin":
        run_method = run_setup_binary
        setup_dict = setups_binary
    elif args.mode == "multi":
        run_method = run_setup_multivar
        setup_dict = setups_multivar
    else:
        raise ValueError(f"Unknown XGB mode '{args.mode}'.")

    # Sets the training setup
    if args.setups is None:
        setups = list(setup_dict.keys())
    else:
        setups = args.setups
        
    # Set the path to training failes
    if args.dirpath is None:
        if basedir is None:
            raise ValueError("Please provide input dir or set an environment-variable for 'ELASTICCDATA'")
        path_to_featurefiles = Path(basedir)
    else:
        path_to_featurefiles = Path(args.dirpath)


    # Sets the sample limit
    max_taxlength = args.max_taxlength
    
    print("path", path_to_featurefiles) 

    # Run
    for setup in setups:
        run_method(setup, max_taxlength, path_to_featurefiles)


if __name__ == "__main__":
    run_xgb()
