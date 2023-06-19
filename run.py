import logging
import os
from pathlib import Path

from elasticc2.taxonomy import var as tax
from elasticc2.utils import load_config

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

config = load_config()
# Path to extracted features
# These will already include subselection based on ndet, restrictions on alerts per object and train/validation split
basedir = os.environ.get("ELASTICCDATA")
if basedir is None:
    raise ValueError("Please set an environment-variable for 'ELASTICCDATA'")
path_to_featurefiles = Path(basedir) / "feature_extraction" / "trainset_all_max3"

setups_binary = {
    1: {
        "key": "galactic",
        "pos_tax": config["galactic"],
        "neg_tax": tax.get_ids(exclude=config["galactic"]),
        "neg_name": "non_galactic",
    },
    2: {
        "key": "recurring",
        "pos_tax": tax.rec.get_ids(),
        "neg_tax": tax.get_ids(exclude=tax.rec.get_ids()),
        "neg_name": "non_recurring",
    },
    3: {
        "key": "kn",
        "pos_tax": tax.ids_from_keys("kn"),
        "neg_tax": tax.get_ids(exclude=tax.ids_from_keys("kn")),
        "neg_name": "all_other",
    },
    4: {
        "key": "parsnip",
        "pos_tax": [*tax.nrec.sn.get_ids(), *tax.nrec.long.get_ids()],
        "neg_tax": tax.get_ids(
            exclude=[*tax.nrec.sn.get_ids(), *tax.nrec.long.get_ids()]
        ),
        "neg_name": "all_other",
    },
    5: {
        "key": "ceph",
        "pos_tax": [tax.rec.periodic.ceph.id],
        "neg_tax": tax.rec.periodic.get_ids(exclude=[tax.rec.periodic.ceph.id]),
        "neg_name": "other_stars",
    },
    6: {
        "key": "deltasc",
        "pos_tax": [tax.rec.periodic.deltasc.id],
        "neg_tax": tax.rec.periodic.get_ids(exclude=[tax.rec.periodic.deltasc.id]),
        "neg_name": "other_stars",
    },
    7: {
        "key": "eb",
        "pos_tax": [tax.rec.periodic.eb.id],
        "neg_tax": tax.rec.periodic.get_ids(exclude=[tax.rec.periodic.eb.id]),
        "neg_name": "other_stars",
    },
    8: {
        "key": "rrlyrae",
        "pos_tax": [tax.rec.periodic.rrlyrae.id],
        "neg_tax": tax.rec.periodic.get_ids(exclude=[tax.rec.periodic.rrlyrae.id]),
        "neg_name": "other_stars",
    },
    9: {
        "key": "ulens+mdwarf",
        "pos_tax": tax.ids_from_keys(["ulens", "mdwarf"]),
        "neg_tax": tax.rec.periodic.get_ids(),
        "neg_name": "periodic_star",
    },
    10: {
        "key": "kn",
        "pos_tax": tax.ids_from_keys("kn"),
        "neg_tax": [*tax.nrec.sn.get_ids(), *tax.nrec.long.get_ids()],
        "neg_name": "parsnip",
    },
}

setups_multivar = {1: {"name": "stars", "tax": tax.rec.periodic.get_ids()}}


def run_setup_binary(num: int):
    pos_tax = setups_binary[num]["pos_tax"]
    neg_tax = setups_binary[num]["neg_tax"]
    key = setups_binary[num]["key"]
    neg_name = setups_binary[num]["neg_name"]

    from elasticc2.train_binary_model import XgbModel

    model = XgbModel(
        pos_tax=pos_tax,
        neg_tax=neg_tax,
        pos_name=key,
        n_iter=10,
        neg_name=neg_name,
        path_to_featurefiles=path_to_featurefiles,
        max_taxlength=max_taxlength,
    )

    model.train()

    model.evaluate()


def run_setup_multivar(num: int):
    tax = setups_multivar[num]["tax"]
    name = setups_multivar[num]["name"]

    from elasticc2.train_multivar_model import XgbModel

    model = XgbModel(
        tax=tax,
        name=name,
        n_iter=10,
        path_to_featurefiles=path_to_featurefiles,
        max_taxlength=max_taxlength,
    )

    model.train()

    model.evaluate()


for setup in [1]:
    max_taxlength = -1
    run_setup_multivar(1)
