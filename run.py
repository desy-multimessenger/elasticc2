import logging
import os
from pathlib import Path

from elasticc2.taxonomy import var as tax
from elasticc2.trainmodel_elasticc2 import XgbModel
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

setups = {
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
        "neg_tax": tax.get_ids(exclude=tax.rec.ids_from_keys("kn")),
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
}


def run_setup(num: int):
    pos_tax = setups[num]["pos_tax"]
    neg_tax = setups[num]["neg_tax"]
    key = setups[num]["key"]
    neg_name = setups[num]["neg_name"]

    model = XgbModel(
        pos_tax=pos_tax,
        neg_tax=neg_tax,
        pos_name=key,
        neg_name=neg_name,
        path_to_featurefiles=path_to_featurefiles,
        max_taxlength=max_taxlength,
    )

    model.train()

    model.evaluate()


for setup in [4]:  # , 2, 4]:
    max_taxlength = 1000
    run_setup(setup)
