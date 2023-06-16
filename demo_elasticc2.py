import logging
import os
from pathlib import Path

from elasticc2.taxonomy import root
from elasticc2.trainmodel_elasticc2 import XgbModel
from elasticc2.utils import load_config

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# Path to extracted features
# These will already include subselection based on ndet, restrictions on alerts per object and train/validation split
basedir = os.environ.get("ELASTICCDATA")
if basedir is None:
    raise ValueError("Please set an environment-variable for 'ELASTICCDATA'")
path_to_featurefiles = Path(basedir) / "feature_extraction" / "trainset_all_max3"

config = load_config()

cl_inv = config["all"]

key = "recurring"

pos_tax = config[key]
neg_tax = [i for i in cl_inv if i not in pos_tax]
max_taxlength = 10000

model = XgbModel(
    pos_tax=pos_tax,
    neg_tax=neg_tax,
    pos_name=key,
    neg_name=f"non_{key}",
    path_to_featurefiles=path_to_featurefiles,
    max_taxlength=max_taxlength,
)

model.train()

model.evaluate()
