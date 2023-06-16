import logging
import os
from pathlib import Path

from elasticc2.trainmodel_elasticc2 import XgbModel

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Path to extracted features
# These will already include subselection based on ndet, restrictions on alerts per object and train/validation split
basedir = os.environ.get("ELASTICCDATA")
if basedir is None:
    raise ValueError("Please set an environment-variable for 'ELASTICCDATA'")
path_to_featurefiles = Path(basedir) / "feature_extraction" / "trainset_all_max3"
print(path_to_featurefiles)

# Train classifer to distinguish SNe Ia (2222) from SNIbc and SNII (2223,224)
# Using a subset of max 10000 rows from the feature filws
pos_tax = [2222]
neg_tax = [2223, 2224]
max_taxlength = 10000

# Train classifer to distinguish AGN from periodic stars
# Using a subset of max 10000 rows from the feature filws
pos_tax = [2332]
neg_tax = [2322, 2323, 2324, 2325]
max_taxlength = 10000

# Train classifier to distinguish KN from ... everything else
pos_tax = [2232]
neg_tax = [
    2222,
    2223,
    2224,
    2225,
    2226,
    2233,
    2234,
    2235,
    2242,
    2243,
    2244,
    2245,
    2246,
    2322,
    2323,
    2324,
    2325,
    2332,
]
max_taxlength = 10000
# To use files where alerts with more than 10 det have been removed uncomment below
# path_to_featurefiles = '/home/jnordin/data/elasticc2/trainset_early_max10'

# Train classifier to separate recurrent from non-recurrent alerts using all alerts
pos_tax = [
    2222,
    2223,
    2224,
    2225,
    2226,
    2232,
    2233,
    2234,
    2235,
    2242,
    2243,
    2244,
    2245,
    2246,
]
neg_tax = [2322, 2323, 2324, 2325, 2332]
max_taxlength = -1


# Train classifer to distinguish Cepheids from other periodic stars
# Using a subset of max 10000 rows from the feature filws
pos_tax = [2322]
neg_tax = [2323, 2324, 2325]
max_taxlength = 10000

model = XgbModel(
    pos_tax=pos_tax,
    neg_tax=neg_tax,
    path_to_featurefiles=path_to_featurefiles,
    max_taxlength=max_taxlength,
)

model.train()

model.evaluate()
