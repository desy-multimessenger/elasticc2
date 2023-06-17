# elasticc2

We now use this for the first stage of classification for the ELAsTiCC2 challenge.

Useful links:
- [ELAsTiCC documentation](https://portal.nersc.gov/cfs/lsst/DESC_TD_PUBLIC/ELASTICC/)
- [Data](https://syncandshare.desy.de/index.php/s/JpfTgWtrC5QDJK7)
- [Taxonomy](https://github.com/LSSTDESC/elasticc/blob/main/taxonomy/taxonomy.ipynb)

## Setup
- Download the data folder above
- set the environment variable `ELASTICCDATA` to match the download destination
- `git clone` this dir, `cd` into it and run `poetry install`

## Usage:
```python
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

basedir = os.environ.get("ELASTICCDATA")
if basedir is None:
    raise ValueError("Please set an environment-variable for 'ELASTICCDATA'")
path_to_featurefiles = Path(basedir) / "feature_extraction" / "trainset_all_max3"

# Separate recurring from non recurring transients
key = "recurring"
pos_tax = tax.rec.get_ids() # get all recurring transients
neg_tax = tax.get_ids(exclude=tax.rec.get_ids()) # non recurring is what is left
neg_name =  "non_recurring" # how to label the non recurring ones
max_taxlength = 10000 # only evaluate the first 10000 entries in the training data. If you want all, set it to -1

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
```