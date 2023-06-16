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

## Test
Run `demo_elasticc2.py`

## Old usage (NEED TO CHANGE THIS AFTER WE'RE DONE):
```python
import os, logging
from train_models import Model

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

path_to_trainingset = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "data",
    "elasticc_feature_trainingset_v3",
)

m = Model(
    stage="1",
    path_to_trainingset=path_to_trainingset,
    n_iter=10,
    random_state=42,
    one_alert_per_stock=True,
)
m.split_sample()
m.train()
m.evaluate()
```