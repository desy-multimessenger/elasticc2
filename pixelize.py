import os, random, multiprocessing

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd

# from sklearn.utils.testing import ignore_warnings

DATADIR = "data"
LCFILE = os.path.join(DATADIR, "plasticc_train_lightcurves.csv.gz")
LCS = pd.read_csv(LCFILE).set_index(["object_id"])

NBIN = 1024


def bin(df):
    return None


ids = LCS.index.unique().values

test_id = ids[-1]

print(test_id)
