import os

import numpy as np
import pandas as pd

DATADIR = "data"
LCFILE = os.path.join(DATADIR, "plasticc_train_lightcurves.csv.gz")
LCS = pd.read_csv(LCFILE).set_index(["object_id"])

METAFILE = os.path.join(DATADIR, "plasticc_train_metadata.csv.gz")
META = pd.read_csv(METAFILE).set_index(["object_id"])


def target_name(object_id, return_name=True):
    # source class id mapping
    category_mapping = {
        90: "SN1a",
        67: "SN1a-91bg",
        52: "SN1ax",
        42: "SN2",
        62: "SN1bc",
        95: "SLSN1",
        15: "TDE",
        64: "KN",
        88: "AGN",
        92: "RRL",
        65: "M-dwarf",
        16: "EB",
        53: "Mira",
        6: "Microlens",
    }
    # match id
    t_id = META[META.index == object_id]["target"].values[0]
    t_name = category_mapping[t_id]
    if return_name:
        return t_name
    else:
        return t_id


target_list = []
for i in META.index:
    target_id = target_name(i, return_name=False)
    target_list.append(target_id)

np.savez("augmented_data/pixelized_class.npz", target_list)
