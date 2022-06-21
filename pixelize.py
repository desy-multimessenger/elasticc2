import os, random, multiprocessing

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.stats as sts

pd.options.mode.chained_assignment = None  # default='warn'

# from sklearn.utils.testing import ignore_warnings

DATADIR = "data"
LCFILE = os.path.join(DATADIR, "plasticc_train_lightcurves.csv.gz")
LCS = pd.read_csv(LCFILE).set_index(["object_id"])

NBIN_X = 2048
NBIN_Y = 256

start_mod = 59580.0343
end_mod = 60674.363


def bin(df):
    return None


def plot_hist(matrix):
    return None


ids = LCS.index.unique().values

for test_id in tqdm(ids[0:1]):
    for passband in range(1):
        # test_id = 1920

        lc = LCS.query("object_id == @test_id and passband == @passband")

        min_flux = np.min(lc.flux)
        lc.flux = lc.flux - min_flux
        peak_flux = 1.3 * np.max(lc.flux)
        lc.flux /= peak_flux
        lc.flux_err /= peak_flux

        pix_to_flux = peak_flux / NBIN_Y
        print(f"Peak flux: {peak_flux}")
        print(f"Flux increase per Pixel: {pix_to_flux}")

        bins = np.linspace(start=start_mod, stop=end_mod, num=NBIN_X) - start_mod

        hist = np.histogram2d(x=lc.mjd, y=lc.flux, bins=[NBIN_X, NBIN_Y])[0]

        k = 0
        for i in range(NBIN_X):
            column = hist[i, :]
            if np.sum(column) == 1:
                flux = lc.flux.values[k]
                err = lc.flux_err.values[k]
                normaldist = sts.norm(loc=flux, scale=err)
                for l in column:
                    if l == 1:
                        central_pixel = l
                k += 1
                quit()
        quit()

# fig, ax = plt.subplots()
# ax.hist2d(x=lc.mjd, y=lc.flux, bins=[NBIN_X, 256])
# fig.savefig("test.png")
