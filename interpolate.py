import os
import random

import multiprocessing
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

# from sklearn.utils.testing import ignore_warnings

DATADIR = "data"
LCFILE = os.path.join(DATADIR, "plasticc_test_lightcurves_01.csv.gz")
LCS = pd.read_csv(LCFILE).set_index(["object_id"])

METAFILE = os.path.join(DATADIR, "plasticc_test_metadata.csv.gz")
META = pd.read_csv(METAFILE).set_index(["object_id"])


BAND_COLORS = ["C4", "C2", "C3", "C1", "k", "C5"]
BAND_STYLE = ["s", "^", "o", "x", "v", "p"]
BAND_NAMES = ["u", "g", "r", "i", "z", "y"]
CATEGORY_MAPPING = {
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


def interpolate_lc(object_id, plot=True):

    simplefilter("ignore", category=ConvergenceWarning)

    lc = LCS[LCS.index == object_id]

    t_id = META[META.index == object_id]["target"].values[0]

    classification = CATEGORY_MAPPING[t_id]

    if plot:
        fig, ax = plt.subplots(figsize=(12, 12.0 / 1.6), tight_layout=True)

    result_df_list = []

    for band, group in lc.groupby("passband"):

        _df = pd.DataFrame()

        k = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(
            length_scale=1, length_scale_bounds=(1e-3, 1e3)
        )

        x = np.atleast_2d(group["mjd"].values).T
        y = group["flux"].values
        y_err = group["flux_err"].values

        gp = GaussianProcessRegressor(
            kernel=k, alpha=y_err**2, n_restarts_optimizer=1000
        )

        gp.fit(x, y)

        start_mod = 59580.0343
        end_mod = 60674.363

        n_points = int(end_mod - start_mod)

        x_sampled = np.atleast_2d(np.linspace(start_mod, end_mod, n_points)).T

        y_sampled_g, sigma_sampled_g = gp.predict(x_sampled, return_std=True)

        # Make the prediction on the meshed x-axis (ask for MSE as well)
        # x_pred = np.atleast_2d(np.linspace(start_mod, end_mod, n_points)).T
        x_pred = x_sampled
        y_pred_g, sigma_g = gp.predict(x_sampled, return_std=True)

        if plot:

            ax.plot(
                x_sampled,
                y_pred_g,
                BAND_COLORS[band] + "-",
                alpha=0.2,
            )

            ax.fill(
                np.concatenate([x_sampled, x_sampled[::-1]]),
                np.concatenate(
                    [y_pred_g - 1.9600 * sigma_g, (y_pred_g + 1.9600 * sigma_g)[::-1]]
                ),
                alpha=0.1,
                fc=BAND_COLORS[band],
                ec="None",
            )  # , label='95% confidence interval')

            ax.errorbar(
                x=group["mjd"],
                y=group["flux"],
                yerr=group["flux_err"],
                ls="",
                color=BAND_COLORS[band],
                label=BAND_NAMES[band],
                marker=BAND_STYLE[band],
                ms=5,
            )

            _df["object_id"] = [object_id] * n_points
            _df["mjd"] = x_pred
            _df["passband"] = [band] * n_points
            _df["flux"] = y_pred_g
            _df["flux_err"] = sigma_g
            _df["detected_bool"] = [1] * n_points

            _df.set_index("object_id", inplace=True)

            result_df_list.append(_df)

    if plot:
        ax.set_title(f"ID = {object_id} / Type = {classification}")
        plt.legend(ncol=6, fontsize=12)
        ax.set_ylabel("Flux")
        ax.set_xlabel("MJD")

        plotdir = "plots"
        if not os.path.exists(plotdir):
            os.makedirs(plotdir)
        plt.savefig(os.path.join(plotdir, f"{object_id}.pdf"), dpi=300)
        plt.close()

    final_df = pd.concat(result_df_list)

    return final_df


if __name__ == "__main__":

    nprocess = 40

    ids = LCS.index.unique().values[:2000]

    result_list = []
    i = 0

    with multiprocessing.Pool(nprocess) as p:
        for res in p.map(interpolate_lc, ids):
            print(f"Processing lightcurve {i+1} of {len(ids)}")
            i += 1
            result_list.append(res)

    final_df = pd.concat(result_list)

    print("Regression done")

    if not os.path.exists("augmented_data"):
        os.makedirs("augmented_data")

    outfile = os.path.join("augmented_data", "gp_test_data.csv")

    final_df.to_csv(outfile)
