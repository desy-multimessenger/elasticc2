import os, random
import numpy as np
import pandas as pd

from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor

import matplotlib.pyplot as plt

DATADIR = "data"
LCFILE = os.path.join(DATADIR, "plasticc_train_lightcurves.csv.gz")
LCS = pd.read_csv(LCFILE)

METAFILE = os.path.join(DATADIR, "plasticc_train_metadata.csv.gz")
META = pd.read_csv(METAFILE)


BAND_COLORS = ["C4", "C2", "C3", "C1", "k", "C5"]
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


def interpolate_lc(object_id, plot=False):

    lc = LCS[LCS["object_id"] == object_id]

    t_id = META[META.object_id == object_id]["target"].values[0]

    classification = CATEGORY_MAPPING[t_id]

    if plot:
        fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)

    for band, group in lc.groupby("passband"):

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
                label="Prediction",
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
                fmt="o",
                color=BAND_COLORS[band],
                label=BAND_NAMES[band],
            )

    if plot:
        ax.set_title(f"ID = {object_id} / Type = {classification}")
        plt.legend(ncol=6)
        ax.set_ylabel("flux")
        ax.set_xlabel("mjd")

        plotdir = "plots"
        if not os.path.exists(plotdir):
            os.makedirs(plotdir)
        plt.savefig(os.path.join(plotdir, f"{object_id}.png"), dpi=300)
        plt.close()

    return None


if __name__ == "__main__":

    ids = LCS["object_id"].unique()
    random_id = random.choice(ids)
    print(f"Drawing a random ID: {random_id}")

    interpolate_lc(object_id=random_id, plot=True)
    print("Regression done")
