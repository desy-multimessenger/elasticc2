import os
import numpy as np
import pandas as pd

from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor

import matplotlib.pyplot as plt

datadir = "data"
lcfile = os.path.join(datadir, "plasticc_train_lightcurves.csv.gz")
lcs = pd.read_csv(lcfile)

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

test_lc = lcs[lcs["object_id"] == 54338231]

band_colors = ["C4", "C2", "C3", "C1", "k", "C5"]
band_names = ["u", "g", "r", "i", "z", "y"]

fig, ax = plt.subplots()

for band, group in test_lc.groupby("passband"):

    k = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(
        length_scale=1, length_scale_bounds=(1e-3, 1e3)
    )

    x = np.atleast_2d(group["mjd"].values).T
    y = group["flux"].values
    y_err = group["flux_err"].values

    gp = GaussianProcessRegressor(kernel=k, alpha=y_err**2, n_restarts_optimizer=1000)

    gp.fit(x, y)

    start_mod = 59580.0343
    end_mod = 60674.363

    n_points = int(end_mod - start_mod)

    x_sampled = np.atleast_2d(np.linspace(start_mod, end_mod, n_points)).T

    y_sampled_g, sigma_sampled_g = gp.predict(x_sampled, return_std=True)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    # x_pred = np.atleast_2d(np.linspace(start_mod, end_mod, n_points)).T
    y_pred_g, sigma_g = gp.predict(x_sampled, return_std=True)

    ax.plot(x_sampled, y_pred_g, band_colors[band] + "-", label="Prediction", alpha=0.2)
    ax.fill(
        np.concatenate([x_sampled, x_sampled[::-1]]),
        np.concatenate(
            [y_pred_g - 1.9600 * sigma_g, (y_pred_g + 1.9600 * sigma_g)[::-1]]
        ),
        alpha=0.1,
        fc=band_colors[band],
        ec="None",
    )  # , label='95% confidence interval')

    ax.errorbar(
        x=group["mjd"],
        y=group["flux"],
        yerr=group["flux_err"],
        fmt="o",
        color=band_colors[band],
        label=band_names[band],
    )

plt.legend(ncol=6)
ax.set_ylabel("flux")
ax.set_xlabel("mjd")
plt.savefig("test.png")
plt.close()
