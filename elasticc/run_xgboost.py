#!/usr/bin/env python3
# License: BSD-3-Clause
import os, pickle
import joblib
from os import listdir
from os.path import isfile, join
import imageio
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn import metrics
from sklearn.preprocessing import KBinsDiscretizer
import xgboost as xgb

from astropy.stats import knuth_bin_width

USE_COLS_JAKOB = [
    "stock",
    "bool_rise",
    "bool_fall",
    "bool_peaked",
    "bool_pure",
    "bool_fastrise",
    "bool_fastfall",
    "bool_hasgaps",
    "mag_det",
    "mag_last",
    "det_bands",
    "peak_bands",
    "last_bands",
    "t_predetect",
    "t_lc",
    "t_rise",
    "t_fall",
    "rise_slope_lsstu",
    "rise_slopesig_lsstu",
    "fall_slope_lsstu",
    "fall_slopesig_lsstu",
    "rise_slope_lsstg",
    "rise_slopesig_lsstg",
    "fall_slope_lsstg",
    "fall_slopesig_lsstg",
    "rise_slope_lsstr",
    "rise_slopesig_lsstr",
    "fall_slope_lsstr",
    "fall_slopesig_lsstr",
    "rise_slope_lssti",
    "rise_slopesig_lssti",
    "fall_slope_lssti",
    "fall_slopesig_lssti",
    "rise_slope_lsstz",
    "rise_slopesig_lsstz",
    "fall_slope_lsstz",
    "fall_slopesig_lsstz",
    "rise_slope_lssty",
    "rise_slopesig_lssty",
    "fall_slope_lssty",
    "fall_slopesig_lssty",
    "lsstu-lsstg_det",
    "lsstg-lsstr_det",
    "lsstr-lssti_det",
    "lssti-lsstz_det",
    "lsstz-lssty_det",
    "lsstu-lsstg_peak",
    "lsstg-lsstr_peak",
    "lsstr-lssti_peak",
    "lssti-lsstz_peak",
    "lsstz-lssty_peak",
    "lsstu-lsstg_last",
    "lsstg-lsstr_last",
    "lsstr-lssti_last",
    "lssti-lsstz_last",
    "lsstz-lssty_last",
    "host_sep",
    "z",
    "z_err",
    "band_det_id",
    "band_last_id",
]


USE_COLS = USE_COLS_JAKOB

BOOL_COLS = [
    "bool_rise",
    "bool_fall",
    "bool_peaked",
    "bool_pure",
    "bool_fastrise",
    "bool_fastfall",
    "bool_hasgaps",
]


def evaluate_model(features, grid_result, target):
    """ """
    print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

    means = grid_result.cv_results_["mean_test_score"]
    stds = grid_result.cv_results_["std_test_score"]
    params = grid_result.cv_results_["params"]
    best_estimator = grid_result.best_estimator_

    print("Evaluating model on the whole training sample:")
    pred = best_estimator.predict(features)
    precision = metrics.precision_score(target, pred)
    recall = metrics.recall_score(target, pred)
    aucpr = metrics.average_precision_score(target, pred)
    print(f"Precision: {precision*100:.2f}")
    print(f"Recall: {recall*100:.2f}")
    print(f"AUCPR: {aucpr*100:.2f}")
    performance_dict = {"precision": precision, "recall": recall, "aucpr": aucpr}

    return performance_dict


def get_random_stock_subsample(df):
    """
    Returns a df consisting of one random datapoint for each unique stock ID
    """
    _df_sample = df.groupby("stock").sample(n=1, random_state=None)

    return _df_sample


def run_model(df, ndet, plot=False, load=False):
    """ """
    df_set = df[(df["ndet"] >= detrange[0]) & (df["ndet"] <= detrange[1])]
    target = df_set.class_short - 1
    feats = df_set[USE_COLS]
    X_train, X_test, y_train, y_test = train_test_split(
        feats, target, test_size=0.3, random_state=42
    )
    X_train = X_train.drop(columns=["stock"])
    df_test = pd.concat([X_test, y_test], axis=1)

    """
    Now we cut the test sample so that only one datapoint
    per stock-ID survives
    """
    df_test_subsample = get_random_stock_subsample(df_test)
    X_test = df_test_subsample.drop(columns=["class_short", "stock"])
    y_test = df_test_subsample.drop(columns=USE_COLS)

    scale_pos_weight = (len(y_train) - np.sum(y_train)) / np.sum(y_train)

    model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        random_state=42,
        objective="binary:logistic",
        eval_metric="aucpr",
    )
    param_grid = {
        "max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "min_child_weight": np.arange(0.0001, 0.5, 0.001),
        "gamma": np.arange(0.0, 40.0, 0.005),
        "learning_rate": np.arange(0.0005, 0.5, 0.0005),
        "subsample": np.arange(0.01, 1.0, 0.01),
        "colsample_bylevel": np.round(np.arange(0.1, 1.0, 0.01)),
        "colsample_bytree": np.arange(0.1, 1.0, 0.01),
    }

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = RandomizedSearchCV(
        model,
        param_grid,
        scoring=None,
        n_iter=3,
        n_jobs=4,
        cv=kfold,
        random_state=42,
        verbose=1,
        error_score="raise",
    )

    grid_result = grid_search.fit(X_train, y_train)

    joblib.dump(grid_result, f"models/grid_result_{ndet}.pkl")

    if load:
        joblib.load("models/grid_result.pkl")

    result = evaluate_model(
        features=X_test,
        grid_result=grid_result,
        target=y_test,
    )

    if plot:
        plot_features(best_estimator=best_estimator, title=ndet)

    return {f"{ndet[0]}-{ndet[1]}": result}


def get_optimal_bins(df, nbins=12):
    """ """
    out, bins = pd.qcut(df.ndet.values, nbins, retbins=True)
    final_bins = []
    for i in range(len(bins) - 1):
        final_bins.append([int(bins[i]), int(bins[i + 1])])

    return final_bins


def plot_features(best_estimator, title):
    """ """
    fig, ax = plt.subplots(figsize=(10, 21))
    ax.barh(USE_COLS, best_estimator.feature_importances_)
    # plt.tight_layout()
    plt.title(ndet, fontsize=25)
    plt.tight_layout()
    fig.savefig(f"plots/{ndet}.png", dpi=300)


def plot_metrics(resultdict):
    """ """
    precision = []
    recall = []
    aucpr = []
    interval_mean = []
    for entry in resultdict.keys():
        interval = entry.split("-")
        start = int(interval[0])
        end = int(interval[1])

        interval_mean.append(np.mean([start, end]))
        precision.append(resultdict[entry]["precision"])
        recall.append(resultdict[entry]["recall"])
        aucpr.append(resultdict[entry]["aucpr"])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(interval_mean, precision)
    ax.set_xlabel("ndet interval center")
    ax.set_ylabel("precision")
    ax.set_ylim([0.5, 1])
    fig.savefig(f"plots/precision.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(interval_mean, recall)
    ax.set_xlabel("ndet interval center")
    ax.set_ylabel("recall")
    ax.set_ylim([0.94, 1])
    fig.savefig(f"plots/recall.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(interval_mean, aucpr)
    ax.set_xlabel("ndet interval center")
    ax.set_ylabel("aucpr")
    ax.set_ylim([0.5, 1])
    fig.savefig(f"plots/aucpr.png", dpi=300)
    plt.close()


if __name__ == "__main__":

    df = pd.read_csv(
        "/Users/simeon/ml_workshop/data_elasticc/elasticc_feature_trainingset_v3.csv"
    ).drop(columns="Unnamed: 0")

    # Should do this already in prep notebook
    for c in BOOL_COLS:
        df[c] = df[c].astype(bool)

    detranges = get_optimal_bins(df=df)

    result = {}

    detranges = [[0, 2000]]

    for detrange in detranges:
        print(f"Calculating bin: {detrange}")
        # for detrange in [[1, 1]]:
        result.update(run_model(df=df, ndet=detrange))

    plot_metrics(result)


# imfiles = [f for f in listdir("plots") if isfile(join("plots", f))]
# print(imfiles)
# quit()

# with imageio.get_writer("plots/features.gif", mode="I", duration=0.5) as writer:
#     for image in imfiles:
#         image = imageio.imread("plots/" + image)
#         writer.append_data(image)
