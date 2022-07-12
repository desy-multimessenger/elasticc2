#!/usr/bin/env python3
# License: BSD-3-Clause

import os
import logging
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)

logger = logging.getLogger(__name__)


class Model:
    """ """

    def __init__(
        self,
        stage: str,
        path_to_trainingset: str,
        n_iter: int = 1,
        random_state: int = 42,
        one_alert_per_stock: bool = False,
    ) -> None:

        # super(Model, self).__init__()  # is this really needed ?
        self.stage = stage
        self.path_to_trainingset = path_to_trainingset
        self.n_iter = n_iter
        self.random_state = random_state
        self.one_alert_per_stock = one_alert_per_stock

        self.create_dirs()
        self.get_df()

    def get_df(self):
        """
        convert to parquet for performance reasons
        and create a dataframe with the training data
        """
        if not os.path.exists(self.path_to_trainingset + ".parquet"):
            df = pd.read_csv(self.path_to_trainingset + ".csv").drop(
                columns="Unnamed: 0"
            )
            logger.info("Saving training data as parquet file")
            df.to_parquet(self.path_to_trainingset + ".parquet")
        df = pd.read_parquet(self.path_to_trainingset + ".parquet")
        bool_cols = [i for i in df.keys().values if "bool" in i]

        for c in bool_cols:
            df[c] = df[c].astype(bool)

        self.df = df

    def create_dirs(self):
        """
        create all dirs for stage
        """
        if self.stage == "1":
            self.plot_dir = os.path.join("plots", "first_stage")
            self.model_dir = os.path.join("models", "first_stage")

        elif self.stage == "2a":
            self.plot_dir = os.path.join("plots", "second_stage", "A")
            self.model_dir = os.path.join("models", "second_stage", "A")

        elif self.stage == "2b":
            self.plot_dir = os.path.join("plots", "second_stage", "B")
            self.model_dir = os.path.join("models", "second_stage", "B")

        else:
            raise ValueError("You must select '1', '2a' or '2b' as stage")

        for path in [self.plot_dir, self.model_dir]:
            if not os.path.exists(path):
                os.makedirs(path)

    def split_sample(self):
        """
        Split the training data in a train and test sample
        """
        logger.info(
            f"The complete dataset has {len(self.df)} entries (before removing data)."
        )

        if self.one_alert_per_stock:
            logger.info("You have selected to only keep one alert per stock ID")
            logger.info(f"Initial length of sample: {len(self.df)}")
            self.df = self.get_random_stock_subsample(self.df)
            logger.info(
                f"Length of sample after keeping one alert per stock only: {len(self.df)}"
            )

        if self.stage == "1":
            # Here we use the full training sample
            target = self.df.class_short - 1

        elif self.stage == "2a":
            """
            Here we cut on everything that SHOULD have been selected
            by stage 1 as non-recurring (11, 12 or 13)
            We will classify 11 and 13 as 0 and 12 as 1
            """
            self.df = self.df.query("class_intermediate in [11, 12, 13]")
            self.df["class_short"] = self.df["class_intermediate"].replace(13, 11)
            self.df.drop(columns="class_intermediate", inplace=True)
            target = self.df.class_short - 11

        elif self.stage == "2b":
            """
            Here we cut on everything that SHOULD have been selected
            by stage 1 as recurring (21 or 22)
            We will classify 21 as 0 and 22 as 1
            """
            self.df = self.df.query("class_intermediate in [21, 22]")
            self.df["class_short"] = self.df.class_intermediate
            self.df.drop(columns="class_intermediate", inplace=True)
            target = self.df.class_short - 21

        else:
            raise ValueError("stage must be '1', '2a' or '2b'")

        logger.info(f"Stage {self.stage} dataset hast {len(self.df)} entries.")

        all_cols = self.df.keys().values.tolist()

        excl_cols = [i for i in all_cols if "class" in i]

        self.cols_to_use = [i for i in all_cols if i not in excl_cols]

        feats = self.df[self.cols_to_use]

        X_train, X_test, y_train, y_test = self.train_test_split_stock(
            X=feats, y=target, test_size=0.3, random_state=self.random_state
        )

        # X_train, X_test, y_train, y_test = train_test_split(
        #     feats, target, test_size=0.3, random_state=self.random_state
        # )

        logger.info("\nSplitting sample.\n")
        logger.info(f"The training sample has {len(X_train)} entries.")
        logger.info(f"The testing sample has {len(X_test)} entries.\n")

        X_train = X_train.drop(columns=["stock"])

        df_test = pd.concat([X_test, y_test], axis=1)

        self.X_train = X_train
        self.y_train = y_train
        self.df_test = df_test

    def train_test_split_stock(self, X, y, test_size, random_state):
        """
        Split sample in train and test, while ensuring that all alerts
        belonging to one stock end up in test OR train, not spread
        over both
        """
        df = X.copy(deep=True)
        df["class_short"] = y.values

        df = shuffle(df, random_state=random_state).reset_index(drop=True)

        # get all unique stock ids
        unique_stock_ids = df.stock.unique()
        nr_unique_stock_ids = len(unique_stock_ids)

        """ select from the unique stock ids the sample size of 
        stock ids belonging to train and test according to test_size
        """
        nr_train_stockids = int((1 - test_size) * nr_unique_stock_ids)
        nr_test_stockids = nr_unique_stock_ids - nr_train_stockids

        # randomly get train stockids
        stock_ids_train = np.random.choice(
            unique_stock_ids, size=nr_train_stockids, replace=False
        )

        # create dataframes based on that selection
        df_train = df.query("stock in @stock_ids_train")
        df_test = df.query("stock not in @stock_ids_train")

        X_train = df_train.drop(columns="class_short").reset_index(drop=True)
        X_test = df_test.drop(columns="class_short").reset_index(drop=True)
        y_train = df_train.filter(["class_short"]).reset_index(drop=True)["class_short"]
        y_test = df_test.filter(["class_short"]).reset_index(drop=True)["class_short"]

        return X_train, X_test, y_train, y_test

    def train(self):
        """
        Do the training
        """
        scale_pos_weight = (len(self.y_train) - np.sum(self.y_train)) / np.sum(
            self.y_train
        )

        model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            random_state=self.random_state,
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

        kfold = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=self.random_state
        )

        grid_search = RandomizedSearchCV(
            model,
            param_grid,
            scoring=None,
            n_iter=self.n_iter,
            cv=kfold,
            random_state=self.random_state,
            verbose=2,
            error_score="raise",
        )

        # Run the actual training
        grid_result = grid_search.fit(self.X_train, self.y_train)

        best_estimator = grid_result.best_estimator_

        self.grid_result = grid_result
        self.best_estimator = best_estimator

        if self.one_alert_per_stock:
            outpath_grid = os.path.join(
                self.model_dir,
                f"grid_result_niter_{self.n_iter}_one_alert_per_stock",
            )
            outpath_model = os.path.join(
                self.model_dir,
                f"model_stage_{self.stage}_niter_{self.n_iter}_one_alert_per_stock",
            )
        else:
            outpath_grid = os.path.join(
                self.model_dir, f"grid_result_niter_{self.n_iter}_all_alerts"
            )
            outpath_model = os.path.join(
                self.model_dir,
                f"model_stage_{self.stage}_niter_{self.n_iter}_all_alerts",
            )

        joblib.dump(grid_result, outpath_grid)
        best_estimator.save_model(fname=outpath_model)

    def evaluate(self):
        """
        Evaluate the model
        """

        # Load the stuff
        if self.one_alert_per_stock:
            infile_grid = os.path.join(
                self.model_dir, f"grid_result_niter_{self.n_iter}_one_alert_per_stock"
            )
        else:
            infile_grid = os.path.join(
                self.model_dir, f"grid_result_niter_{self.n_iter}_all_alerts"
            )
        grid_result = joblib.load(infile_grid)
        best_estimator = grid_result.best_estimator_

        self.grid_result = grid_result
        self.best_estimator = best_estimator

        # Plot feature importance for full set
        self.plot_features()

        """
        Now we cut the test sample so that only one datapoint
        per stock-ID survives
        """
        df_test_subsample = self.get_random_stock_subsample(self.df_test)

        logger.info(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

        self.df_test_subsample = df_test_subsample

        # We get even sized binning (at least as far as possible)
        evaluation_bins, nbins = self.get_optimal_bins(nbins=14)

        self.evaluation_bins = evaluation_bins

        logger.info(f"\nWe now plot the evaluation using {nbins} time bins")

        precision_list = []
        recall_list = []
        aucpr_list = []
        timebin_mean_list = []

        for timebin in evaluation_bins:
            df_test_bin = df_test_subsample[
                (df_test_subsample["ndet"] >= timebin[0])
                & (df_test_subsample["ndet"] <= timebin[1])
            ]
            X_test = df_test_bin.drop(columns=["class_short", "stock"])

            self.cols_to_use.append("stock")
            y_test = df_test_bin.drop(columns=self.cols_to_use)
            features = X_test
            target = y_test

            pred = best_estimator.predict(features)

            precision_list.append(metrics.precision_score(target, pred))
            recall_list.append(metrics.recall_score(target, pred))
            aucpr_list.append(metrics.average_precision_score(target, pred))

            timebin_mean_list.append(np.mean([timebin[0], timebin[1]]))

        if self.one_alert_per_stock:
            outfiles = [
                os.path.join(
                    self.plot_dir, f"{i}_niter_{self.n_iter}_one_alert_per_stock.png"
                )
                for i in ["precision", "recall", "aucpr"]
            ]
        else:
            outfiles = [
                os.path.join(self.plot_dir, f"{i}_niter_{self.n_iter}_all_alerts.png")
                for i in ["precision", "recall", "aucpr"]
            ]

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(timebin_mean_list, precision_list)
        ax.set_xlabel("ndet interval center")
        ax.set_ylabel("precision")
        ax.set_ylim([0.5, 1])
        fig.savefig(outfiles[0], dpi=300)
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(timebin_mean_list, recall_list)
        ax.set_xlabel("ndet interval center")
        ax.set_ylabel("recall")
        ax.set_ylim([0.75, 1])
        fig.savefig(outfiles[1], dpi=300)
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(timebin_mean_list, aucpr_list)
        ax.set_xlabel("ndet interval center")
        ax.set_ylabel("aucpr")
        ax.set_ylim([0.5, 1])
        fig.savefig(outfiles[2], dpi=300)
        plt.close()

    def get_optimal_bins(self, nbins=20):
        """
        Determine optimal time bins (requirement: same number
        of alerts per bin). This cannot always be fulfilled, so duplicates=drop is passed.
        """
        out, bins = pd.qcut(
            self.df_test_subsample.ndet.values, nbins, retbins=True, duplicates="drop"
        )
        final_bins = []
        for i in range(len(bins) - 1):
            final_bins.append([int(bins[i]), int(bins[i + 1])])
        nbins = len(final_bins)
        return final_bins, nbins

    def get_random_stock_subsample(self, df):
        """
        Returns a df consisting of one random datapoint for each unique stock ID
        """
        df_sample = df.groupby("stock").sample(n=1, random_state=self.random_state)

        return df_sample

    def plot_features(self):
        """
        Plot the features in their importance for the classification decision
        """

        fig, ax = plt.subplots(figsize=(10, 21))

        cols = self.cols_to_use

        cols.remove("stock")

        ax.barh(cols, self.best_estimator.feature_importances_)
        plt.title("Feature importance", fontsize=25)
        plt.tight_layout()
        if self.one_alert_per_stock:
            outfile = os.path.join(
                self.plot_dir,
                f"feature_importance_{self.n_iter}_one_alert_per_stock.png",
            )
        else:
            outfile = os.path.join(
                self.plot_dir, f"feature_importance_{self.n_iter}_all_alerts.png"
            )
        fig.savefig(
            outfile,
            dpi=300,
        )
