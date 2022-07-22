#!/usr/bin/env python3
# License: BSD-3-Clause

import os
import logging
import joblib
import time
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
        grid_search_sample_size: int = 10000,
    ) -> None:

        # super(Model, self).__init__()  # is this really needed ?
        # It's the default in sublime when creating a new class, so
        # I threw it in for good measure :-)
        self.stage = stage
        self.path_to_trainingset = path_to_trainingset
        self.n_iter = n_iter
        self.random_state = random_state
        self.grid_search_sample_size = grid_search_sample_size

        self.create_dirs()
        self.get_df()

    def get_df(self):
        """
        convert to parquet for performance reasons
        and create a dataframe with the training data
        """
        filename, file_extension = os.path.splitext(self.path_to_trainingset)

        if not os.path.exists(filename + ".parquet"):
            if file_extension == ".csv":
                df = pd.read_csv(self.path_to_trainingset).drop(columns="Unnamed: 0")
            elif file_extension == ".pkl":
                df = pd.read_pickle(self.path_to_trainingset)

                # Create shortened class columns
                logger.info("Creating additional columns. This might take a moment")

                class_full = df.class_full.values
                class_intermediate = []
                class_short = []
                for c in class_full:
                    class_intermediate.append(int(c / 10))
                for c in class_intermediate:
                    class_short.append(int(c / 10))

                df["class_intermediate"] = class_intermediate
                df["class_short"] = class_short

            logger.info("Saving training data as parquet file")
            df.to_parquet(filename + ".parquet")

        df = pd.read_parquet(filename + ".parquet")

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
        t_start = time.time()

        scale_pos_weight = (len(self.y_train) - np.sum(self.y_train)) / np.sum(
            self.y_train
        )

        model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            random_state=self.random_state,
            objective="binary:logistic",
            eval_metric="aucpr",
            colsample_bytree=1.0,
        )

        param_grid = {
            "max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "min_child_weight": np.arange(0.0001, 0.5, 0.001),
            "gamma": np.arange(0.0, 40.0, 0.005),
            "learning_rate": np.arange(0.0005, 0.5, 0.0005),
            "subsample": np.arange(0.01, 1.0, 0.01),
            "colsample_bylevel": np.round(np.arange(0.1, 1.0, 0.01)),
            # "colsample_bytree": np.arange(0.1, 1.0, 0.01),
        }

        kfold = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=self.random_state + 3
        )

        grid_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            scoring=None,
            n_iter=self.n_iter,
            cv=kfold,
            random_state=self.random_state + 4,
            verbose=2,
            error_score="raise",
        )

        """
        Now we downsample our training set to do a
        fine-grained grid search. Training will be done 
        on the best estimator from that search and uses 
        the full sample
        """
        logger.info(
            f"Downsampling for grid search to {self.grid_search_sample_size} entries\n"
        )
        X_train_subset = self.X_train.sample(
            n=self.grid_search_sample_size, random_state=self.random_state + 5
        )
        y_train_subset = self.y_train.sample(
            n=self.grid_search_sample_size, random_state=self.random_state + 5
        )

        grid_result = grid_search.fit(X_train_subset, y_train_subset)

        """
        Run the actual training with the best estimator
        on the full training sample
        """
        logger.info("--------------------------------------------")
        logger.info(
            "\n\nNow fitting with the best estimator from the grid search. This will take time\n"
        )
        logger.info("--------------------------------------------")

        best_estimator = grid_result.best_estimator_.fit(self.X_train, self.y_train)

        self.grid_result = grid_result
        self.best_estimator = best_estimator

        outpath_grid = os.path.join(
            self.model_dir,
            f"grid_result_niter_{self.n_iter}_nsample_{self.grid_search_sample_size}",
        )
        outpath_model = os.path.join(
            self.model_dir,
            f"model_stage_{self.stage}_niter_{self.n_iter}_nsample_{self.grid_search_sample_size}",
        )

        joblib.dump(grid_result, outpath_grid)
        joblib.dump(best_estimator, outpath_model)

        t_end = time.time()

        logger.info("------------------------------------")
        logger.info("           FITTING DONE             ")
        logger.info(f"  This took {(t_end-t_start)/60} minutes")
        logger.info("------------------------------------")

    def evaluate(self):
        """
        Evaluate the model
        """

        # Load the stuff
        infile_grid = os.path.join(
            self.model_dir,
            f"grid_result_niter_{self.n_iter}_nsample_{self.grid_search_sample_size}",
        )

        grid_result = joblib.load(infile_grid)
        best_estimator = grid_result.best_estimator_

        self.grid_result = grid_result
        self.best_estimator = best_estimator

        logger.info(f"Loading best estimator. Parameters:\n{self.best_estimator}")

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

        outfiles = [
            os.path.join(
                self.plot_dir,
                f"{i}_niter_{self.n_iter}_nsample_{self.grid_search_sample_size}.png",
            )
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

        outfile = os.path.join(
            self.plot_dir,
            f"feature_importance_niter_{self.n_iter}_nsample_{self.grid_search_sample_size}.png",
        )

        fig.savefig(
            outfile,
            dpi=300,
        )
