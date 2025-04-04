#!/usr/bin/env python3
# License: BSD-3-Clause

import glob
import itertools
import logging
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV  # type: ignore
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

import joblib  # type: ignore
import xgboost as xgb
from elasticc2.taxonomy import var as vartax

logger = logging.getLogger(__name__)


class XgbModel:
    """
    Version of Model which assumes datasets have been preprocessed as follows:
    - Combining z, galcol info with features.
    - Limit to det, time ranges.
    - Subselect to max nbr alerts per stock.
    - Divide into training and validation samples.
    - Stored as parquet.
    Assumes data structure from process_features[_step3].py

    Will glob all files in target directory.
    """

    def __init__(
        self,
        tax: list[int],
        name: str,
        path_to_featurefiles: str | Path,
        max_taxlength: int = -1,
        n_iter: int = 5,
        random_state: int = 42,
        plotdir: str | Path = ".",
        grid_search_sample_size: int = 10000,
        cols_to_use: list[str] = [],
        n_threads: int | None = None,
        objective: str | None = None,
    ) -> None:
        self.tax = tax
        self.name = name
        self.path_to_featurefiles = Path(path_to_featurefiles)
        self.max_taxlength = max_taxlength
        self.n_iter = n_iter
        self.random_state = random_state
        self.grid_search_sample_size = grid_search_sample_size
        self.plotdir = Path(plotdir)
        self.readlog: list = []
        self.n_threads = n_threads

        self.create_dirs()
        self.df_train = self.read_featuredata(training=True)
        self.df_val = self.read_featuredata(training=False)

        if objective is None:
            self.objective = "multi:softmax"
        else:
            self.objective = objective

        if len(cols_to_use) > 0:
            # Use input features
            self.cols_to_use = cols_to_use
        else:
            # Use training data columns (except target)
            self.cols_to_use = list(self.df_train.columns)
            self.cols_to_use.remove("target")

        logger.info(
            f"\n-------------------"
            f"\nTraining multivariate model for {len(self.tax)} classes\n"
            f"{self.name} ({', '.join(vartax.keys_from_ids(self.tax))})\n"
            f"-------------------"
        )

    def create_dirs(self):
        """
        Create the directories required
        """
        self.plot_dir = self.plotdir / "plots_multivar" / f"{self.name}"
        self.model_dir = self.plotdir / "models_multivar" / f"{self.name}"

        for path in [self.plot_dir, self.model_dir]:
            path.mkdir(exist_ok=True, parents=True)

    def read_featuredata(self, training=True):
        """
        Parse file directory for files belonging to desired input and output data.
        """
        if not self.path_to_featurefiles.is_dir():
            import tarfile

            logger.info(
                f"{self.path_to_featurefiles}.tar.xz has not yet been extracted, doing "
                f"that now. Extracting to {self.path_to_featurefiles.parents[0]}"
            )
            with tarfile.open(f"{self.path_to_featurefiles}.tar.xz") as f:
                f.extractall(path=self.path_to_featurefiles.parents[0])
            logger.info("Extraction done")

        if training:
            files = glob.glob(
                str(self.path_to_featurefiles / "features*_train.parquet")
            )
        else:
            files = glob.glob(
                str(self.path_to_featurefiles / "features*_validate.parquet")
            )

        fulldf = None

        for fname in files:
            
            # Format either with each taxonmy class as individual files, or a joint one
            if (ndetmatch:=re.search(r"features_(\d+)_ndet", fname)) is not None:
                taxclass = int(ndetmatch[1])
                if taxclass not in self.pos_tax + self.neg_tax:
                    continue
                df = pd.read_parquet(fname)
                df["target"] = vartax.keys_from_ids(taxclass)[0]

            else:
                # Assuming pure file with taxonomy "taxid" column 
                df = pd.read_parquet(fname)
                df['target'] = df['taxid'].apply(lambda x: vartax.keys_from_ids(x)[0])
                df = df.drop(columns=['taxid'])

            inrows = df.shape[0]

            # Limit rows per class
            if self.max_taxlength > 0 and inrows > self.max_taxlength:
                df = df.sample(n=self.max_taxlength, random_state=self.random_state)
            userows = df.shape[0]

            cols_to_drop = [
                colname for colname in df.keys() if colname in ["stock", "true"]
            ]


            df = df.drop(columns=cols_to_drop)
            df = df.fillna(np.nan)

            if fulldf is None:
                fulldf = df
            else:
                fulldf = pd.concat([fulldf, df], ignore_index=True)

            self.readlog.append([fname, inrows, userows])

        self.n_classes = len(fulldf.target.unique())

        self.le = LabelEncoder()
        self.le.fit(np.unique(fulldf.target))

        fulldf.target = self.le.transform(fulldf.target.tolist())

        if training:
            self.label_to_num = {
                self.le.inverse_transform([num])[0]: num
                for num in np.unique(fulldf.target)
            }
            self.num_to_label = {v: k for k, v in self.label_to_num.items()}

        return fulldf

    def plot_balance(self):
        """
        Look through the read archive for balance between classes.

        """
        raise ("NotImplementedError")

    def train(self):
        """
        Do the training
        """
        t_start = time.time()

        y_train = self.df_train["target"]

        model = xgb.XGBClassifier(
            random_state=self.random_state,
            objective=self.objective,
            num_class=self.n_classes,
            eval_metric="aucpr",
            colsample_bytree=1.0,
            n_jobs=self.n_threads,
        )

        param_grid = {
            "learning_rate": [0.1, 0.01, 0.001],
            "gamma": [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2],
            "max_depth": [2, 4, 7, 10],
            "colsample_bytree": [0.3, 0.6, 0.8, 1.0],
            "subsample": [0.2, 0.4, 0.5, 0.6, 0.7],
            "reg_alpha": [0, 0.5, 1],
            "reg_lambda": [1, 1.5, 2, 3, 4.5],
            "min_child_weight": [1, 3, 5, 7],
            "n_estimators": [100, 250, 500, 1000],
        }
        # Copied from aug 17 elasticc run
        param_grid = {
            "learning_rate": [0.1, 0.01, 0.5],
            "gamma": [1, 1.5, 2, 2.5],
            "max_depth": [10, 12, 14],
            "colsample_bytree": [0.1, 0.3, 0.6, 0.8, 1.0],
            "subsample": [0.4, 0.5, 0.6, 0.7, 0.8],
            "reg_alpha": [0.5, 1, 1.5],
            "reg_lambda": [2, 3, 4],
            "min_child_weight": [1, 5],
            "n_estimators": [500],
        }
        # updated sep 26 2024 based on noiztf trial runs
        param_grid = {
            "learning_rate": [0.1, 0.01, 0.5],
            "gamma": [0.8, 1, 1.5, 2],
            "max_depth": [11, 12, 13],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "subsample": [0.75, 0.8],
            "reg_alpha": [0.4, 0.5, 0.6],
            "reg_lambda": [1.5, 2, 2.5],
            "min_child_weight": [1, 2],
            "n_estimators": [200, 500, 1000],
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
        X_train_subset = self.df_train[self.cols_to_use].sample(
            n=self.grid_search_sample_size, random_state=self.random_state + 5
        )
        y_train_subset = y_train.sample(
            n=self.grid_search_sample_size, random_state=self.random_state + 5
        )
        class_weights = class_weight.compute_sample_weight(
            class_weight="balanced", y=y_train_subset
        )
        grid_result = grid_search.fit(
            X_train_subset, y_train_subset, sample_weight=class_weights
        )

        """
        Run the actual training with the best estimator
        on the full training sample
        """
        logger.info(
            f"\n\n{'':-^42}\n"
            f"{'START FITTING':^42}\n"
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()):^42}\n"
            "Now fitting with the best estimator from the grid search "
            f"for {len(self.tax)} classes\n"
            f"({self.name}: {', '.join(vartax.keys_from_ids(self.tax))}).\n"
            f"This will take time\n"
            f"{'':-^42}\n"
        )

        class_weights_full = class_weight.compute_sample_weight(
            class_weight="balanced", y=y_train
        )

        best_estimator = grid_result.best_estimator_.fit(
            self.df_train[self.cols_to_use],
            y_train,
            sample_weight=class_weights_full,
        )

        self.grid_result = grid_result
        self.best_estimator = best_estimator

        outpath_grid = self.model_dir / "grid_result"

        outpath_model = self.model_dir / f"model_{self.name}"

        joblib.dump(grid_result, outpath_grid)
        joblib.dump(
            {"columns": self.cols_to_use, "model": best_estimator}, outpath_model
        )

        t_end = time.time()
        t_delta = (t_end - t_start) / 60
        duration_str = f"This took {t_delta:1.1f} minutes"

        logger.info(
            f"\n\n{'':-^42}\n"
            f"{'FITTING DONE':^42}\n"
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()):^42}\n"
            f"{duration_str:^42}\n"
            f"{'':-^42}\n"
        )

    def evaluate(self):
        """
        Evaluate the model
        """
        # Load the stuff
        infile_grid = self.model_dir / "grid_result"

        grid_result = joblib.load(infile_grid)
        best_estimator = grid_result.best_estimator_

        self.grid_result = grid_result
        self.best_estimator = best_estimator

        logger.info(f"Loading best estimator. Parameters:\n{self.best_estimator}")

        # Plot feature importance for full set
        self.plot_features()

        """
        Now we cut the test sample so that only one datapoint per stock-ID survives
        """
        self.df_test_subsample = self.df_val

        logger.info(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

        # We get even sized binning (at least as far as possible)
        # Not used anymore? Causing errors when only using parsnip, i.e. ndet not being available
        #evaluation_bins, nbins = self.get_optimal_bins(nbins=14)
        #self.evaluation_bins = evaluation_bins

        # now we plot the confusion matrix

        features = self.df_test_subsample[self.cols_to_use]
        target = self.df_test_subsample.target

        y_true = target.values
        y_pred = list(best_estimator.predict(features))

        self.plot_confusion(y_true=y_true, y_pred=y_pred)
        self.plot_confusion(y_true=y_true, y_pred=y_pred, normalize="all")
        self.plot_confusion(y_true=y_true, y_pred=y_pred, normalize="pred")
        self.plot_confusion(y_true=y_true, y_pred=y_pred, normalize="true")

    def get_optimal_bins(self, nbins=20):
        """
        Determine optimal time bins (requirement: same number of alerts per bin).
        This cannot always be fulfilled, so duplicates=drop is passed.
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

        ax.barh(cols, self.best_estimator.feature_importances_)
        plt.title("Feature importance", fontsize=25)
        plt.tight_layout()

        outfile = self.plot_dir / f"feature_importance_{self.name}.pdf"

        fig.savefig(
            outfile,
            dpi=300,
        )

        plt.close()

    def plot_confusion(
        self, y_true: np.ndarray, y_pred: np.ndarray, normalize: str = None
    ):
        """
        Plot the confusion matrix for the binary classification
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        cm = confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            normalize=normalize,
            labels=list(self.num_to_label.keys()),
        )

        if normalize == "all":
            cmlabel = "Fraction of objects"
            fmt = ".2f"
            vmax = cm.max()
        elif normalize in ["pred", "true"]:
            cmlabel = "Fraction of objects"
            fmt = ".2f"
            vmax = 1
        else:
            vmax = cm.max()
            cmlabel = "Nr. of objects"
            fmt = ".0f"

        im = plt.imshow(
            cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=vmax
        )

        plt.xticks(
            list(self.num_to_label.keys()), list(self.label_to_num.keys()), ha="center"
        )
        plt.yticks(list(self.num_to_label.keys()), list(self.label_to_num.keys()))

        outpath = self.plot_dir / f"confusion_{self.name}_norm_{normalize}.pdf"

        thresh = cm.max() / 2.0

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

        plt.ylabel("True Type", fontsize=12)
        plt.xlabel("Predicted Type", fontsize=12)

        # Make a colorbar that is lined up with the plot
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.25)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(label=cmlabel, fontsize=12)

        plt.tight_layout()
        plt.savefig(outpath, dpi=300)
        logger.info(f"We saved the evaluation to {outpath}")

        plt.close()
