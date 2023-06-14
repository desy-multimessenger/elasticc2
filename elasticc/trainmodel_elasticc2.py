#!/usr/bin/env python3
# License: BSD-3-Clause

import os, re
import glob
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
        pos_tax: list[int],
        neg_tax: list[int],
        path_to_featurefiles: str,
        max_taxlength: int = -1,  
        n_iter: int = 1,
        random_state: int = 42,
        plotdir: str = '.',
        grid_search_sample_size: int = 10000,
        cols_to_use: list[str] = []
    ) -> None:

        self.pos_tax = pos_tax
        self.neg_tax = neg_tax
        self.path_to_featurefiles = path_to_featurefiles
        self.max_taxlength = max_taxlength
        self.n_iter = n_iter
        self.random_state = random_state
        self.grid_search_sample_size = grid_search_sample_size
        self.plotdir = plotdir
        self.readlog = []
        self.random_state: int = 42

        self.create_dirs()
        self.df_train = self.read_featuredata(training=True)
        self.df_val = self.read_featuredata(training=False)

        if len(cols_to_use)>0:
            # Use input features
            self.cols_to_use = cols_to_use
        else:
            # Use training data columns (except target) 
            self.cols_to_use = list(self.df_train.columns)
            self.cols_to_use.remove('target')


    def create_dirs(self):
        """
        create all dirs for stage
        """
        self.plot_dir = os.path.join(self.plotdir, "plots")
        self.model_dir = os.path.join(self.plotdir, "models")

        for path in [self.plot_dir, self.model_dir]:
            if not os.path.exists(path):
                os.makedirs(path)



    def read_featuredata(self, training=True):
        """
        Parse file directory for files belonging to desired input and output data.
                
        """
        if training:
	        files = glob.glob( os.path.join(self.path_to_featurefiles,"features*_train.parquet") ) 
        else:
	        files = glob.glob( os.path.join(self.path_to_featurefiles,"features*_validate.parquet") ) 

	
        fulldf = None
        for fname in files:
	        taxclass = int( re.search("features_(\d+)_ndet", fname)[1] )
	        if not taxclass in self.pos_tax+self.neg_tax:
	                continue
		
	        df = pd.read_parquet(fname)
	        inrows = df.shape[0]

	        # Limit rows per class
	        if self.max_taxlength>0 and inrows>self.max_taxlength:
		        df = df.sample(n=self.max_taxlength, random_state=self.random_state)
	        userows = df.shape[0]
		
	        df = df.drop(columns=["stock"])
	        df['target'] = (taxclass in self.pos_tax)
#	        df = df.replace({np.nan: None})
	        df = df.fillna(np.nan)
		
	        if fulldf is None:
	                fulldf = df
	        else:
	                fulldf = pd.concat([fulldf,df], ignore_index=True)
	        self.readlog.append([taxclass, fname, inrows, userows])
		
        return fulldf

    def plot_balance(self):
        """
        Look through the read archive for balance between classes.
                
        """
        raise("NotImplementedError")
		


    def train(self):
        """
        Do the training
        """
        t_start = time.time()

        y_train = self.df_train['target']
        scale_pos_weight = (len(y_train) - np.sum(y_train)) / np.sum( y_train )
        logger.info(f"Find scale weight {scale_pos_weight} from {np.sum(y_train)} pos out of {len(y_train)} rows.")
        
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
        X_train_subset = self.df_train[self.cols_to_use].sample(
            n=self.grid_search_sample_size, random_state=self.random_state + 5
        )
        y_train_subset = y_train.sample(
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

        best_estimator = grid_result.best_estimator_.fit(self.df_train[self.cols_to_use], y_train)

        self.grid_result = grid_result
        self.best_estimator = best_estimator

        outpath_grid = os.path.join(
            self.model_dir,
            f"grid_result_niter_{self.n_iter}_nsample_{self.grid_search_sample_size}",
        )
        outpath_model = os.path.join(
            self.model_dir,
            f"model_pos{'-'.join(map(str,self.pos_tax))}_neg{'-'.join(map(str,self.neg_tax))}_niter_{self.n_iter}_nsample_{self.grid_search_sample_size}",
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
        #df_test_subsample = self.get_random_stock_subsample(self.df_val)
        # self.df_test_subsample = df_test_subsample
        # With the new larger feature files we have now assumed that subselection has already been done.
        self.df_test_subsample = self.df_val

        logger.info(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")



        # We get even sized binning (at least as far as possible)
        evaluation_bins, nbins = self.get_optimal_bins(nbins=14)

        self.evaluation_bins = evaluation_bins

        logger.info(f"\nWe now plot the evaluation using {nbins} time bins")

        precision_list = []
        recall_list = []
        aucpr_list = []
        timebin_mean_list = []

        for timebin in evaluation_bins:
            df_test_bin = self.df_test_subsample[
                (self.df_test_subsample["ndet"] >= timebin[0])
                & (self.df_test_subsample["ndet"] <= timebin[1])
            ]
            
            #X_test = df_test_bin.drop(columns=["class_short", "stock"])
            #self.cols_to_use.append("stock")
            
#            y_test = df_test_bin.drop(columns=self.cols_to_use)
#            features = X_test
            features = df_test_bin[self.cols_to_use]
            target = df_test_bin.target

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
#        cols =  self.df_train.drop(columns=['target']).columns

        #cols.remove("stock")

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