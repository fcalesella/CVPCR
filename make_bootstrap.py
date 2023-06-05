#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 15:52:36 2022

@author: psicobiologia
"""

import pickle
import scipy.io
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict, cross_validate, StratifiedKFold, permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn import metrics
from sklearn.utils import shuffle, resample
import sys
# import personalized modules
sys.path.append('/path_to_repository/utils')
from confounds import ConfoundRegressor
from bootstrap import bootstrap

# define full path to input data (predictors)
input_data = '/path_to_data/data.mat'
# define full path to confounders (covariates)
covariates_data = '/path_to_confounders/covariates.csv'
# define full path to target variable
target_data = '/path_to_target/target.csv'
# define full path to the estimated model on which bootstrap should be done
model_name = '/path_to_output_directory/model.pkl'
# define full path where to save bootstrap results
save_results = '/path_to_output_directory/bootstrap.pkl'
# define number of resamplings
n_bootstrap = 5000

# load the data and model parameters
data = scipy.io.loadmat(input_data)['data']
covariates = np.genfromtxt(covariates_data, delimiter=',')
target = np.genfromtxt(target_data, delimiter=',')
model = pickle.load(open(model_name, 'rb'))

# check for NaNs and remove them
data_nan = np.isnan(data).any(axis=1)
covariates_nan = np.isnan(covariates).any(axis=1)
target_nan = np.isnan(target)
all_nan = data_nan | covariates_nan | target_nan
data = data[~all_nan, :]
covariates = covariates[~all_nan, :]
target = target[~all_nan]

# define a class wrapper in order to create a "fit" method that returns the coefficients of the pipeline
class FitWrapper():
    
    def __init__(self, model):
        
        self.model = model
        
    def fit(self, X, y):
        
        coefs = self.model.fit(X, y)['reg'].coef_
        
        return coefs

# fit the first part of the model to project the data on the principal components
model_preproc = model[0:3]
data_preproc = model_preproc.fit_transform(data)

# run bootstrap estimation on the regression part to get bootstrapped coefficients
model_estimate = model[3:]
fw = FitWrapper(model_estimate)
results = bootstrap(data_preproc, target, fw.fit, n_bootstrap,
                    confidence_level=0.95, method='bca', random_state=1234)

# calculate distribution statistics of the bootstrapped coefficients
boot_mean = np.mean(results['Coefficients'], axis=1)
boot_median = np.median(results['Coefficients'], axis=1)
boot_sd = np.std(results['Coefficients'], axis=1)
boot_vip = (np.sum(results['Coefficients']!=0, axis=1) * 100) / n_bootstrap;

# store the statistics in a dictionary
bootstats = {'Mean': boot_mean,
             'Median': boot_median,
             'SD': boot_sd,
             'LowerCI': results['LowerCI'],
             'UpperCI': results['UpperCI'],
             'VIP': boot_vip}

# save the bootstrapped coefficients and the their distribution statistics
pickle.dump((results['Coefficients'], bootstats), open(save_results, 'wb'))
