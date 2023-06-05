#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 11:48:36 2022

@author: psicobiologia
"""

import pickle
import scipy.io
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict, cross_validate, StratifiedKFold, permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
import sys
# import personalized modules
sys.path.append('/path_to_repository/utils')
from confounds import ConfoundRegressor

# define full path to input data (predictors)
input_data = '/path_to_data/data.mat'
# define full path to confounders (covariates)
covariates_data = '/path_to_confounders/covariates.csv'
# define full path to target variable
target_data = '/path_to_target/target.csv'
# define full path to the estimated model
model_name = '/path_to_output_directory/model.pkl'
# define full path where to save permutations results
save_results = '/path_to_output_directory/permutations.pkl'
# define the number of permutations
n_permutations = 5000

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

# define CV scheme. Here 5-Fold nested CV was used
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

# run the permutations on the estimated model
true_score, null_scores, pvalue = permutation_test_score(model, data, target, cv=cv,
                                                         n_permutations=n_permutations,
                                                         random_state=1234, n_jobs=1)

# save the score of the model, the score distribution under the null model, and the p-value
pickle.dump((true_score, null_scores, pvalue), open(save_results, 'wb'))
