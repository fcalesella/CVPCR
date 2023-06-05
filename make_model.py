#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 17:57:47 2022

@author: psicobiologia
"""

import pickle
import scipy.io
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
# define full path where to save the model
save_model = '/path_to_output_directory/model.pkl'
# define full path where to save the predictions
save_predictions = '/path_to_output_directory/predictions.csv'

# load the data
data = scipy.io.loadmat(input_data)['data']
covariates = np.genfromtxt(covariates_data, delimiter=',')
target = np.genfromtxt(target_data, delimiter=',') 

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

# define the structure of the pipeline
pipe = Pipeline([('cleaner', ConfoundRegressor(covariates, data)),
                 ('scaler', StandardScaler()), 
                 ('pca', PCA(whiten=True, svd_solver='full', random_state=1234)),
                 ('reg', LogisticRegression(penalty='elasticnet', 
                                              solver='saga', class_weight='balanced', 
                                              n_jobs=-1, random_state=1234))])

# define the sets of values on which to optimize the pipeline
param_grid = [{
    "pca__n_components": [None] + list(np.linspace(0.95, 0.5, 10)),
    "reg__C": np.logspace(-5, 5, 100),
    "reg__l1_ratio": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    }]

# run the model estimation
grid_search = GridSearchCV(pipe, param_grid, cv=cv, scoring='neg_log_loss', 
                           n_jobs=1, refit=True)
cv_results = cross_validate(grid_search, data, y=target, cv=cv, n_jobs=-1,
                            return_estimator=True)

# extract a single model based on median parameters across CV folds
ncomp = list()
c = list()
l1 = list()
for estimator in cv_results['estimator']:
    ncomp.append(estimator.best_estimator_['pca'].n_components_)
    c.append(estimator.best_params_['reg__C'])
    l1.append(estimator.best_params_['reg__l1_ratio'])

# set the median best parameters in the pipeline    
n_components = np.median(ncomp)
C = np.median(c)
l1_ratio = np.median(l1)
pipe.set_params(pca__n_components=int(n_components))
pipe.set_params(reg__C=C)
pipe.set_params(reg__l1_ratio=l1_ratio)

# estimate the model without paramter optimization to get the predictions
predictions = cross_val_predict(pipe, data, y=target, cv=cv, n_jobs=-1)

# save the final model and its predicitons
pickle.dump(pipe, open(save_model, 'wb'))
np.savetxt(save_predictions, predictions, delimiter=",")
