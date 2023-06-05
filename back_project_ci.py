#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 16:24:25 2022

@author: psicobiologia
"""

import pickle
import scipy.io
import numpy as np
import nibabel as nib
from sklearn.decomposition import PCA
import scipy.stats as st
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

# define full path to input data (predictors)
input_data = '/path_to_data/data.mat'
# define full path to confounders (covariates)
covariates_data = '/path_to_confounders/covariates.csv'
# define full path to the estimated model
model_name = '/path_to_output_directory/model.pkl'
# define the full path to the results of the bootstrap 
boot_results = '/path_to_output_directory/bootstrap.pkl'
# define the full path to the 3D mask
mask_path = '/path_to_mask/mask.nii'
# define the full path of the directory where to save the maps
map_path = 'path_to_output_directory'
# define the significance threshold
alpha = 0.05

# load the data, the model parameters, and the results of the bootstrap
data = scipy.io.loadmat(input_data)['data']
covariates = np.genfromtxt(covariates_data, delimiter=',')
model = pickle.load(open(model_name, 'rb'))
resboot = pickle.load(open(boot_results, 'rb'))

# check for NaNs and remove them
data_nan = np.isnan(data).any(axis=1)
covariates_nan = np.isnan(covariates).any(axis=1)
all_nan = data_nan | covariates_nan
data = data[~all_nan, :]
covariates = covariates[~all_nan, :]

# run the first part of pipeline to extract the principal components
model_preproc = model[0:3]
data_preproc = model_preproc.fit(data)
components = model_preproc['pca'].components_

# back project the bootstrapped regresison coefficients
bc = resboot[0]
back_proj = np.matmul(components.T, bc)

# calculate the distribution statistics
boot_mean = np.mean(back_proj, axis=1)
boot_median = np.median(back_proj, axis=1)
boot_sd = np.std(back_proj, axis=1)
boot_se = st.sem(back_proj, axis=1)
boot_lowerci, boot_upperci = st.t.interval(alpha=1-alpha,
                                          df=back_proj.shape[0]-1,
                                          loc=boot_mean,
                                          scale=boot_se)

# create a mask for statistically significant voxels
lci_sig = boot_lowerci > 0
uci_sig = boot_upperci < 0
sig_mask = lci_sig + uci_sig > 0

# load the 3D mask and vectorize it
mask = nib.load(mask_path)
affine = mask.affine
shape = mask.shape
mask = mask.get_fdata()
mask = np.reshape(mask, np.prod(shape))

# recreate the 3D map of mean bootstrap regression coefficients
img_vec = np.zeros(np.prod(shape))
img_vec[mask>0] = boot_mean
mean_data = np.reshape(img_vec, shape)
mean_img = nib.Nifti1Image(mean_data, affine)
mean_name = map_path + '/' + 'map_mean_coefficient.nii'
nib.save(mean_img, mean_name)

# recreate the 3D map of median bootstrap regression coefficients
img_vec[mask>0] = boot_median
median_data = np.reshape(img_vec, shape)
median_img = nib.Nifti1Image(median_data, affine)
median_name = map_path + '/' + 'map_median_coefficient.nii'
nib.save(median_img, median_name)

# create the 3D map of significant voxels
sig_vec = np.zeros(np.prod(shape))
sig_vec[mask>0] = sig_mask
sig_data = np.reshape(sig_vec, shape)
sig = nib.Nifti1Image(sig_data, affine)
significance_name = map_path + '/' + 'map_significance.nii'
nib.save(sig, significance_name)