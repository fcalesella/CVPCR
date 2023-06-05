#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 11:03:33 2022

@author: psicobiologia
"""

import numpy as np
import nibabel as nib
import pandas as pd
from scipy import stats

# define full path to the coefficients map
coefficient_path = 'path_to_output_directory/map_median_coefficient.nii'
# define the full path to the 3D mask
mask_path = '/path_to_mask/mask.nii'
# define full path to the atlas
atlas_path = '/path_to_atlas/atlas.nii'
# define full path to the labels of the atlas
label_path = '/path_to_atlas/labels.xml'
# define full path to the output directory
output_path = '/path_to_output_directory/rois.csv'

# load the 3D coefficients map and vecotrize it
coefficient_map = nib.load(coefficient_path)
original_shape = coefficient_map.shape
original_affine = coefficient_map.affine
coefficient_map = coefficient_map.get_fdata()
coefficient_map = np.reshape(coefficient_map, np.prod(original_shape))

# load the 3D mask and vectorize it
mask = nib.load(mask_path).get_fdata()
mask = np.reshape(mask, np.prod(mask.shape))

# load the atlas and store the labels in a list
atlas = nib.load(atlas_path).get_fdata()
atlas = np.reshape(atlas, np.prod(atlas.shape))
with open(label_path, "r") as f:
    labels = f.readlines()
labels = [label.strip().split('>') for label in labels if label.startswith('<label')]
coordinates = [label[0].replace('<label ', '') for label in labels]
labels = [label[1].replace('</label', '') for label in labels]

# apply the mask both on the atlas and the coefficients map
atlas = atlas[mask>0]
coefficient_map = coefficient_map[mask>0]

# project the data on each ROI and calculate statistics 
tracts = np.unique(atlas)
results = np.zeros((len(tracts), 5))
for tract_id in tracts:
    tract_id = int(tract_id)
    tract_coefficient = coefficient_map[atlas == tract_id]
    results[tract_id, 0] = np.mean(tract_coefficient)
    results[tract_id, 1] = np.median(tract_coefficient)
    results[tract_id, 2] = np.std(tract_coefficient)
    t, p = stats.ttest_1samp(tract_coefficient, popmean=0)
    results[tract_id, 3] = t
    results[tract_id, 4] = p
    
# store the results in a dataframe
cols = ['Mean', 'Median', 'S.D.', 'T', 'p-value']
results = pd.DataFrame(results, columns=cols)
results.insert(loc=0, column='Coordinates', value=coordinates)
results.insert(loc=1, column='Label', value=labels)
    
# save the results
results.to_csv(output_path, index=False)
    