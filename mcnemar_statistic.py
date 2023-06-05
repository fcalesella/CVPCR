#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 15:05:52 2022

@author: psicobiologia
"""

import numpy as np
from sklearn import metrics
from statsmodels.stats.contingency_tables import mcnemar

# define full path to target variable
target_path = '/path_to_target/target.csv'
# define full path to the predictions of a model
model1_preds = '/path_to_output_directory/predictions.csv'
# define full path to the predictions of another model
model2_preds = '/path_to_output_directory/predictions.csv'

# load the true (target) and predicted (predictions) values of the two models
target = np.genfromtxt(target_path, delimiter=',')
predictions1 = np.genfromtxt(model1_preds, delimiter=',')
predictions2 = np.genfromtxt(model2_preds, delimiter=',')

# calculate marginal statistics
correct1 = target == predictions1
incorrect1 = target != predictions1
correct2 = target == predictions2
incorrect2 = target != predictions2

n00 = sum(incorrect1 * incorrect2)
n01 = sum(incorrect1 * correct2)
n10 = sum(correct1 * incorrect2)
n11 = sum(correct1 * correct2)

table = [[n00, n01],
         [n10, n11]]

# calculate McNemar's test
# stat = ((abs(n01 - n10) - 1)**2) / (n01 + n10)
results = mcnemar(table, exact=False, correction=True)
print('McNemar’s test statistic = %f\np-value = ' % results.statistic, results.pvalue)