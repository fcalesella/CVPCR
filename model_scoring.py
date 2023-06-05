#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 15:31:31 2022

@author: psicobiologia
"""

import numpy as np
from sklearn import metrics

# define full path to the predictions of the model
predictions_path = '/path_to_output_directory/predictions.csv'
# define full path to the target variable
target_path = '/path_to_target/target.csv'

# load predicted (predictions) and true (target) values
predictions = np.genfromtxt(predictions_path, delimiter=',')
target = np.genfromtxt(target_path, delimiter=',')

# calculate performance metrics
accuracy = metrics.accuracy_score(target, predictions)
ba = metrics.balanced_accuracy_score(target, predictions)
f1 = metrics.f1_score(target, predictions)
auc = metrics.roc_auc_score(target, predictions)
tn, fp, fn, tp = metrics.confusion_matrix(target, predictions).ravel()
sensitivity = tp/(tp+fn)
specificity = tn/(tn+fp)
ppv = tp/(tp+fp)
npv = tn/(fn+tn)

print('Accuracy = %.2f\nBA = %.2f\nF1 = %.2f\nSensitivity = %.2f\nSpecificity = %.2f\nPPV = %.2f\nNPV = %.2f\nAUC = %.2f\n' % (accuracy, 
      ba, f1, sensitivity, specificity, ppv, npv, auc))