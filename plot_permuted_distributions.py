# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 10:25:14 2023

@author: fede_
"""
import pickle
import matplotlib.pyplot as plt

# define full path to the results of the permutations test
results = '/path_to_output_directory/permutations.pkl'
# define full path where to save the image
image = 'path_to_output_directory/figure.svg'
# define the image format
image_format = 'svg'

# load the results of the permutations test
perm = pickle.load(open(results, 'rb'))
permc = perm[1]
# plot
fig, ax = plt.subplots()
plt.hist(permc, color='0.5')
plt.axvline(perm[0], 0, 1400, color='red', linestyle='dashed')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.show()
fig.savefig(image, format=image_format, dpi=1200)