# CVPCR
 Cross-Validated Principal Component Regression for voxelwise magnetic resonance imaging (MRI) images
 
## Table of Contents
1. [Project Overview](#Project_Overview)
2. [Setup](#Setup)\
3. [Instructions](#Instructions)\
   3.1 [Script](#script)\
   3.2 [Function](#function)\
   3.3 [Standalone application](#standalone_application)\
   3.4 [Relevant outputs](#relevant_outputs)

## 1. Project Overview <a name="Project_Overview"></a>
This repository contains the code used for the differentiation between patients affected by major depressive disorder and bipolar disorder, based on structural neuroimaging. The code was developed to run a state-of-the-art machine learning pipeline on voxelwise MRI images, minimizing data leakage. With this aim in mind, the pipeline comprised a first (i) confound regression step to remove the effects of nuisance covariates, (ii) data standardization, (iii) a dimensionality reduction step with principal component analysis (PCA), in order to project the data in a compressed space with a reduced number of features, and enter the compressed data into an elastic-net penalized regression. The entire pipeline was set into a nested cross-validation scheme to optimize the hyper-parameters, yet avoiding data leakage. The optimized hyper-parameters were the number of principal components (PCs), the elastic-net penalization strength, and the elastic-net trade-off between the LASSO and ridge penalization terms.\
The repository provides the code for: 
- model estimation and scoring
- model investigation through permutation test and bootstrap procedures, as well as the McNemar's test to assess differences across different models
- back-projection of regression weights in the original space through PCs and projection on atlases for statistical inference and interpretation of the neuroimaging results

## 2. Setup <a name="Setup"></a>
The code is organized in scripts that are used for different purposes. Each script starts with the imports, and then there are all the needed inputs. Just download the repository and change the paths at the beginning of each script. 
