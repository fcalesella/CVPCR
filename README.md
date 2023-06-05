# CVPCR
 Cross-Validated Principal Component Regression for voxelwise magnetic resonance imaging (MRI) images
 
## Table of Contents
1. [Project Overview](#Project_Overview)
2. [Setup](#Setup)
3. [Usage](#Usage)

## 1. Project Overview <a name="Project_Overview"></a>
This repository contains the code used for the differentiation between patients affected by major depressive disorder and bipolar disorder, based on structural neuroimaging. The code was developed to run a state-of-the-art machine learning pipeline on voxelwise MRI images, minimizing data leakage. With this aim in mind, the pipeline comprised a first (i) confound regression step to remove the effects of nuisance covariates, (ii) data standardization, (iii) a dimensionality reduction step with principal component analysis (PCA), in order to project the data in a compressed space with a reduced number of features, and enter the compressed data into an elastic-net penalized regression. The entire pipeline was set into a nested cross-validation scheme to optimize the hyper-parameters, yet avoiding data leakage. The optimized hyper-parameters were the number of principal components (PCs), the elastic-net penalization strength, and the elastic-net trade-off between the LASSO and ridge penalization terms.\
The repository provides the code for: 
- model estimation and scoring
- model investigation through permutation test and bootstrap procedures, as well as the McNemar's test to assess differences across different models
- back-projection of regression weights in the original space through PCs and projection on atlases for statistical inference and interpretation of the neuroimaging results

## 2. Setup <a name="Setup"></a>
The code is organized in scripts that are used for different purposes. Each script starts with the imports, and then there are all the needed inputs. Just download the repository and change the paths at the beginning of each script. The required python packages are:
- matplotlib
- nibabel
- numpy
- pandas
- pickle
- scipy
- sklearn
- statsmodels
**N.B.** the ```back_project_ci```, ```make_bootstrap```, ```make_model```, and ```make_permutations``` scripts requires to modify the path also in the section of the imports, since the path to the "utils" directory of the repository is needed to import the customized modules.

## 3. Usage <a name="Usage"></a>
**Data requirements:**\
The code expects the input data to be a MATLAB (.mat) file with a two-dimensional array named "data" where rows are subjects and columns are voxels. The covariates and the target should be in a CSV (.csv) file where each row is a subject in the same order as the input data. In the covariates file, each column should be a covariate, while in the target file only one column is expected with the target class of each subject. To make the atlas based inference a nifti mask of the same size as the input data is required, as well as an atlas structured as in the FSL software https://fsl.fmrib.ox.ac.uk/fsl/fslwiki

. The atlas should be a 4D nifti with a a ROI mask in each dimension. An XML file with the label for each ROI is also needed (please see the 
The predictors (i.e., input data) 
The ```make_model``` script is the first to be run. It estimates the model and save it along with its predictions. 
The ```model_scoring``` script calculate and prints the performance metrics.
The ```make_bootstrap``` script performs the bootstrap procedure. It should be noted that the data are first projected in compressed space through the PCs, and then the bootstrap procedure is performed on the compressed data. This script saves the regression coefficients over the bootstrap resamplings for each PC, and their distribution statistics.
The ```make_permutations``` script runs permutations analysis to assess model's significance. It saves the score of the true model, the scores under the null models, and the p-value of the model. 
The ```mcnemar_statistic``` script invetsigates any signifcant difference between the predicitons of two different models. The McNemar's statistic and its p-value are printed.
The ```back_project_ci``` script back-project the regression coefficients over the bootstrap resamplings in the original space through the PCs. For each centrality measures and inferential statistics are calculated. The script saves three 3D maps: a map with the mean contribution to prediciton of a voxel over the bootstrap resamplings, a map with the median contribution to prediction of a voxel over the bootstrap resamplings, a map of voxels with contribution to prediction significantly different from zero.
The ```atlas_based_inference``` script project a dired coeffients map (e.g., the mean coefficients map) on a disered atlas. It calculates the mean contribution of each region of interest (ROI) and calculates if the contribution of each ROI is significantly different from zero. It saves a CSV file with central tendecny measures and statistical significance for each ROI.
The ```plot_permuted_distributions``` script plots the distribution of the permuted scores under the null models against the score of the true model.

It requires the path to the input data (i.e., the predictors), the path to the covariates, the path to the target, the name of the files where to save the model parameters and the predictions.  The model are saved in pickle (.pkl) file, whereas the predictions are saved in a CSV file.

**N.B.** the CSV files must not have headers
**N.B.** in the ```make_model``` script some modifications can be made throughout the script to change the pipeline structure.

(the code expects a )
