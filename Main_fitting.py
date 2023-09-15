import pandas as pd
from functions_fitting import *
from functions_plot import *

# Variables
# -------------------------
# uMelt files are all same format, names need to follow reference in uMelt_naming
# experimental file should be readable as long as the headers contains the temperatures and only the temperature as numbers

fitting_func='antiderivate'
uMelt_folder='./raw_data/mcr_variant'
df_features_exp = pd.read_csv('./raw_data/dPCR_Melting_Curves.csv',sep=';')
outlier_thres=3

# Data Fitting
# -------------------------
# Code for loading and preprocessing data

df_features_exp_fitted,df_original=extract_fit_to_gaussian(df_features_exp,fitting_func)

# Unique id
# -------------------------
# Code for assigning unique id and removing outliers
df_features_exp_fitted_unique_id,combination=assign_unique_id(df_features_exp_fitted)
df_features_exp_inliers_unique_id,df_original_inliers=remove_outliers(df_features_exp_fitted_unique_id,df_original,fitting_func,outlier_thres)

# uMelt
# -------------------------
# Code for loading and fitting uMelt data, different format than experimental

df_features_uMelt_variant_fitted=uMelt_fit(uMelt_folder,fitting_func)

# Saving and loading data
# -------------------------
# Code for saving the fitted data

df_original.to_csv('./fitted_data/df_original'+f'_{fitting_func}.csv', index=True)
df_features_exp_fitted.to_csv('./fitted_data/df_features_exp_fitted'+f'_{fitting_func}.csv', index=True)
df_features_exp_fitted_unique_id.to_csv('./fitted_data/df_features_exp_fitted_unique_id'+f'_{fitting_func}.csv', index=True)
df_features_exp_inliers_unique_id.to_csv('./fitted_data/df_features_exp_inliers_unique_id'+f'_{fitting_func}.csv', index=True)
df_original_inliers.to_csv('./fitted_data/df_original_inliers'+f'_{fitting_func}.csv', index=True)
pd.DataFrame(combination).to_csv('./fitted_data/combination'+f'_{fitting_func}.csv', index=True)
df_features_uMelt_variant_fitted.to_csv('./fitted_data/df_features_uMelt_variant_fitted'+f'_{fitting_func}.csv', index=True)

df_original=pd.read_csv('./fitted_data/df_original'+f'_{fitting_func}.csv', index_col=0)
df_features_exp_fitted=pd.read_csv('./fitted_data/df_features_exp_fitted'+f'_{fitting_func}.csv', index_col=0)
df_features_exp_fitted_unique_id=pd.read_csv('./fitted_data/df_features_exp_fitted_unique_id'+f'_{fitting_func}.csv', index_col=0)
df_features_exp_inliers_unique_id=pd.read_csv('./fitted_data/df_features_exp_inliers_unique_id'+f'_{fitting_func}.csv', index_col=0)
df_original_inliers=pd.read_csv('./fitted_data/df_original_inliers'+f'_{fitting_func}.csv', index_col=0)
df_features_uMelt_variant_fitted=pd.read_csv('./fitted_data/df_features_uMelt_variant_fitted'+f'_{fitting_func}.csv', index_col=0)
combination=pd.read_csv('./fitted_data/combination'+f'_{fitting_func}.csv', index_col=0)

# Plots
# -------------------------
# Code for plotting for visual verification
plot_verif_exp(df_original_inliers,df_features_exp_inliers_unique_id,fitting_func)
plot_verif_uMelt(uMelt_folder,df_features_uMelt_variant_fitted,fitting_func)
plot_compare_fitted(df_features_exp_inliers_unique_id,df_features_uMelt_variant_fitted,fitting_func)