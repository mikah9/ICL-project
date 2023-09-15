from functions_preprocessing import *
from functions_plot import *
from pickle import dump

# Variables
# -------------------------
# Variables

fitting_func='gaus3'

# Data Loading 
# -------------------------
# Code for loading and preprocessing data
try:
    id_variant_association=pd.read_csv('./preprocessed_data/id_variant_association'+f'_{fitting_func}.csv', index_col=0)
except:
    id_variant_association=pd.DataFrame()

df_features_exp_inliers_unique_id=pd.read_csv('./fitted_data/df_features_exp_inliers_unique_id'+f'_{fitting_func}.csv', index_col=0)
df_features_uMelt_variant_fitted=pd.read_csv('./fitted_data/df_features_uMelt_variant_fitted'+f'_{fitting_func}.csv', index_col=0)

# Preprocessing
# -------------------------
# Finding experiment_id and uMelt variant association and preparing input and output data

df_Output,df_Input, id_variant_association= Input_output_prep_unique_id_variant(df_features_exp_inliers_unique_id,df_features_uMelt_variant_fitted,fitting_func,id_variant_association=id_variant_association)

# Saving
# -------------------------
# Saving and loading data

df_Output.to_csv('./preprocessed_data/df_Output'+f'_{fitting_func}.csv', index=True)
df_Input.to_csv('./preprocessed_data/df_Input'+f'_{fitting_func}.csv', index=True)
id_variant_association.to_csv('./preprocessed_data/id_variant_association'+f'_{fitting_func}.csv', index=True)

# Plots
# -------------------------
# Code for plotting for visual verification
plot_compare_fitted(df_features_exp_inliers_unique_id,df_features_uMelt_variant_fitted,fitting_func,id_variant_association=id_variant_association)