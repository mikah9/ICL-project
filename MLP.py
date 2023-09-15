import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from functions_plot import *
from functions_MLP import *
from pickle import load,dump

# Variables
# -------------------------
# Variables

fitting_func='antiderivate'
method='target' #'id_variant', 'target', 'leak'
regressor_trained=False

activation='tanh'
solver='sgd'
max_iter=1000
alpha=0.0001
tol=0
n_iter_no_change =1000

# Data Loading 
# -------------------------
# Code for loading and preprocessing data

df_Input = pd.read_csv('preprocessed_data/df_Input'+f'_{fitting_func}.csv', index_col=0)
df_Output = pd.read_csv('preprocessed_data/df_Output'+f'_{fitting_func}.csv', index_col=0)

df_Input=df_Input[['mcr-8' not in row for row in df_Input.index]]
df_Output=df_Output[['mcr-8' not in row for row in df_Output.index]]

#create MLP
hidden_layer_sizes=[160]*2
regressor = MLPRegressor(hidden_layer_sizes, activation=activation,
solver=solver, max_iter=max_iter,alpha=alpha,tol=tol,n_iter_no_change =n_iter_no_change)

# MLP
# -------------------------
# Code for MLP training

Output_prediction=MLP(df_Output,df_Input,regressor_trained,method,fitting_func,regressor)

# Saving
# -------------------------
# Saving and loading data
#save
Output_prediction.to_csv('./MLP/Output_prediction'+f'_{fitting_func}_{method}.csv', index=True)
#load
Output_prediction=pd.read_csv('./MLP/Output_prediction'+f'_{fitting_func}_{method}.csv', index_col=0)

plot_MLP_result(df_Output,Output_prediction,method,fitting_func)