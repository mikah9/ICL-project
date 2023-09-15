import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from functions_fitting import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit,GroupKFold,train_test_split, cross_val_score
from pickle import load,dump
import itertools
import similaritymeasures

def plot_on_axis(axs,i,row,col,t_exp,DataFrame,func,twin_axis,line_id,linewidth,color):
    
    if twin_axis==False:
        if id.size != 0:
            if fitting_func=='gaus3':
                [a1, b1, c1, a2, b2, c2, a3, b3, c3]=DataFrame.iloc[i,:9] 
                line_id, =axs[row][col].plot(t_exp, gaussian_fit(t_exp,a1, b1, c1, a2, b2, c2, a3, b3, c3),linewidth=linewidth,color=color)
        
            elif fitting_func=='gaus1':
                [a1, b1, c1]=DataFrame.iloc[i,:3] 
                line_id, =axs[row][col].plot(t_exp, single_gaussian_fit(t_exp,a1, b1, c1),linewidth=linewidth,color=color)
    else:


nrows=5
ncols=2

fig, axs = plt.subplots(nrows, ncols, figsize=(5, 10))
#the temperature range for this specific dataset
t_exp=[]
data_start=0
count=0
for header in df_features_original.columns:
    try:
        float(header)
    except:
        count=count+1
        continue
    else:
        t_exp.extend([float(header)])
        if data_start==0:
            data_start=count

plot_on_axis(axs,i,row,col,t_exp,df_exp_plot,fitting_func,is_uMelt=False,line_id=line1,color='blue')