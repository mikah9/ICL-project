import pandas as pd
from scipy.optimize import curve_fit, dual_annealing
from scipy.integrate import cumtrapz
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

c=c+1
folder_name='./raw_data/mcr_variant'
#find the files with the matching names
prefixed = [filename for filename in os.listdir(folder_name) if filename.startswith("mcr-")]
n=len(prefixed)

filename=prefixed[c]

path = folder_name+'/'+filename
df_features_uMelt = pd.read_csv(path)
target=filename.split('_')[0]

# extracting the data
name=filename.split('.')[0]
t_uMelt=[]
data_start=0
count=0
for header in df_features_uMelt.loc[:,'uMelt']:
    try:
        float(header)
    except:
        count=count+1
        continue
    else:
        t_uMelt.extend([float(header)])
        if data_start==0:
            data_start=count
if df_features_uMelt['DNA-UTAH.ORG'].iloc[-1]=='undefined':
    t_uMelt=t_uMelt[:-1]
    MC=df_features_uMelt.iloc[data_start:-1,1].astype('float')
    MC.index=t_uMelt
else:     
    MC=df_features_uMelt.iloc[data_start:,1].astype('float')
    MC.index=t_uMelt


def step(t, t1,L,b1):
    return (np.sign(t1-t)*0.5+0.5)*(L*t+b1)

def sigmoid_fit(t, a,b,k, t0,L,b1):
    """
    Defining the form of the function to which the curves will be fitted
    """
    return a / (1 + np.exp(k*(t-t0)))+b+step(t,t0,L,b1)


test=MC
b=((0,-1,0,70,-0.1,0),(2*max(test),1,4,100,0,5))

p0 = [max(test)/2,0,1,np.median(t_uMelt),-0.01,1] # this is an mandatory initial guess

params,_=curve_fit(sigmoid_fit, np.array(t_uMelt),test,p0=p0,bounds=b,method='trf')

plt.plot(t_uMelt,test)

plt.plot(t_uMelt,sigmoid_fit(np.array(t_uMelt), *params))