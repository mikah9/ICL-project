import pandas as pd
import numpy as np
import re
import similaritymeasures
from functions_fitting import *

# Intermiediary functions
# -------------------------
# Functions used to fit the raw data to the target function 

def find_closest_uMelt_variant(df_features_exp_unique_id, df_features_uMelt_variant_target,fitting_func):
    """
    Finding the closest match in the uMelt prediction for this target and the group of unique id experimental curves
    params:
        df_features_exp_unique_id: only contains the uMelt variants of a single target
        df_features_exp_unique_id: only contains the experimental prediction of a single unique id
    returns:
        closest_uMelt_variant: unique id in first column and closest variant name in second column
    """
    t_exp= np.arange(65, 97, 0.5)
    exp_size,_=df_features_exp_unique_id.shape
    uMelt_size,_=df_features_uMelt_variant_target.shape

    similarity_matrix=np.zeros((exp_size, uMelt_size))

    for i in range(exp_size):
        if fitting_func=='gaus3':
            [a1, b1, c1, a2, b2, c2, a3, b3, c3]=df_features_exp_unique_id.iloc[i,:-1] 
        elif fitting_func=='gaus1':
            [a1, b1, c1]=df_features_exp_unique_id.iloc[i,:-1] 
        elif fitting_func=='antiderivate':
            [a,b,k,t0,L,b1]=df_features_exp_unique_id.iloc[i,:6] 
        uMelt1=c3
        exp_data=np.zeros((len(t_exp), 2))
        exp_data[:, 0] = t_exp
        if fitting_func=='gaus3':
            exp_data[:, 1]=gaussian_fit(t_exp,a1, b1, c1, a2, b2, c2, a3, b3, c3)
        elif fitting_func=='gaus1':
            exp_data[:, 1]=single_gaussian_fit(t_exp,a1, b1, c1)
        elif fitting_func=='antiderivate':
            exp_data[:, 1]=sigmoid_fit(np.array(t_exp),a,b,k, t0,L,b1)
        
        A=max(exp_data[:, 1])
        exp_data[:, 1]=exp_data[:, 1]/A
        for j in range(uMelt_size):
            if fitting_func=='gaus3':
                [a1, b1, c1, a2, b2, c2, a3, b3, c3]=df_features_uMelt_variant_target.iloc[j,:-1]
            elif fitting_func=='gaus1':
                [a1, b1, c1]=df_features_uMelt_variant_target.iloc[j,:-1] 
            elif fitting_func=='antiderivate':
                [a,b,k,t0,L,b1]=df_features_uMelt_variant_target.iloc[j,:6] 
            uMelt_data=np.zeros((len(t_exp), 2))
            uMelt_data[:, 0] = t_exp
            if fitting_func=='gaus3':
                uMelt_data[:, 1]=gaussian_fit(t_exp,a1, b1, c1, a2, b2, c2, a3, b3, c3)
            elif fitting_func=='gaus1':
                uMelt_data[:, 1]=single_gaussian_fit(t_exp,a1, b1, c1)
            elif fitting_func=='antiderivate':
                uMelt_data[:, 1]=sigmoid_fit(np.array(t_exp),a,b,k, t0,L,b1)
            uMelt2=c3
            A=max(uMelt_data[:, 1])
            uMelt_data[:, 1]=uMelt_data[:, 1]/A

            dist=abs(uMelt1-uMelt2)
            # dist= similaritymeasures.area_between_two_curves(exp_data, uMelt_data)
            similarity_matrix[i,j]=dist

    similairity_mean=np.mean(similarity_matrix,axis=0)
    id=similairity_mean.argmin()

    closest_uMelt_variant=df_features_uMelt_variant_target.index[id]

    return closest_uMelt_variant

# Main function
# -------------------------
# Functions used to fit the raw data to the target function qnd remove outliers, only work for dPCR_Melting_Curves.csv file type

def Input_output_prep_unique_id_variant(df_features_exp_inliers_unique_id,df_features_uMelt_variant_fitted,fitting_func,id_variant_association=pd.DataFrame()):
    """
    Duplicating the uMelt predictions to match the number of experimental features per target
    """
    if fitting_func=='gaus3':
        df_features_uMelt_variant_fitted=df_features_uMelt_variant_fitted.drop(df_features_uMelt_variant_fitted[df_features_uMelt_variant_fitted.mu2 ==1].index)
    df_features_exp_inliers_unique_id=df_features_exp_inliers_unique_id[['ntc' not in row for row in df_features_exp_inliers_unique_id.index]]
    df_features_exp_inliers_unique_id=df_features_exp_inliers_unique_id[['mcr-8' not in row for row in df_features_exp_inliers_unique_id.index]]

    unique_id=np.unique(df_features_exp_inliers_unique_id.iloc[:,-1])

    if id_variant_association.empty==True:
        
        id_variant_association=pd.DataFrame()
        for id in unique_id:
            df_features_exp_unique_id=df_features_exp_inliers_unique_id.loc[df_features_exp_inliers_unique_id['unique_id'] == id]
            target=list(dict.fromkeys([row[:5] for row in df_features_exp_unique_id.index]))
            df_features_uMelt_variant_target=df_features_uMelt_variant_fitted[[target[0] in row for row in df_features_uMelt_variant_fitted.index]]

            closest_uMelt_variant=find_closest_uMelt_variant(df_features_exp_unique_id, df_features_uMelt_variant_target,fitting_func)
            id_variant=pd.DataFrame([id,closest_uMelt_variant], index=["unique_id","variant_name"]).transpose()

            id_variant_association=pd.concat([id_variant_association,id_variant],ignore_index=True)
            print(id)
   
    #get the indices of all unique_ids
    length,_=df_features_exp_inliers_unique_id.shape

    _,m, _ = np.unique(df_features_exp_inliers_unique_id.iloc[:,-1], return_index=True, return_inverse=True)
    m=np.sort(m)
    m=np.append(m,length)

    #Empty datafram for final data
    df_Input=pd.DataFrame()
    df_Output=pd.DataFrame()

    for i in range(len(m)-1):
        #output prep
        exp=df_features_exp_inliers_unique_id.iloc[m[i]:m[i+1],:]
        df_Output=pd.concat([df_Output,exp])

        #input_prep
        unique_id=df_features_exp_inliers_unique_id.iloc[m[i],-1]
        row=id_variant_association.loc[id_variant_association["unique_id"]==unique_id]
        variant_name=row["variant_name"]
        n,_=exp.shape

        row_with_value = df_features_uMelt_variant_fitted.loc[variant_name]

        Amplicon_length=row_with_value.loc[:,'amplicon_seq'].str.len()
        GC_content=(row_with_value.loc[:,'amplicon_seq'].str.count('G')+row_with_value.loc[:,'amplicon_seq'].str.count('C'))/Amplicon_length
        df_features_uMelt_extended=pd.concat([row_with_value.iloc[:,:-1],Amplicon_length,GC_content], axis=1,ignore_index=True)
        df_features_uMelt_extended.index =[variant_name]
        
        uMelt_column=pd.concat([df_features_uMelt_extended] * n)
        df_Input=pd.concat([df_Input, uMelt_column])

    df_Output.columns=df_features_exp_inliers_unique_id.columns
    df_Input.columns=np.append(df_features_exp_inliers_unique_id.columns[:-1].values,['Amplicon length','GC content'])

    return df_Output, df_Input, id_variant_association