import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import cumtrapz
import numpy as np
import re
import os
import similaritymeasures
import itertools

# Intermiediary functions
# -------------------------
# Functions used to fit the raw data to the target function 
def gaussian_fit(t, a1, b1, c1, a2, b2, c2, a3, b3, c3):
    """
    Defining the form of the function to which the curves will be fitted
    params:
        a1,b1,c1: parameters of first gaussian
        a2,b2,c2: parameters of second gaussian
        a3,b3,c3: parameters of third gaussian
    returns:
        the function defined by the parameters
    """
    a1, b1, c1, a2, b2, c2, a3, b3, c3=np.float64(a1),np.float64(b1),np.float64(c1),np.float64(a2),np.float64(b2),np.float64(c2),np.float64(a3),np.float64(b3),np.float64(c3)
    return a1 * np.exp(-((t - b1) / c1)**2) + a2 * np.exp(-((t - b2) / c2)**2) + a3 * np.exp(-((t - b3) / c3)**2)

def single_gaussian_fit(t, a1, b1, c1):
    """
    Defining the form of the function to which the curves will be fitted
    params:
        a1,b1,c1: parameters of the gaussian
    returns:
        the function defined by the parameters
    """
    a1,b1,c1=np.float64(a1),np.float64(b1),np.float64(c1)
    return a1 * np.exp(-((t - b1) / c1)**2) 

def sigmoid_gauss_fit(t, a1, b1, c1,a2, t0,k,b):
    """
    Defining the form of the function to which the curves will be fitted
    """
    return a1 * np.exp(-((t - b1) / c1)**2) + a2 / (1 + np.exp(-k*(t-t0)))+b

def get_gaussian_params(params):
    """
    Used to seperate the different feature types from the list of fitted gaussian 3 features. 
    The 3 amplitudes, 3 mean and 3 variances are grouped together.
    params:
        params: parameters of the fitted function comprise of three gaussian
    returns:
        fitA: all the amplitude parameters
        fitmu: all the mean parameters
        fitsigma: all the variance parameters
    """
    fitA = params[0::3]
    fitmu = params[1::3]
    fitsigma = params[2::3]
    return fitA, fitmu, fitsigma
    
def remove_outliers_by_group(group,group_original,threshold):
    """
    Define a function to remove outliers within each group
    params:
        group: the data group in which to find outliers
        threshold: z score threshold
    return:
        group with the outliers removed
    """
    subset = group.copy()  # Create a copy of the group
    if len(subset.index)==9:
        z_scores = np.abs(subset.sub(subset.transpose().mean(), axis=0).div(subset.transpose().std(), axis=0))  # Calculate Z-scores
    else :  
        z_scores = np.abs(subset.iloc[:-1,:].sub(subset.iloc[:-1,:].transpose().mean(), axis=0).div(subset.iloc[:-1,:].transpose().std(), axis=0))  # Calculate Z-scores
    condition=(z_scores <= threshold).all()
    return group.loc[:, condition],group_original.loc[:, condition]  # Keep values within the threshold

def fitting(fitting_func,t_exp,data):
    """
    Define a fitting function
    params:
        fitting_func: what fitting function to use
        t_exp: x axis
        data: y axis
    return:
        params: the fitting parameters
    """
    A_max=np.max(data)
    mean = np.float64(sum(t_exp * data) / sum(data))
    sigma = np.sqrt(sum(data * (t_exp-mean)**2) / sum(data))
    err=0
    if fitting_func=="gaus3":
        params=[1,1,1,1,1,1,1,1,1]

        p0 = [max(data)/2, mean, sigma, max(data)/2, mean, sigma, max(data)/2, mean, sigma]
        b = ((-A_max, 60, 0, -A_max, 60, 0,-A_max, 60, 0), (A_max, 100, 100, A_max, 100, 100, A_max, 100, 100))
        b1 = ((float('-inf'), 0, 0, float('-inf'), 0, 0, float('-inf'), 0, 0), (A_max, 100, 1000, A_max, 100, 1000, A_max, 100, 1000))
        err=0
        try:
            curve_fit(gaussian_fit, t_exp, data, p0=p0,bounds=b)
        except :
            err=1
        else:
            params, _ = curve_fit(gaussian_fit, t_exp, data, p0=p0,bounds=b)
        if err==1:
            try:
                curve_fit(gaussian_fit, t_exp, data, p0=p0,bounds=b1)
            except :
                err=2
            else:
                params, _ = curve_fit(gaussian_fit, t_exp, data, p0=p0,bounds=b1)
        if err==2:
            try:
                curve_fit(gaussian_fit, t_exp, data, p0=p0)
            except :
                pass
            else:
                params, _ = curve_fit(gaussian_fit, t_exp, data, p0=p0)
                
    if fitting_func=="gaus1":
        params=[1,1,1]

        p0 = [max(data)/2, mean, sigma]
        b = ((-A_max, 60, 0), (A_max, 100, sigma+1))
        b1 = ((float('-inf'), 0, 0), (A_max, 100, sigma+1))
        try:
            curve_fit(single_gaussian_fit, t_exp, data, p0=p0,bounds=b)
        except :
            err=1
        else:
            params, _ = curve_fit(single_gaussian_fit, t_exp, data, p0=p0,bounds=b)
        if err==1:
            try:
                curve_fit(single_gaussian_fit, t_exp, data, p0=p0,bounds=b1)
            except :
                err=2
            else:
                params, _ = curve_fit(single_gaussian_fit, t_exp, data, p0=p0,bounds=b1)
        if err==2:
            try:
                curve_fit(single_gaussian_fit, t_exp, data, p0=p0)
            except :
                pass
            else:
                params, _ = curve_fit(single_gaussian_fit, t_exp, data, p0=p0)
    if fitting_func=="antiderivate":
        b=((0,-1,0,70,-0.1,4),(2*max(data),1,4,100,0,5))
        p0 = [max(data)/2,0,1,np.median(t_exp),-0.01,4.5] # this is an mandatory initial guess
        params,_=curve_fit(sigmoid_fit, np.array(t_exp),data,p0=p0,bounds=b,method='trf')

    return params

def step(t, t1,L,b1):
    return (np.sign(t1-t)*0.5+0.5)*(L*t+b1)

def sigmoid_fit(t, a,b,k, t0,L,b1):
    """
    Defining the form of the function to which the curves will be fitted
    """
    return a / (1 + np.exp(k*(t-t0)))+b+step(t,t0,L,b1)

# Main function
# -------------------------
# Functions used to fit the raw data to the target function qnd remove outliers, only work for dPCR_Melting_Curves.csv file type

def extract_fit_to_gaussian(df_features_exp,fitting_func):
    """
    Extract raw dMC data from the experimental file dPCR_Melting_Curves.csv
    params:
        df_features_exp: raw csv data
        fitting_func: function used for fitting
    return:
        df_features_exp_fitted: table of fitting features
    """
    n,_=df_features_exp.shape

    #Empty dataframe to save the features in
    if fitting_func=="gaus3":
        features_exp = np.zeros((9, n))
    elif fitting_func=="gaus1":
        features_exp = np.zeros((3, n))
    elif fitting_func=="antiderivate":
        features_exp = np.zeros((6, n))
    exp = []

    #the temperature range for this specific dataset
    t_exp=[]
    data_start=0
    count=0
    for header in df_features_exp.columns:
        try:
            float(header)
        except:
            count=count+1
            continue
        else:
            t_exp.extend([float(header)])
            if data_start==0:
                data_start=count
    MC = np.zeros((len(t_exp),n))

    #Fitting every curve
    for i in range(n):
        dMC = df_features_exp.iloc[i,data_start:].astype('float')
        if fitting_func=="gaus3":
            params=fitting(fitting_func,t_exp,dMC)
            fitA, fitmu, fitsigma = get_gaussian_params(params)
            idx = np.argsort(fitsigma)[::-1][:3]
            features_exp[:, i] = [fitA[idx[0]],fitmu[idx[0]],fitsigma[idx[0]],fitA[idx[1]],fitmu[idx[1]],fitsigma[idx[1]],fitA[idx[2]],fitmu[idx[2]],fitsigma[idx[2]]]  
            MC[:,i] =dMC
        elif fitting_func=="gaus1":
            params=fitting(fitting_func,t_exp,dMC)
            features_exp[:, i] =params
            MC[:,i] =dMC
        elif fitting_func=="antiderivate":
            antiderivate = cumtrapz(dMC, t_exp, initial=0)
            original=-(antiderivate-antiderivate[-1])
            params=fitting(fitting_func,t_exp,original)
            features_exp[:, i] =params
            MC[:,i] =original
            
        #target names at 3rd column
        exp.extend([df_features_exp.loc[:,'Target' ][i]])

    # Code for organising the dataframe
    # df_features_exp_fitted has the features in the rows and different samples in different columns
    df_features_exp_fitted = pd.DataFrame(features_exp)
    if fitting_func=="gaus3":
        new_index = ['A1', 'mu1', 'sigma1', 'A2', 'mu2', 'sigma2','A3', 'mu3', 'sigma3']  # New row names
    elif fitting_func=="gaus1":
        new_index = ['A', 'mu', 'sigma']  # New row names
    elif fitting_func=="antiderivate":
        new_index = ['A', 'b', 'k','t0','L','b1']  # New row names
    
    df_features_exp_fitted.index = new_index
    df_features_exp_fitted.columns = exp

    df_original = pd.DataFrame(MC)
    
    df_original.index = exp
    df_original.columns = t_exp

    return df_features_exp_fitted,df_original.transpose()

def assign_unique_id(df_features_exp_fitted):
    """
    Assign unique id to data based on pannel and experiment id
    params:
        df_features_exp_fitted: table of fitting features
    return:
        f_features_exp_fitted_unique_id: entry modified with a last column that indicates its unique id
        combinations: list of unique id from the pannel and experiment id
    """
    df_features_exp = pd.read_csv('./raw_data/dPCR_Melting_Curves.csv',sep=';')

    all_pannels=np.unique(df_features_exp.iloc[:, 1])
    all_exp_id=np.unique(df_features_exp.iloc[:, 6])

    combinations = list(itertools.product(all_pannels, all_exp_id))
    _,nsample=df_features_exp_fitted.shape

    unique_id=[]

    for i in range(nsample):
        exp=[df_features_exp.iloc[i, 1],df_features_exp.iloc[i, 6]]
        test = [value for value in np.isin(combinations,exp)]
        id=[i for i in range(len(test)) if test[i].all()==True]

        unique_id.append(id[0])

    temp=pd.DataFrame(unique_id).transpose()
    temp.columns=df_features_exp_fitted.columns
    temp.rename(index={0:'unique_id'},inplace=True)

    df_features_exp_fitted_unique_id=pd.concat([df_features_exp_fitted,temp])

    return df_features_exp_fitted_unique_id,combinations

def remove_outliers(df_features_exp,df_original,fitting_func,threshold):
    """
    Remove outliers from the fitted curves
    params:
        df_features_exp: table of fitting features
        fitting_func: function used for fitting
    returns:
        df_features_exp_inliers: inliers dataframe
    """

    # Define the partial group names to match
    assay_names=[col[:5] if col[:3] == 'mcr' else col[:3] for col in df_features_exp.columns]
    partial_group_names =list(dict.fromkeys(assay_names))

    # Apply the outlier removal function within each group
    df_features_exp_inliers = pd.DataFrame()
    df_features_original_inliers = pd.DataFrame()
    for partial_name in partial_group_names:
        columns = [col for col in df_features_exp.columns if re.search(partial_name, col, re.IGNORECASE)]
        group_df,group_original_df = remove_outliers_by_group(df_features_exp[columns],df_original[columns],threshold)
        df_features_exp_inliers = pd.concat([df_features_exp_inliers, group_df],axis=1)
        df_features_original_inliers = pd.concat([df_features_original_inliers, group_original_df],axis=1)

    # Code for organising the dataframe
    # df_features_exp_fitted is transposed to have different samples in different rows

    if fitting_func=='gaus3':
        new_index = ['A1', 'mu1', 'sigma1', 'A2', 'mu2', 'sigma2','A3', 'mu3', 'sigma3','unique_id']  # New row names
    elif fitting_func=='gaus1':
        new_index = ['A', 'mu', 'sigma','unique_id']  # New row names
    elif fitting_func=="antiderivate":
        new_index = ['A', 'b', 'k','t0','L','b1','unique_id']  # New row names
    df_features_exp_inliers.index = new_index

    df_features_exp_inliers=df_features_exp_inliers.transpose()
    df_features_original_inliers=df_features_original_inliers.transpose()
    return df_features_exp_inliers,df_features_original_inliers

def uMelt_fit(folder_name,fitting_func):
    """
    Extract raw dMC data from the uMelt file mcr-*_uMelt.csv
    params:
        folder_name: location of all uMelt files
        fitting_func: function used for fitting
    returns:
        df_uMelt_fitted: fitted uMelt data from the csv files
    """

    #find the files with the matching names
    prefixed = [filename for filename in os.listdir(folder_name) if filename.startswith("mcr-")]
    n=len(prefixed)
    index = []
    if fitting_func=='gaus3':
        features = np.zeros((9, n))
    elif fitting_func=='gaus1':
        features = np.zeros((3, n))
    elif fitting_func=="antiderivate":
        features = np.zeros((6, n))
    amp=[]

    count=0
    #add information of the amplicon
    amplicon_info=[['TGGCGTTCAGCAGTCATTATGCCAGTTTCTTTCGCGTGCATAAGCCGCTGCGTAGCTATGTCAATCCGATCATGCCAATCTACTCGGTGGGTAAGCTTGCCAGTATTGAGTATAAAAAAGCCAGTGCGCCAAAAGATACCATTTATCACGCCAAAGACGCGGTACAAGCAACCAAGCCTGATATGCGTAAGCCACGCCTAGTGGTGTTCGTCGTCGGTGAGACGGCACGCGCCGATCATGTCAGCTTCAATGGCTATGAGCGCGATACTTTCCCACAGCTTGCCAAGATCGATGGCGTGACCAATTTTAGCAATGTCACATCGTGCGGCACATCGACGGCGTATTCTGTGCCGTGTATGTTCAGCTATCTGGGCGCGGATGAGTATGATGTCGATACCGCCAAATACCAAGAAAATGTGCTGGATACGCTGGATCGCTTGGGCGTAAGTATCTTGTGGCGTGATAATAATTCGGACTCAAAAGGCGTGATGGATAAGCTGCCAAAAGCGCAATTTG'],
                ['CTGTATCGGATAACTTAGGCTTTATCATCTCAATGGCGGTGGCGGTGATGGGTGCTATGCTACTGATTGTCGTGCTGTTATCCTATCGCTATGTGCTAAAGCCTGTCCTGATTTTGCTACTGATTATGGGTGCGGTGACGAGCTATTTTACCGATACTTATGGCACGGTCTATGACACCACCATGCTCCAAAATGCCATGCAAACCGACCAAGCCGAGTCTAAGGACTTGATGAATTTGGCGTTTTTTGTGCGAATTATCGGGCTTGGCGTGTTGCCAAGTGTGTTGGTCGCAGTTGCCAAAGTCAATTATCCAACATGGGGCAAAGGTCTGATTCAGCGTGCGATGACATGGGGTGTCAGCCTTGTGCTGTTGCTTGTGCCGATTGGACTATTTAGCAGTCAGTAT'],
                ['AGACACCAATCCATTTACCAGTAAATCTGGTGGCGTGATCTCCTTTAATGATGTTCGTTCGTGTGGGACTGCAACCGCTGTATCCGTCCCCTGCATGTTCTCCAATATGGGGAGAAAGGAGTTTGATGATAATCGC'],
                ['TTGCAGACGCCCATGGAATACCAACAACTTGGCCTAGATGCGAAGAATGCCAGTCGTAACCCGAACACTAAACCTAACTTATTAGTGGTTGTTGTGGGTGAAACTGCGCGCTCAATGAGCTATCAATATTATGGATATAACAAGCCAACCAATGCTCATACCCAAAATCAGGGGCTGATTGCGTTTAACGATACTAGCTCATGCGGC'],
                ['GGTTGAGCGGCTATGAACGACAAACCACCCCTGAGTTGGCCGCACGCGACGTGATCAATTTTTCCGATGTCACCAGTTGCGGGACGGATACGGCTACATCCCTTCCCTGCATGTTTTCCCTCAATGGTCGGCGCGACTACGACGAACGCCAGATTCGTCGGCGCGAGTCCGTGCTGCACGTTTTAAACCGTAGTGACGTCAACATTC'],
                ['GTCCGGTCAATCCCTATCTGTTGATGAGCGTGGTCGCTTTATTTTTGTCAGCGACAGCAAACCTAACTTTCTTTGATAAAATCACCAATACTTATCCGATGGCACAAAACGCAGGCTTTGTGATCTCAACGGCGCTTGTGCTATTTGGGGCGATGCTATTGATTACTGTGCTGTTATCGTATCGCTATGTGCTTAAGCCTGTGTTGATTTTGCTGCTTATCATGGGTGCGGTGACGAGCTATTTTACCGATACTTATGGCACCGTTTATGACACCACCATGCTCCAAAATGCCTTGCAAACTGACCAAGCCGAGTCTAAGGACTTGATGAATATGGCGTTTTTTGTGCGGATTATCGGGCTTGGCGTGTTGCCAAGTATCTTGGTGGCGTGGGTCAAGGTGGATTATCCGACATTGGGTAAGAGTCTGATTCAGCGTGCGATGACTTGGGGTGTGGCAGTGGTGATGGCACTTGTGCCGATTTTGGCATTTAGTAGTCACTACGCCAGTTTCTTTCGTGAACATAAGCCACTGCGTAGCTATGTCAATCCCGTGAT'],
                ['TGCTCAAGCCCTTCTTTTCGTTGTTGATCCTGACAGGCTCCATCGTCAGTTACGCCATGCTCAAATACGGCGTCATCTTCGATGCCAGCATGATCCAGAACATAGTGGAGACCAACAACAGTGAGGCGACCTCCTACCTGAATGTGCCGGTCGTGCTCTGGTTCCTGCTGACCGGTGTGTTGCCCATGGTGGTGCTCTGGTCGCTGAAGGTGCGCTATCCGGCAAACTGGTACAAGGGGCTGGCCATCAGGGCTGGTGCTCTGGCCTTCTCGCTGCTGTTCGTGGGAGGCGTTGCCGCACTTTACTATCAGGATTACGTCTCGATCGGCCGCAATCACCGGATCCTGGGCAAGCAGATAGTGCCGGCCAACTATGTCAACGGCATCTACAAATATGCCCGCGACGTGGTATTTGCTACCCCCATCCCTTATCAACCGCTGGGGACTGATGCCAAAGTCGTCGCCAA'],
                ['CGAAACCGCCAGAGCACAGAATTTCCAGCTGAATGGCTATTCGCGGGTAACCAACCCCTATCTTTCCAGACGACATGATGTTATCAGTTTCAAAAATGTGTCGTCATGCGGAACGGCTACCGCAATATCACTACCCTGCATGTTCTCGCGAATGTCACGTAACGAATACAATGAAGTCCGTGCCGCATCAGAAGAAAACTTGCTGGATATCCTTAAACGTACAGGTGTTGAGGTGCTATGGCGCAACAATAACAATGGTGGTTGTAAGGGAATCTGCAAGCGAGTACCCACAGATGATATGCCGGCAATGAAAGTAATTGGGGAATGTGTTAACAAAGATGGTACATGCTTTGATGAGGTGTTATTAAATCAACTCTCATCCCGAATTAATGCAATGCAGGGTGATGCGCTTATTGTTTTACATCAAATGGGCAGTCATGGACCAACATATTTTGAACGTTATCCGTCTACAAGTAAAGTCTTTAGCCCAACTTGCGACAGCAACCTGATCGAAAAATGCTCAAATAAAGAACTGGTCAATACATACGACAATACGCTAGTTTATACTGATCGTATGCTGAGCAAAACTATTGAACTGTTGCAACGTTATTCCGGGA'],
                ['TATAAAGGCATTGCTTACCGTTTGCTCTCCGTGCTGGCATCGTTGAGTTTGATTGCAGGTGTTGCCGCACTTTATTATCAGGATTATGCCTCTGTCGGCCGCAATAACTCGACATTGAATAAAGAGATCATCCCGGCGAACTACGCTTACAGCACTTTCCAGTATGTTAAGGATACGTACTTTACGACTAAAGTGCCTTTCC']]
    amplicon_info=pd.DataFrame([amplicon_info]).transpose()
    amplicon_info.index=['mcr-1','mcr-2','mcr-3','mcr-4','mcr-5','mcr-6','mcr-7','mcr-8','mcr-9']
    amplicon_info.columns=['sequence']
    nfile=0
    #fit every uMelt curve
    for filename in prefixed:
        #read data from file
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
        if fitting_func=="gaus3" or fitting_func=="gaus1":
            if df_features_uMelt['All Rights Reserved'].iloc[-1]=='undefined':
                t_uMelt=t_uMelt[:-1]
                dMC=df_features_uMelt.iloc[data_start:-1,2].astype('float')
                dMC.index=t_uMelt
            else:     
                dMC=df_features_uMelt.iloc[data_start:,2].astype('float')
                dMC.index=t_uMelt

        elif fitting_func=="antiderivate":
            if df_features_uMelt['DNA-UTAH.ORG'].iloc[-1]=='undefined':
                t_uMelt=t_uMelt[:-1]
                MC=df_features_uMelt.iloc[data_start:-1,1].astype('float')
                MC.index=t_uMelt
            else:     
                MC=df_features_uMelt.iloc[data_start:,1].astype('float')
                MC.index=t_uMelt

        if fitting_func=="gaus3":  
            params=fitting(fitting_func,t_uMelt,dMC)    
            fitA, fitmu, fitsigma = get_gaussian_params(params)
            idx = np.argsort(fitsigma)[::-1][:3]
            features[:,nfile]= [fitA[idx[0]],fitmu[idx[0]],fitsigma[idx[0]],fitA[idx[1]],fitmu[idx[1]],fitsigma[idx[1]],fitA[idx[2]],fitmu[idx[2]],fitsigma[idx[2]]]
            index.extend([name])
            amp.extend(amplicon_info.loc[target,'sequence'])
            nfile=nfile+1
        elif fitting_func=="gaus1":
            params=fitting(fitting_func,t_uMelt,dMC)
            features[:,nfile]= params
            index.extend([name])
            amp.extend(amplicon_info.loc[target,'sequence'])
            nfile=nfile+1
        elif fitting_func=="antiderivate":
            original=MC
            params=fitting(fitting_func,t_uMelt,original)
            params[-1]=5
            features[:,nfile]= params
            index.extend([name])
            amp.extend(amplicon_info.loc[target,'sequence'])
            nfile=nfile+1
    
    # Code for organising the dataframe
    df_uMelt_fitted = pd.DataFrame(features)
    amp=pd.DataFrame(amp)
    df_uMelt_fitted=pd.concat([df_uMelt_fitted,amp.transpose()]).transpose()

    df_uMelt_fitted.index=index

    if fitting_func=="gaus3":
        column=['A1','mu1','sigma1','A2','mu2','sigma2','A3','mu3','sigma3','amplicon_seq']
    elif fitting_func=="gaus1":
        column=['A','mu','sigma','amplicon_seq']
    elif fitting_func=="antiderivate":
        column=['A', 'b', 'k','t0','L','b1','amplicon_seq']
    df_uMelt_fitted.columns=column

    return df_uMelt_fitted   