import pandas as pd
import numpy as np
import re
import itertools
from pickle import dump,load
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.neural_network import MLPRegressor

# Intermiediary functions
# -------------------------
# Functions used to fit the raw data to the target function 

def MLP_param_testing(Input_train,Output_train,groups):
    #k fold splitting by group for cross validation
    n_splits=5
    gkf = GroupKFold(n_splits)   

    hidden_layers=[2,4,8,16]
    nb_neurons=[80,160,320]
    activation_functions=['tanh', 'logistic']
    solvers=['adam','sgd']
    alpha_list=np.logspace(-8,-4, num=5, base=10)

    #empty test dataframe to store cross validation results
    test=pd.DataFrame()

    #Hyperparameter test number, needed to prevent dataframe from overwritting each other
    counter=1

    parameter_combinations = itertools.product(hidden_layers, nb_neurons, activation_functions, solvers, alpha_list)

    # Iterate over the parameter combinations
    for layers, neurons, activation, solver, alpha in parameter_combinations:
            
            hidden_layer_sizes=[neurons]*layers
            regressor = MLPRegressor(hidden_layer_sizes, activation=activation, solver=solver,alpha=alpha, max_iter=1000)

            dat=pd.DataFrame([layers,neurons,activation,solver, alpha])
            dat.index=['layers','neurons','activation','solver','alpha']

            scores = pd.DataFrame(cross_val_score(regressor, Input_train, Output_train, groups=groups, cv=gkf, scoring='neg_mean_squared_error'))

            step=pd.concat([dat,scores],axis=0)
            step.columns=[counter] #prevent overwritting by giving it a unique name
            test=pd.concat([test,step],axis=1)
            counter+=1

    start_row = 5  # Starting row (inclusive)
    end_row = 9  # Ending row (inclusive)

    column_averages = test.iloc[start_row:end_row+1,:].mean()
    test.loc[test.index[-1] + 1] = column_averages

    best_column=pd.DataFrame(test.iloc[:,pd.to_numeric(test.iloc[-1]).idxmax()]) 

    return best_column

# Main function
# -------------------------
# Functions used in Main code

def param_evaluation(df_Output,df_Input):
    """
    Evaluate the optimal random forest architecture when leaving one target out as test set
    """

    #Determine the groups (assays)
    assay_names=[row[:5] for row in df_Input.index]
    partial_group_names=np.unique(assay_names)
    _, _, groups = np.unique(assay_names, return_index=True, return_inverse=True)

    #Cross validation and final testing data splitting
    group_param=pd.DataFrame()
    n=len(partial_group_names)

    gkf= GroupKFold(n_splits=n)

    # Random forest
    # -------------------------
    # Random forest regressor optimisation
    col=[]
    
    for i, (train_id, test_id) in enumerate(gkf.split(df_Input, df_Output, groups)):
        train_group=np.unique(groups[train_id])
        test_group=np.unique(groups[test_id])
        test_group_name=partial_group_names[test_group]
        if test_group_name[0]=='mcr-8':
            continue
        col.append(test_group_name[0])
        print(f" Train group={train_group}, test group={test_group}") #for verification purpose

        Input_train, Input_test=df_Input.iloc[train_id,:],df_Input.iloc[test_id,:]
        Output_train, Output_test=df_Output.iloc[train_id,:],df_Output.iloc[test_id,:]

        #remove badly fitted mcr-8
        Input_train[['mcr-8' not in row for row in Input_train.index]]
        Output_train[['mcr-8' not in row for row in Output_train.index]]
        
        assay_names=[col[:5] for col in Input_train.index]
        #sub groups for cross validation from the training data
        _, _, cv_groups = np.unique(assay_names, return_index=True, return_inverse=True) 

        best_column=MLP_param_testing(Input_train,Output_train,cv_groups)

        group_param=pd.concat([group_param,best_column],axis=1)
        
    group_param.columns=col

    return group_param

def scale_fitted(Input_train,Output_train):
    """
    Standard scale the fitted experimental and fitted curve on exp and uMelt data.
    """
    assay_names=[ind[:5] if ind[:3] == 'mcr' else ind[:3] for ind in Input_train.index]
    _,m,_=np.unique(assay_names, return_index=True, return_inverse=True)

    indexes=Output_train.index
    columns=Output_train.columns
    #create and fit Output scaler
    scaler_output= MaxAbsScaler()
    Output_train=pd.DataFrame(scaler_output.fit_transform(Output_train))
    Output_train.index=indexes
    Output_train.columns=columns

    indexes=Input_train.index
    columns=Input_train.columns
    #create and fit Input scaler
    scaler_input= MaxAbsScaler()
    scaler_input.fit(Input_train.iloc[m,:])
    Input_train=pd.DataFrame(scaler_input.transform(Input_train))
    Input_train.index=indexes
    Input_train.columns=columns

    return Input_train,Output_train,scaler_output,scaler_input

def MLP(df_Output,df_Input,regressor_trained,method,fitting_func,opt_regressor):
    """
    Training and testing using the best parameters from group_param
    """
    Output_prediction=pd.DataFrame()

    if method=='leak':
        assay_names=[row[:5] for row in df_Input.index]
        _, _, groups = np.unique(assay_names, return_index=True, return_inverse=True)
        partial_group_names =list(dict.fromkeys(assay_names))
        Input_train, Input_test, Output_train, Output_test = train_test_split(
            df_Input, df_Output, test_size=0.2, #transposed
            random_state=42) #random_state = seed

        Input_train_scaled,Output_train_scaled,scaler_output,scaler_input=scale_fitted(Input_train,Output_train)
        dump(scaler_output, open(f'MLP/scaler_output_{fitting_func}_{method}.pkl', 'wb'))
        dump(scaler_input, open(f'MLP/scaler_input_{fitting_func}_{method}.pkl', 'wb'))
    
        trained=regressor.fit(Input_train_scaled,Output_train_scaled)

        dump(trained, open(f'MLP/MLP_{fitting_func}_{method}.pkl', 'wb'))
        #testing
        Input_test_scaled=pd.DataFrame(scaler_input.transform(Input_test))
        Input_test_scaled.index=Input_test.index
        Input_test_scaled.columns=Input_test.columns

        Output_predicted_scaled=pd.DataFrame(trained.predict(Input_test_scaled))
        Output_predicted_scaled.index=Output_test.index
        Output_predicted_scaled.columns=Output_test.columns

        Output_prediction=pd.DataFrame(scaler_output.inverse_transform(Output_predicted_scaled))
        Output_prediction.index=Output_predicted_scaled.index
        Output_prediction.columns=Output_predicted_scaled.columns

    elif method=='target':
        n_splits=np.unique([row[:5] for row in df_Output.index]).size
        gkf = GroupKFold(n_splits)   
        #Determine the groups (targets)
        assay_names=[row[:5] for row in df_Input.index]
        partial_group_names=np.unique(assay_names)
        n, m, groups = np.unique(assay_names, return_index=True, return_inverse=True)
    elif method=='id_variant':
        n_splits=5
        gkf = GroupKFold(n_splits)   
        #Determine the groups (targets)
        unique_id=df_Output['unique_id'].values
        partial_group_names=np.unique(unique_id)
        n, m, groups = np.unique(unique_id, return_index=True, return_inverse=True)

    df_Output=df_Output.iloc[:,:-1]

    if method!='leak':
        for i, (train_id, test_id) in enumerate(gkf.split(df_Input, df_Output, groups)):
            train_group=np.unique(groups[train_id])
            test_group=np.unique(groups[test_id])
            test_group_name=partial_group_names[test_group]
            print(f" Train group={train_group}, test group={test_group}") #for verification purpose

            if regressor_trained==False:
                Input_train, Input_test=df_Input.iloc[train_id,:],df_Input.iloc[test_id,:]
                Output_train, Output_test=df_Output.iloc[train_id,:],df_Output.iloc[test_id,:]

                #Scale train values
                Input_train_scaled,Output_train_scaled,scaler_output,scaler_input=scale_fitted(Input_train,Output_train)
                opt_regressor.fit(Input_train_scaled,Output_train_scaled)
                
                #testing
                Input_test_scaled=pd.DataFrame(scaler_input.transform(Input_test))
                Input_test_scaled.index=Input_test.index
                Input_test_scaled.columns=Input_test.columns

                Output_predicted_scaled=pd.DataFrame(opt_regressor.predict(Input_test_scaled))
                Output_predicted_scaled.index=Output_test.index
                Output_predicted_scaled.columns=Output_test.columns

                Output_predicted=pd.DataFrame(scaler_output.inverse_transform(Output_predicted_scaled))
                Output_predicted.index=Output_predicted_scaled.index
                Output_predicted.columns=Output_predicted_scaled.columns

                Output_prediction=pd.concat([Output_prediction,Output_predicted])
                
                dump(opt_regressor, open(f'MLP/MLP_{method}_{test_group_name[0]}.pkl', 'wb'))
                dump(scaler_output, open(f'MLP/scaler_output_{method}_{test_group_name[0]}.pkl', 'wb'))
                dump(scaler_input, open(f'MLP/scaler_input_{method}_{test_group_name[0]}.pkl', 'wb'))
            else:
                Input_test=df_Input.iloc[m,:]
                Input_test[['mcr-8' not in row for row in Input_test.index]]

                #testing
                Input_test_scaled=pd.DataFrame(scaler_input.transform(Input_test))
                Input_test_scaled.index=Input_test.index
                Input_test_scaled.columns=Input_test.columns
                
                opt_regressor =load(open(f'MLP/{method}_{test_group_name[0]}.pkl', 'rb'))
                scaler_output =load(open(f'MLP/scaler_output_{method}_{test_group_name[0]}.pkl', 'rb'))
                scaler_input =load(open(f'MLP/scaler_input_{method}_{test_group_name[0]}.pkl', 'rb'))

                Output_predicted_scaled=pd.DataFrame(opt_regressor.predict(Input_test_scaled))
                Output_predicted_scaled.index=Output_test.test_group_name
                Output_predicted_scaled.columns=Output_test.Input_test.columns[:-2]

                Output_predicted=pd.DataFrame(scaler_output.inverse_transform(Output_predicted_scaled))
                Output_predicted.index=Output_predicted_scaled.index
                Output_predicted.columns=Output_predicted_scaled.columns

                Output_prediction=pd.concat([Output_prediction,Output_predicted])

    return Output_prediction