import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit,GroupKFold,train_test_split, cross_val_score
from pickle import load, dump

# Intermiediary functions
# -------------------------
# Functions used to calculate the cross validation score

def random_forest_param_testing(Input_train,Output_train,groups):
    #k fold splitting by group for cross validation
    n_splits=5
    gkf = GroupKFold(n_splits)   

    max_features=[0.1,0.5,1.0,'log2','sqrt']
    n_estimators=[10,100,1000]
    max_depth=[2,4,6,8]

    parameter_combinations = itertools.product(max_features,n_estimators,max_depth)

    #empty test dataframe to store cross validation results
    test=pd.DataFrame()

    #Hyperparameter test number, needed to prevent dataframe from overwritting each other
    counter=1

    for max_features, n_estimators, max_depth in parameter_combinations:
        arch=pd.DataFrame([max_features,n_estimators,max_depth])

        regressor = RandomForestRegressor(max_features=max_features,max_depth=max_depth,n_estimators=n_estimators, random_state=0)

        score= pd.DataFrame(cross_val_score(regressor, 
                                            Input_train, 
                                            Output_train, 
                                            groups=groups, 
                                            cv=gkf, 
                                            scoring='neg_mean_squared_error'))

        step=pd.concat([arch,score],axis=0)
        step.index=['max features','n estimators','max depth','negative mse  fold 1','negative mse  fold 2','negative mse  fold 3','negative mse  fold 4','negative mse  fold 5']
        step.columns=[counter] #prevent overwritting by giving it a unique name
        test=pd.concat([test,step],axis=1)
        counter+=1

    #scores location
    start_row = 3  # Starting row index (inclusive)
    end_row = 7  # Ending row index (inclusive)

    column_averages = pd.DataFrame(test.iloc[start_row:end_row+1,:].mean())
    test=pd.concat([test,column_averages.transpose()],axis=0)
    
    best_column=pd.DataFrame(test.iloc[:,pd.to_numeric(test.iloc[-1]).idxmax()])   

    return best_column

# Main functions
# -------------------------
# Functions used to determine the best architecture

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

    gkf= GroupKFold(n_splits=9)

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

        best_column=random_forest_param_testing(Input_train,Output_train,cv_groups)

        group_param=pd.concat([group_param,best_column],axis=1)
        
    group_param.columns=col

    return group_param

def random_forest(df_Output,df_Input,group_param,regressor_trained):
    """
    Training and testing using the best parameters from group_param
    """
    Output_prediction=pd.DataFrame()

    n_splits=9
    gkf = GroupKFold(n_splits)   
    #Determine the groups (targets)
    assay_names=[row[:5] for row in df_Input.index]
    partial_group_names=np.unique(assay_names)
    n, m, groups = np.unique(assay_names, return_index=True, return_inverse=True)

    for i, (train_id, test_id) in enumerate(gkf.split(df_Input, df_Output, groups)):
        train_group=np.unique(groups[train_id])
        test_group=np.unique(groups[test_id])
        test_group_name=partial_group_names[test_group]
        if test_group_name[0]=='mcr-8':
            continue
        print(f" Train group={train_group}, test group={test_group}") #for verification purpose

        if regressor_trained==False:
            Input_train, Input_test=df_Input.iloc[train_id,:],df_Input.iloc[test_id,:]
            Output_train, Output_test=df_Output.iloc[train_id,:],df_Output.iloc[test_id,:]

            #remove badly fitted mcr-8
            Input_train[['mcr-8' not in row for row in Input_train.index]]
            Output_train[['mcr-8' not in row for row in Output_train.index]]

            try:
                max_features=float(group_param.loc['max features',test_group_name[0]])
            except:
                max_features=group_param.loc['max features',test_group_name[0]]
            n_estimators=int(group_param.loc['n estimators',test_group_name[0]])
            max_depth=int(group_param.loc['max depth',test_group_name[0]])
            
            opt_regressor = RandomForestRegressor(max_features=max_features,max_depth=max_depth,n_estimators=n_estimators, random_state=0)
            opt_regressor.fit(Input_train,Output_train)
            Output_predicted=pd.DataFrame(opt_regressor.predict(Input_test))
            Output_predicted.index=Output_test.index
            Output_predicted.columns=Output_test.columns

            Output_prediction=pd.concat([Output_prediction,Output_predicted])
            
            dump(opt_regressor, open(f'random_forest/{test_group_name[0]}.pkl', 'wb'))
        else:
            Input_test=df_Input.iloc[m,:]
            Input_test[['mcr-8' not in row for row in Input_test.index]]
            
            opt_regressor =load(open(f'random_forest/{test_group_name[0]}.pkl', 'rb'))
            Output_predicted=pd.DataFrame(opt_regressor.predict(Input_test.loc[test_group_name]))
            Output_predicted.index=test_group_name
            Output_predicted.columns=Input_test.columns[:-2]

            Output_prediction=pd.concat([Output_prediction,Output_predicted])

    return Output_prediction