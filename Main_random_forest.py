from functions_random_forest import *
from functions_plot import *

# Parameters tuning
# -------------------------
# Code for hyperparameter testing with cross validation

# load data
df_Output = pd.read_csv('./preprocessed_data/df_Output.csv',index_col=0)
df_Input = pd.read_csv('./preprocessed_data/df_Input.csv',index_col=0)

# evaluating the parameters and saving the results
group_param=param_evaluation(df_Output,df_Input)
group_param.to_csv('./random_forest/group_param.csv', index=True)

# Training random forest
# -------------------------
# Code for training random forest

group_param = pd.read_csv('./random_forest/group_param.csv',index_col=0)
Output_prediction=random_forest(df_Output,df_Input,group_param,regressor_trained=False)
Output_prediction.to_csv('./random_forest/Output_prediction.csv', index=True)

# verifying the length of Output prediction, 'mcr-8' not used due to poor fitting
verif=df_Input[['mcr-8' not in row for row in df_Input.index]]

# Plot
# -------------------------
# Plotting the results
df_Output = pd.read_csv('./preprocessed_data/df_Output.csv',index_col=0)
Output_prediction = pd.read_csv('./random_forest/Output_prediction.csv',index_col=0)

plot_random_forest_result(df_Output,Output_prediction,show_umelt=True)
plot_random_forest_result(df_Output,Output_prediction,show_umelt=False)


df_Output = pd.read_csv('./preprocessed_data/df_Output.csv',index_col=0)
Output_prediction=random_forest(df_Output,df_Input,group_param,regressor_trained=True)
scaler=load(open('preprocessed_data/scaler.pkl', 'rb'))
plot_random_forest_result(df_Output,Output_prediction,show_umelt=True)