import matplotlib.pyplot as plt
from functions_fitting import *

# Plotting functions
# -------------------------
# Functions used to plot the fitted gaussians vs original for visual inspection

def plot_verif_exp(df_features_original,df_features_fitted,fitting_func):
    """
    Plotting the experimental fitted curves inliers and the original curves
    
    """

    #increase the resolution of the plots
    plt.rcParams['figure.dpi'] = 300

    # Code for loading and selecting curve to plot to verify the fitting
    df_features_fitted=df_features_fitted[['ntc' not in row for row in df_features_fitted.index]]

    # Random select 0.1% to plot (the order of the two dataframes are identical, different rows are different samples)
    n, _ = df_features_fitted.shape
    nsample=n//1000

    index_list=pd.DataFrame([range(n)])
    sample_index=index_list.transpose().sample(nsample)
    
    all_assay_names=[ind[:5] for ind in df_features_fitted.index]
    partial_group_names =list(dict.fromkeys(all_assay_names))
    partial_group_names.sort()

    df_exp_plot=df_features_fitted.iloc[[a for a in sample_index.index],:]
    df_original_plot=df_features_original.iloc[[a for a in sample_index.index],:]

    # Define the differeht assays that were selected
    assay_names=[ind[:5] for ind in df_exp_plot.index]
    assay_names2=[ind[:5] for ind in df_original_plot.index]

    # Code for plotting for visual verification
    # Color code: blue for exp fitted, orange for exp original, red for uMelt, green for NN predicted
    # Plotting variables
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

    for i in range(nsample):
        name = assay_names[i]
        name2 = assay_names2[i] #id of original and fitted don't necessarily match because of outlier removal
        id = np.where(np.isin(partial_group_names, name))[0]
        id2 = np.where(np.isin(partial_group_names, name2))[0]
        if id.size != 0:
            row=int(id//(ncols))
            col=int(id%(ncols))
            #fitted removed
            if fitting_func=="gaus3":
                [a1, b1, c1, a2, b2, c2, a3, b3, c3]=df_exp_plot.iloc[i,:9] 
                line1, =axs[row][col].plot(t_exp, gaussian_fit(t_exp,a1, b1, c1, a2, b2, c2, a3, b3, c3),color='blue')
    
                axs[row][col].set_ylim([0, 0.11])
            elif fitting_func=="gaus1":
                [a1, b1, c1]=df_exp_plot.iloc[i,:3] 
                line1, =axs[row][col].plot(t_exp, single_gaussian_fit(t_exp,a1, b1, c1),color='blue')
                    
                axs[row][col].set_ylim([0, 0.11])
            elif fitting_func=="antiderivate":
                [a,b,k,t0,L,b1]=df_exp_plot.iloc[i,:6] 
                line1, =axs[row][col].plot(t_exp, sigmoid_fit(np.array(t_exp),a,b,k, t0,L,b1),color='blue')
                axs[row][col].set_ylim([0, 0.8])
            #original 
            
            axs[row][col].set_title(name)
            axs[row][col].grid(True)
            axs[row][col].set_xlim([60, 100])
            row=int(id2//(ncols))
            col=int(id2%(ncols))
            dMC = df_original_plot.iloc[i,:]
            line2, =axs[row][col].plot(t_exp, dMC,color='orange')
            
    #removing empty subplots
    for ax_row in axs:
        for ax in ax_row:
            if not ax.lines:
                fig.delaxes(ax)

    fig.legend([line1, line2], ['Fitted experimental', 'Original experimental'], loc='lower right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("Original and fitted experimental data comparison on 0.1 percent of all samples"+f' {fitting_func}')
    plt.show()

def plot_verif_uMelt(uMelt_folder,df_features_fitted,fitting_func):
    """
    Plot the fitted uMelt curves with the original curves
    """

    #increase plot resolution
    plt.rcParams['figure.dpi'] = 300

    # Data Loading
    # -------------------------
    # Code for loading and selecting curve to plot to verify the fitting
    prefixed = [filename for filename in os.listdir(uMelt_folder) if filename.startswith("mcr-")]
    
    max_t_length=0
    for filename in prefixed: 
        path = uMelt_folder+'/'+filename
        df_features_uMelt = pd.read_csv(path)
        
        t_uMelt=[]
        for header in df_features_uMelt.loc[:,'uMelt']:
            try:
                float(header)
            except:
                continue
            else:
                t_uMelt.extend([float(header)])
        if len(t_uMelt)>max_t_length:
            max_t_length=len(t_uMelt)
    
    df_uMelt_original=np.empty((len(prefixed),max_t_length))
    df_uMelt_original[:]=np.nan
    index=[]
    
    nFile=0
    for filename in prefixed: 
        path = uMelt_folder+'/'+filename
        df_features_uMelt = pd.read_csv(path)
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
            else:     
                dMC=df_features_uMelt.iloc[data_start:,2].astype('float')
        elif fitting_func=="antiderivate":
            if df_features_uMelt['DNA-UTAH.ORG'].iloc[-1]=='undefined':
                t_uMelt=t_uMelt[:-1]
                dMC=df_features_uMelt.iloc[data_start:-1,1].astype('float')
            else:     
                dMC=df_features_uMelt.iloc[data_start:,1].astype('float')
        
        index.extend([name])
        df_uMelt_original[nFile,:len(dMC)]=dMC
        nFile=nFile+1

    df_uMelt_original=pd.DataFrame(df_uMelt_original)
    df_uMelt_original.index = index

    df_exp_plot=df_features_fitted
    df_original_plot=df_uMelt_original

    # Define the differeht assays that were selected
    assay_names_original=[ind.split('.')[0] for ind in df_uMelt_original.index]
    assay_names_fitted=[ind.split('.')[0] for ind in df_exp_plot.index]
    partial_group_names =np.unique([name.split('_')[0] for name in assay_names_original])

    # Plotting
    # -------------------------
    # Code for plotting for visual verification
    # Color code: blue for exp fitted, black for exp original, red for uMelt, green for NN predicted

    # Plotting variables
    nrows=5
    ncols=2
    nsample,_=df_exp_plot.shape
    fig, axs = plt.subplots(nrows, ncols, figsize=(5, 10))
    current_target = partial_group_names[0] #Used to separate different assays in different subplots

    t_exp = np.arange(60, 100, 0.25)
    t_exp_2 = np.arange(60, 100, 0.1)

    low_res=['mcr-1','mcr-6','mcr-8']
    t_exp_low_res = np.arange(60, 100, 1)
    t_exp_low_res=np.append(t_exp_low_res,100.0)

    for i in range(nsample):
        name = assay_names_original[i].split('_')[0]
        id_original = np.where(np.isin(partial_group_names, name))[0]
        
        name = assay_names_fitted[i].split('_')[0]
        id_fitted = np.where(np.isin(partial_group_names, name))[0]
        if id_original.size != 0:
            row=int(id_fitted//(ncols))
            col=int(id_fitted%(ncols))
            #fitted
            if fitting_func=='gaus3':
                [a1, b1, c1, a2, b2, c2, a3, b3, c3]=df_exp_plot.iloc[i,:-1] 
            elif fitting_func=='gaus1':
                [a1, b1, c1]=df_exp_plot.iloc[i,:-1] 
            elif fitting_func=='antiderivate':
                [a,b,k,t0,L,b1]=df_exp_plot.iloc[i,:6] 
                

            # original 
            if (name in low_res)==False:
                dMC=df_original_plot.iloc[i,:len(t_exp)]
                
                row=int(id_original//(ncols))
                col=int(id_original%(ncols))
                if fitting_func=='gaus3':
                    line1, =axs[row][col].plot(t_exp_2, gaussian_fit(t_exp_2,a1, b1, c1, a2, b2, c2, a3, b3, c3),linewidth=0.1,color='red')
                elif fitting_func=='gaus1':
                    line1, =axs[row][col].plot(t_exp_2, single_gaussian_fit(t_exp_2,a1, b1, c1),linewidth=0.1,color='red')
                elif fitting_func=='antiderivate':
                    line1, =axs[row][col].plot(t_exp_2, sigmoid_fit(np.array(t_exp_2),a,b,k, t0,L,b1),linewidth=0.1,color='red')
            
                line2, =axs[row][col].plot(t_exp, dMC,linewidth=0.1,color='purple')
            else:
                dMC=df_original_plot.iloc[i,:len(t_exp_low_res)]
                
                row=int(id_original//(ncols))
                col=int(id_original%(ncols))
                if fitting_func=='gaus3':
                    line1, =axs[row][col].plot(t_exp_2, gaussian_fit(t_exp_2,a1, b1, c1, a2, b2, c2, a3, b3, c3),linewidth=0.1,color='red')
                elif fitting_func=='gaus1':
                    line1, =axs[row][col].plot(t_exp_2, single_gaussian_fit(t_exp_2,a1, b1, c1),linewidth=0.1,color='red')
                elif fitting_func=='antiderivate':
                    line1, =axs[row][col].plot(t_exp_2, sigmoid_fit(np.array(t_exp_2),a,b,k, t0,L,b1),linewidth=0.1,color='red')
            
                line2, =axs[row][col].plot(t_exp_low_res, dMC[:41],linewidth=0.1,color='purple')
                
            axs[row][col].set_title(name)
            axs[row][col].grid(True)
            axs[row][col].set_xlim([60, 100])
            axs[row][col].set_ylim([0, 160])

    #removing empty subplots
    for ax_row in axs:
        for ax in ax_row:
            if not ax.lines:
                fig.delaxes(ax)

    fig.legend([line1, line2], ['Fitted experimental', 'Original experimental'], loc='lower right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("Original and fitted uMelt metl curves"+f' {fitting_func}')
    plt.show()

def plot_compare_fitted(df_features_fitted,df_features_fitted_uMelt,fitting_func,id_variant_association=pd.DataFrame()):
    """
    Plotting the experimental fitted curves with the uMelt fitted curve
    """
        
    #increase the resolution of the plots
    plt.rcParams['figure.dpi'] = 300
    df_features_fitted=df_features_fitted[['mcr-8' not in row for row in df_features_fitted.index]]
    # Random select 0.1% to plot (the order of the two dataframes are identical, different rows are different samples)
    n, _ = df_features_fitted.shape
    nsample=n//1000

    index_list=pd.DataFrame([range(n)])
    sample_index=index_list.transpose().sample(nsample)

    df_exp_plot=df_features_fitted.iloc[[a for a in sample_index.index],:-1]

    # Define the differeht assays that were selected
    assay_names=[ind.split('.')[0] for ind in df_exp_plot.index]
    partial_group_names =list(dict.fromkeys(assay_names))
    partial_group_names.sort()

    # Code for plotting for visual verification
    # Color code: blue for exp fitted, orange for exp original, red for uMelt fitted, purple for uMelt original, green for NN predicted
    # Plotting variables
    nrows=5
    ncols=2
    fig, axs = plt.subplots(nrows, ncols, figsize=(5, 10))
    t_exp = np.arange(65, 97, 0.5)

    for i in range(nsample):
        name = assay_names[i]
        id = np.where(np.isin(partial_group_names, name))[0]
        if id.size != 0:
            row=int(id//(ncols))
            col=int(id%(ncols))
            #fitted removed
            if fitting_func=="gaus3":
                [a1, b1, c1, a2, b2, c2, a3, b3, c3]=df_exp_plot.iloc[i,:] 
                line1, =axs[row][col].plot(t_exp, gaussian_fit(t_exp,a1, b1, c1, a2, b2, c2, a3, b3, c3),color='blue')
                axs[row][col].set_ylim([0, 0.11])
            elif fitting_func=="gaus1":
                [a1, b1, c1]=df_exp_plot.iloc[i,:] 
                line1, =axs[row][col].plot(t_exp, single_gaussian_fit(t_exp,a1, b1, c1),color='blue')
                axs[row][col].set_ylim([0, 0.11])
            elif fitting_func=="antiderivate":
                [a,b,k,t0,L,b1]=df_exp_plot.iloc[i,:6] 
                axs[row][col].set_ylim([0, 0.7])
                line1, =axs[row][col].plot(t_exp, sigmoid_fit(np.array(t_exp),a,b,k, t0,L,b1),color='blue')
            
            axs[row][col].set_title(name)
            axs[row][col].grid(True)
            axs[row][col].set_xlim([60, 100])
            axs[row][col].set_ylabel('Fitted experimental', color='b')

    #now plot uMelt fitted curves
    df_exp_plot=df_features_fitted_uMelt.iloc[:,:-1]
    assay_names=[ind.split('_')[0] for ind in df_exp_plot.index]
    nsample,_=df_exp_plot.shape

    t_exp_uMelt = np.arange(60, 100, 0.1)
  
    for i in range(nsample):
        name = assay_names[i]
        id = np.where(np.isin(partial_group_names, name))[0]
        if id.size != 0:

            row=int(id//(ncols))
            col=int(id%(ncols))
            #fitted
            
            ax=axs[row][col]
            
            ax2 = ax.twinx()
            ax2.set_ylabel('uMelt', color='r')
            
            if fitting_func=="gaus3":
                [a1, b1, c1, a2, b2, c2, a3, b3, c3]=df_exp_plot.iloc[i,:] 
                line2, =ax2.plot(t_exp_uMelt, gaussian_fit(t_exp_uMelt,a1, b1, c1, a2, b2, c2, a3, b3, c3),linewidth=0.1,color='red')
            elif fitting_func=="gaus1":
                [a1, b1, c1]=df_exp_plot.iloc[i,:] 
                line2, =ax2.plot(t_exp_uMelt, single_gaussian_fit(t_exp_uMelt,a1, b1, c1),linewidth=0.1,color='red')
            elif fitting_func=='antiderivate':
                [a,b,k,t0,L,b1]=df_exp_plot.iloc[i,:6] 
                line2, =ax2.plot(t_exp_uMelt, sigmoid_fit(np.array(t_exp_uMelt),a,b,k, t0,L,b1),linewidth=0.1,color='red')
            
            if id_variant_association.empty==False:
                uMelt_variant=pd.unique(id_variant_association[[name in variant for variant in id_variant_association.loc[:,'variant_name']]].loc[:,'variant_name'])
                for uMelt in uMelt_variant:
                    plot=df_features_fitted_uMelt.loc[uMelt]
                    if fitting_func=="gaus3":
                        [a1, b1, c1, a2, b2, c2, a3, b3, c3]=plot.values[:9]
                        line2, =ax2.plot(t_exp_uMelt, gaussian_fit(t_exp_uMelt,a1, b1, c1, a2, b2, c2, a3, b3, c3),linewidth=0.1,color='green')
                    elif fitting_func=="gaus1":
                        [a1, b1, c1]=plot.values[:3] 
                        line2, =ax2.plot(t_exp_uMelt, single_gaussian_fit(t_exp_uMelt,a1, b1, c1),linewidth=0.1,color='green')
                    elif fitting_func=="antiderivate":
                        [a,b,k,t0,L,b1]=plot.iloc[:6] 
                        line2, =ax2.plot(t_exp_uMelt, sigmoid_fit(np.array(t_exp_uMelt),a,b,k, t0,L,b1),linewidth=0.1,color='green')
            
            ax2.set_ylim([0, 160])

        for name in partial_group_names:
            id = np.where(np.isin(partial_group_names, name))[0]
            row=int(id//(ncols))
            col=int(id%(ncols))

    #removing empty subplots
    for ax_row in axs:
        for ax in ax_row:
            if not ax.lines:
                fig.delaxes(ax)

    fig.legend([line1, line2], ['Fitted experimental', 'Fitted uMelt'], loc='lower right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("Fitted experimental data on 0.1 percent of all samples compared with fitted uMelt curves"+f' {fitting_func}')
    plt.show()

def plot_standard_scaled(Input_train_scaled,Output_train_scaled):
    """
    Plot to visualize the standard scaled values
    """
    assay_names=[ind[:5] if ind[:3] == 'mcr' else ind[:3] for ind in Input_train_scaled.index]
    _,m,_=np.unique(assay_names, return_index=True, return_inverse=True)
    
    df_all=pd.concat([Output_train_scaled,Input_train_scaled.iloc[m,:-2]])
    _,m=df_all.shape

    _,m=Output_train_scaled.shape
    if m==10:
        nrows=3
        ncols=3
    elif m==4:
        nrows=2
        ncols=2
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 10))

    for i in range(m):
        row=int(i//(ncols))
        col=int(i%(ncols))
        exp=df_all.iloc[:-9,i].values
        uMelt=df_all.iloc[-9:,i].values
        
        line1 =axs[row][col].hist(exp,color="blue",alpha=0.5,label='Exp')

        ax=axs[row][col]
        ax.set_ylabel('Experimental', color='b')
        ax2 = ax.twinx()
        ax2.set_ylabel('uMelt', color='r')
        line2 =ax2.hist(uMelt,color="red",alpha=0.5,label='uMelt')

        axs[row][col].set_title(df_all.columns[i])

    plt.tight_layout()
    plt.suptitle("Standard scaled values histogram")
    plt.show()

def plot_MLP_result(Output_test,Output_predicted,method,fitting_func):
    """
    Plot to evaluate the prediction of the random forest
    """
    
    #increase plot resolution
    plt.rcParams['figure.dpi'] = 300

    # Define the differeht assays that were selected
    assay_names=[ind[:5] for ind in Output_test.index]
    partial_group_names =list(dict.fromkeys(assay_names))
    partial_group_names.sort()

    #plotting
    nsample, _ = Output_test.shape
    nrows=4
    ncols=2
    fig, axs = plt.subplots(nrows, ncols, figsize=(5, 10))
    t_exp = np.arange(65, 97, 0.5)

    for i in range(nsample):
        name = assay_names[i]
        id = np.where(np.isin(partial_group_names, name))[0]
        if id.size != 0:
            row=int(id//(ncols))
            col=int(id%(ncols))
            #Output truth
            if fitting_func=='gaus3':
                [a1, b1, c1, a2, b2, c2, a3, b3, c3]=Output_test.iloc[i,:9] 
                line1, =axs[row][col].plot(t_exp, gaussian_fit(t_exp,a1, b1, c1, a2, b2, c2, a3, b3, c3),color='blue')
           
            if fitting_func=='gaus1':
                [a1, b1, c1]=Output_test.iloc[i,:3] 
                line1, =axs[row][col].plot(t_exp, single_gaussian_fit(t_exp,a1, b1, c1),color='blue')
            
            if fitting_func=='antiderivate':
                [a,b,k,t0,L,b1]=Output_test.iloc[i,:6] 
                line1, =axs[row][col].plot(t_exp, sigmoid_fit(np.array(t_exp),a,b,k, t0,L,b1),color='blue')
            
    # Define the differeht assays that were selected
    assay_names=[ind[:5] for ind in Output_predicted.index]
    partial_group_names =list(dict.fromkeys(assay_names))
    partial_group_names.sort()

    nsample, _ = Output_predicted.shape
    for i in range(nsample):
        name = assay_names[i]
        id = np.where(np.isin(partial_group_names, name))[0]
        if id.size != 0:
            row=int(id//(ncols))
            col=int(id%(ncols)) 
            if fitting_func=='gaus3':
                [a1, b1, c1, a2, b2, c2, a3, b3, c3]=Output_predicted.iloc[i,:9] 
                line2, =axs[row][col].plot(t_exp, gaussian_fit(t_exp,a1, b1, c1, a2, b2, c2, a3, b3, c3),color='green')
                axs[row][col].set_ylim([0, 0.11])
            if fitting_func=='gaus1':
                [a1, b1, c1]=Output_predicted.iloc[i,:3] 
                line2, =axs[row][col].plot(t_exp, single_gaussian_fit(t_exp,a1, b1, c1),color='green')
                axs[row][col].set_ylim([0, 0.11])
            if fitting_func=='antiderivate':
                [a,b,k,t0,L,b1]=Output_predicted.iloc[i,:6] 
                line2, =axs[row][col].plot(t_exp, sigmoid_fit(np.array(t_exp),a,b,k, t0,L,b1),color='green')
                axs[row][col].set_ylim([0, 0.7])

            axs[row][col].set_title(name)
            axs[row][col].grid(True)
            axs[row][col].set_xlim([60, 100])
            axs[row][col].set_ylabel('Output', color='b')

    #now plot uMelt fitted curves
    df_features_fitted_uMelt = pd.read_csv(f'fitted_data/df_features_uMelt_variant_fitted_{fitting_func}.csv',index_col=0)
    
    df_exp_plot=df_features_fitted_uMelt[['mcr-8' not in row for row in df_features_fitted_uMelt.index]]
    assay_names=[ind[2:7] for ind in df_exp_plot.index]
    partial_group_names =list(dict.fromkeys(assay_names))
    nsample,_=df_exp_plot.shape

    t_exp_uMelt = np.arange(60, 100, 0.1)

    for i in range(nsample):
        name = assay_names[i]
        id = np.where(np.isin(partial_group_names, name))[0]
        if id.size != 0:
            row=int(id//(ncols))
            col=int(id%(ncols))
            #fitted
            
            ax=axs[row][col]
            
            ax2 = ax.twinx()
            ax2.set_ylabel('Input', color='r')
            if fitting_func=='gaus3':
                [a1, b1, c1, a2, b2, c2, a3, b3, c3]=df_exp_plot.iloc[i,:9] 
                line3, =ax2.plot(t_exp_uMelt, gaussian_fit(t_exp_uMelt,a1, b1, c1, a2, b2, c2, a3, b3, c3),linewidth=0.1,color='red')
           
            if fitting_func=='gaus1':
                [a1, b1, c1]=df_exp_plot.iloc[i,:3] 
                line3, =ax2.plot(t_exp_uMelt, single_gaussian_fit(t_exp_uMelt,a1, b1, c1),linewidth=0.1,color='red')
           
            if fitting_func=='antiderivate':
                [a,b,k,t0,L,b1]=df_exp_plot.iloc[i,:6] 
                line3, =ax2.plot(t_exp_uMelt, sigmoid_fit(np.array(t_exp_uMelt),a,b,k, t0,L,b1),linewidth=0.1,color='red')

            ax2.set_ylim([0, 160])

    fig.legend([line1, line2, line3], ['Fitted experimental','Predicted experimental', 'Fitted uMelt'], loc='lower right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("MLP prediction with uMelt curve, fitted experimental curve and predicted experimental curve"+r'\n'+ f"{method} {fitting_func}")
    plt.show()