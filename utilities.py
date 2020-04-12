import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



def var_to_cat(dataframe, kind, ref, col=4, size=(13,6), **kwargs):
    
    df_col = dataframe.select_dtypes(kind)
    col_idx = list(df_col.columns[df_col.columns != ref])
    lenght = len(col_idx)
    row =len(list(range(0, lenght, col)))
    
    fig, axs = plt.subplots(row, 
                            col, 
                            figsize=size, 
                            constrained_layout=True)
    
    for i, ax in enumerate(fig.axes): 
        
        if i < lenght:
            if kind == 'category':
                sns.countplot(data=dataframe, x=col_idx[i] , hue=ref, ax=ax, **kwargs)    
            
            else:
                sns.boxplot(data=dataframe, x=ref, y=col_idx[i], ax=ax, width=.5, **kwargs)
        
        else:
            ax.remove()

        

        
def hist(dataframe, 
         col = 4, 
         ref = None, 
         size=(13,10), 
         edgecolor='k', 
         bins=15,
         alpha=.5,
         **kwargs):
    
    col_idx = list(dataframe.columns[dataframe.columns != ref])
    lenght = len(col_idx)
    row = len(list(range(0, lenght, col)))
    label = dataframe[ref].unique()
   
    fig, axs = plt.subplots(row, 
                            col, 
                            figsize=size, 
                            constrained_layout=True)
    
    for i, ax in enumerate(fig.axes):
        
        if i < lenght:
            dataframe.groupby(ref)[col_idx[i]].hist(histtype='stepfilled',
                                                    density=True,  
                                                    ax=ax,
                                                    alpha=alpha,
                                                    bins=bins,
                                                    edgecolor='k',
                                                    **kwargs)
                                                     
            ax.set_xlabel(col_idx[i])
            ax.legend(label, title=ref, loc='best')
            
        else:
            ax.remove()
    
    

    
def outliers_IQR(x):
    
    q1 = x.quantile(.25)
    q3 = x.quantile(.75)
    IQR = q3 - q1
    q1 = q1 - 1.5 * IQR
    q3 = q3 + 1.5 * IQR
    
    return x.clip(q1, q3, axis=0)




def corr_matrix(dataframe, **kwargs):
    
    plt.figure(figsize=(9,7))
    sns.heatmap(dataframe.corr(), 
                annot=True, 
                cmap='coolwarm', 
                mask=np.triu(dataframe.corr(), k=1), 
                square=True,
                fmt='2.1f',
                **kwargs)
    
    plt.show()
