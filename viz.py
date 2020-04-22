import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



def var_vs_target(dataframe, 
                  kind, 
                  ref = None, 
                  col = 4, 
                  xsize = 12, 
                  **kwargs):
    
    df_col = dataframe.select_dtypes(kind)
    col_idx = df_col.columns[df_col.columns != ref]
    lenght = len(col_idx)
    row =len(list(range(0, lenght, col)))
    
    fig, axs = plt.subplots(row, 
                            col, 
                            figsize=(xsize,((xsize/col)*.75)*row), 
                            constrained_layout=True)
    
    for i, ax in enumerate(axs.flat):
        
        if i < lenght:
            if kind == 'category':
                sns.countplot(data=dataframe, 
                              x=col_idx[i] , 
                              hue=ref, 
                              ax=ax,
                              alpha=.9,
                              **kwargs)
            
            else:
                sns.boxplot(data=dataframe, 
                            x=ref, 
                            y=col_idx[i], 
                            ax=ax, 
                            width=.4,
                            saturation=.8,
                            **kwargs)
                
                mean = df_col[col_idx[i]].mean()
                ax.axhline(y=mean, c='r', ls='--', label='mean')
                ax.legend(loc='best', framealpha=.3)
        
        else:
            ax.remove()




def hist(dataframe, 
         col = 4, 
         ref = None, 
         xsize = 12,
         **kwargs):
    
    col_idx = dataframe.columns[dataframe.columns != ref]
    lenght = len(col_idx)
    row = len(list(range(0, lenght, col)))
    label = dataframe[ref].unique()
    
    g0 = dataframe[dataframe[ref] == 0]
    g1 = dataframe[dataframe[ref] == 1]
    
    fig, axs = plt.subplots(row, 
                            col, 
                            figsize=(xsize,((xsize/col)*.75)*row), 
                            constrained_layout=True)

    for i, ax in enumerate(axs.flat):
        
        if i < lenght:
            if dataframe.dtypes[i].name != 'category':
                ax.hist(dataframe[col_idx[i]], 
                        histtype='stepfilled', 
                        alpha=.6,
                        density=True,
                        color='gray',
                        **kwargs)
                
                sns.kdeplot(g0[col_idx[i]], ax=ax)
                sns.kdeplot(g1[col_idx[i]], ax=ax)
                
            else:
                ax.hist(dataframe[col_idx[i]], 
                        histtype='bar', 
                        alpha=.8)
                
                ax.hist(g0[col_idx[i]], 
                        histtype='bar', 
                        alpha=.8)
            
            ax.set_title(col_idx[i])
            ax.legend(label, 
                      title=ref, 
                      loc='best', 
                      framealpha=.3)
            
        else:
            ax.remove()




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

    
    
