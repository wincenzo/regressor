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
                              **kwargs)
            
            else:
                sns.boxplot(data=dataframe, 
                            x=ref, 
                            y=col_idx[i], 
                            ax=ax, 
                            width=.5,
                            saturation=.5,
                            **kwargs)
                
                mean = df_col[col_idx[i]].mean()
                ax.axhline(y=mean, c='r', ls='--', label='mean')
                ax.legend(loc='best', framealpha=.3)
        
        else:
            ax.remove()

        

        
#def hist(dataframe, 
#         col = 4, 
#         ref = None, 
#         xsize = 11,
#         **kwargs):
#    
#    col_idx = dataframe.columns[dataframe.columns != ref]
#    lenght = len(col_idx)
#    row = len(list(range(0, lenght, col)))
#    label = dataframe[ref].unique()
#    
#    (_, g0), (_, g1) = dataframe.groupby(ref)
#   
#    fig, axs = plt.subplots(row, 
#                            col, 
#                            figsize=(xsize,((xsize/col)*.75)*row), 
#                            constrained_layout=True)
#    
#    for i, ax in enumerate(axs.flat):
#        
#        if i < lenght:
#            if dataframe[col_idx[i]].dtype.name == 'category':
#                kde = False
#                norm_hist=False
#            else:
#                kde = True
#                norm_hist = True
#                
#            sns.distplot(g0[col_idx[i]],
#                         kde=kde,
#                         hist_kws={'histtype':'step', 'fill':True}, 
#                         kde_kws={'shade':False}, 
#                         norm_hist=norm_hist, 
#                         ax=ax, 
#                         axlabel=False, 
#                         **kwargs)
#            sns.distplot(g1[col_idx[i]],
#                         kde=kde,
#                         hist_kws={'histtype':'step', 'fill':True}, 
#                         kde_kws={'shade':False}, 
#                         norm_hist=norm_hist, 
#                         ax=ax, 
#                         axlabel=False, 
#                         **kwargs)
#            ax.set_title(col_idx[i])
#            ax.legend(label, title=ref, loc='best')
#            
#        else:
#            ax.remove()


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
                        alpha=.55,
                        density=True,
                        color='gray',
                        **kwargs)
                sns.kdeplot(g0[col_idx[i]], ax=ax)
                sns.kdeplot(g1[col_idx[i]], ax=ax)
                
            else:
                ax.hist(dataframe[col_idx[i]], 
                        histtype='stepfilled', 
                        alpha=.6, 
                        **kwargs)
                ax.hist(g0[col_idx[i]], 
                        histtype='stepfilled', 
                        alpha=.8, 
                        **kwargs)
            
            ax.set_title(col_idx[i])
            ax.legend(label, 
                      title=ref, 
                      loc='best', 
                      framealpha=.2)
            
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
