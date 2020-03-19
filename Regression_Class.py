#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
plt.style.use('seaborn')


# In[2]:


class Regressor:
    
    def __init__(self,
                 label,
                 l_rate = .5,
                 stop = 1e-3,
                 reg_rate = 0.,
                 beta = .5,
                 epochs = 1000,
                 logistic=True):
        
        self.label = label
        self.l_rate = l_rate
        self.stop = stop
        self.reg_rate = reg_rate
        self.beta = beta
        self.epochs = epochs
        self.logistic = logistic
        
#---------------------------------------------------------------------------------------------------------------------------       
                   
    def matrix(self, dataset):
        
        X = dataset.drop(self.label, axis=1).to_numpy()
        X = np.insert(X, 0, 1, axis=1)
        y = dataset[[self.label]].to_numpy()

        return X, y   
    
#----------------------------------------------------------------------------------------------------------------------------
 
    def split(self, dataset, cutoff = .8):
        
        n = len(dataset)
        
        df_train, df_test = dataset.iloc[0:int(n*cutoff)], dataset.iloc[int(n*cutoff):]
        
        X_train, y_train = Regressor.matrix(self, df_train)
        X_test, y_test = Regressor.matrix(self, df_test)
          
        return  X_train, y_train, X_test, y_test
    
#---------------------------------------------------------------------------------------------------------------------------    
    
    def fit(self, X, y, graph = False):
        
        n = len(X)
        
        np.random.seed(3479)
        weights = np.random.rand(1, X.shape[1])
    
        Gamma = np.full((1, X.shape[1]), self.reg_rate/n)
        Gamma[0, 0] = 0
    
        W_list = [weights]
        for i in range(self.epochs):
            H = (1 / (1 + np.exp(-X @ weights.T))) if self.logistic else (X @ weights.T)
 
            dJ = (1/n) * ((H-y).T @ X)
            update = weights - (self.l_rate * dJ)

            soft_thresholding = lambda W, K: np.sign(W) * np.maximum(abs(W)-K, 0)
            shrinkage = 1 / (1 + 2*self.l_rate*(1-self.beta)*Gamma)
            weights_ = shrinkage * soft_thresholding(update, self.l_rate*self.beta*Gamma) 
            
            W_list.append(weights_)
        
            delta = weights - weights_
            weights = weights_
        
            if (abs(delta) <= self.stop).all():
                break
        else:
            print('max epochs reached')
    
        W_list = np.array(W_list) 
        
        setattr(Regressor, 'weights', weights)
  
        if graph:
            plt.figure(figsize=(15, 10))
            for i in range(W_list.shape[2]):
                plt.plot(W_list[...,i], label=f'W_{i}')
            plt.legend(ncol=3, frameon=True, loc='upper right')
            plt.title('PARAMETERS\' EVOLUTION', fontsize=15)
            plt.xlabel('epochs', fontsize=13)
            plt.ylabel('parameters\' values', fontsize=13)
            plt.show()

        return self
    
#---------------------------------------------------------------------------------------------------------------------------    
    
    def predict(self, X):
        
        output = 1 / (1 + np.exp(-X @ self.weights.T))
        setattr(Regressor, 'prediction', output)
        
        return self
    
#---------------------------------------------------------------------------------------------------------------------------    
    
    def fit_predict(self, X_train, y_train, X_test, graph=False):
        
        weights = Regressor.fit(self, X_train, y_train, graph)
        setattr(Regressor, 'weights', weights)
        
        output = 1 / (1 + np.exp(-X_test @ self.weights.T))
        setattr(Regressor, 'prediction', output)
        
        return output, weights
    
#---------------------------------------------------------------------------------------------------------------------------    
    
    def metrics(self, test, threshold = 0.5, verbose = True):
        
        classification = np.array((self.prediction >= threshold), dtype=np.int)
    
        TP = len(classification[(classification == 1) & (classification == test)])
        FP = len(classification[(classification == 1) & (classification != test)])
        TN = len(classification[(classification == 0) & (classification == test)])
        FN = len(classification[(classification == 0) & (classification != test)])

        epsilon = 1e-10 
        a = (TP+TN) / (TP+TN+FP+FN)
        p = TP / (TP+FP+epsilon)
        r = TP / (TP+FN+epsilon)
        F1 = (2*p*r) / (p+r+epsilon) 

        confusion_matrix = np.array([[TP, FP], [TN, FN]])
        setattr(Regressor, 'confusion_matrix', confusion_matrix)
    
        if verbose:
            print(f'Accuracy: {a}\nPrecision: {p}\nRecall: {r}\nF1 score: {F1}\n\nConfusion Matrix:\n {confusion_matrix}')
    
        return confusion_matrix
    
#---------------------------------------------------------------------------------------------------------------------------

    def ROC(self, test):
        
        results = (Regressor.metrics(self, test, i, False).reshape(4) for i in np.arange(0.0, 1.1, 0.001))
        TP, FP, TN, FN = zip(*results)
    
        epsilon = 1e-7
        TPR = np.array(TP) / (np.array(TP) + np.array(FN) + epsilon)
        FPR = np.array(FP) / (np.array(FP) + np.array(TN) + epsilon)
    
        AUC = np.around(np.trapz(TPR, FPR), 2)
    
        plt.figure(figsize=(8, 8))
        plt.plot(FPR, TPR, color='b')
        plt.plot([0,1], [0,1], 'r--')
        plt.text(0.7, 0.2, f'AUC = {abs(AUC)}', fontsize=16, color='k')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC', fontsize=15)
        plt.show()
        
#---------------------------------------------------------------------------------------------------------------------------       
        
    def cross_val(self, dataset, folds = 10, threshold = 0.5,):
        
        dataset = dataset.sample(frac=1)
        
        slices= np.arange(0, len(dataset), len(dataset)//folds)
        results = np.zeros((2, 2), dtype=np.int)
        for i in range(folds):
            df_train = dataset.drop(range(slices[i], slices[i+1]))
            df_test = dataset.iloc[slices[i]:slices[i+1]]
        
            X_train, Y_train = Regressor.matrix(self, df_train)
            X_test, Y_test = Regressor.matrix(self, df_test)
        
            output = Regressor.fit(self, X_train, Y_train).predict(X_test)
            results += Regressor.metrics(self, Y_test, threshold, False)
        
        (TP, FP), (TN, FN) = results
    
        epsilon = 1e-7
        a = (TP+TN) / (TP+TN+FP+FN)
        p = TP / (TP+FP+epsilon)
        r = TP / (TP+FN+epsilon)
        F1 = (2*p*r) / (p+r)
    
        confusion_matrix = np.array([[TP, FP], [TN, FN]])
    
        print(f'Accuracy: {a}\nPrecision: {p}\nRecall: {r}\nF1 score: {F1}\n\nConfusion Matrix:\n {confusion_matrix}')
           



