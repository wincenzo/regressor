import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



class Regressor:
    
    def __init__(self,
                 l_rate = .5,
                 stop = 1e-3,
                 reg_rate = 0.,
                 beta = .5,
                 epochs = 1000,
                 logistic = None):
        
        self.l_rate = l_rate
        self.stop = stop
        self.reg_rate = reg_rate
        self.beta = beta
        self.epochs = epochs
        self.logistic = logistic
        self.weights = None
        
        
        
               
    def _X(self, data):
        
        X = data.drop(self.target, axis=1).to_numpy()
        X = np.insert(X, 0, 1, axis=1)
        
        return X
        
                
        

    def _y(self, data):
    
        y = data[[self.target]].to_numpy() #with double squares-brakets return a column vector
        
        return y
    
    
    
     
    def fit(self, 
            df_train, 
            target = None, 
            reset = False):
        
        assert target is not None, 'Please choose a target column'
        
        self.target = target

        self.features = list(df_train.columns)
        self.features.remove(target)    
        
        X_train = self._X(df_train)
        y_train = self._y(df_train)
        self.X, self.y = X_train, y_train
        
        self.n = len(X_train)
        
        n = self.n
        l = self.l_rate
        r = self.reg_rate
        b = self.beta
        s = self.stop
        log = self.logistic
        
        np.random.seed(3)
        weights = np.random.rand(1, X_train.shape[1]) if self.weights is None or reset else self.weights
        #weights = np.ones((1, X_train.shape[1])) if self._weights_ is None or reset else self._weights_
    
        gamma = np.full((1, X_train.shape[1]), r/n)
        gamma[0, 0] = 0
        self.gamma = gamma
    
        w_list = [weights]
        
        for e in range(self.epochs):
            
            H = X_train @ weights.T
            
            if log is not None:
                H = np.where(H>=0, 1/(1+np.exp(-H)), np.exp(H)/(1+np.exp(H))) if log else H
                
            else:
                if df_train[self.target].nunique() == 2:
                    H = np.where(H>=0, 1/(1+np.exp(-H)), np.exp(H)/(1+np.exp(H))) 
                    self.logistic = True
                    
                else:
                    self.logistic = False
                    
            #gradient descend
            dJ = (1/n) * ((H-y_train).T @ X_train)
            _weights = weights - (l*dJ)
            
            #elastic-net ISTA
            _weights = np.sign(_weights) * np.maximum(np.abs(_weights)-(l*b*gamma), 0)
            shrinkage = 1 / (1 + l*(1-b)*gamma)
            _weights *= shrinkage 
        
            delta = weights - _weights
            weights = _weights
            
            w_list.append(weights)
        
            if (abs(delta) <= s).all():
                break
        else:
            print('max epochs reached')
            
            
        self.w_list = np.array(w_list) 
        self.weights = weights

        return self
    
        
 
    
    def predict(self, df_test):
        
        X_test = self._X(df_test)
        self.y_test = self._y(df_test)
        
        H = X_test @ self.weights.T
    
        self.prediction = np.where(H>=0, 1/(1+np.exp(-H)), np.exp(H)/(1+np.exp(H))) if self.logistic else H
                
        return self
    
    

    
    def fitnpredict(self, 
                    df_train, 
                    df_test, 
                    target = None, 
                    reset = False):
        
        assert target is not None, 'Please choose a target column'
            
        self.fit(df_train, target, reset).predict(df_test)
        
        return self
    
    
     
    @property
    def parameters(self):
        
        features = self.features.copy()  
        features.insert(0, 'bias')
        parameters = pd.DataFrame({'features': features, 
                                   'weights': self.weights.ravel()})
        
        col = lambda x: 'color:red' if x == 0 else "color:''"
        parameters = parameters.style.applymap(col, subset=['weights'])
        
        return parameters
    
        
    
    
    def graph(self, size = (11,11)):
       
        w_list_T = np.transpose(self.w_list, axes=[0,2,1])
            
        H = self.X @ w_list_T
        H = np.where(H>=0, 1/(1+np.exp(-H)), np.exp(H)/(1+np.exp(H))) if self.logistic else H
                
        if self.logistic:
            J = (-1/self.n) * (self.y.T @ np.log(H) + (1-self.y).T @ np.log(1-H))
            
        else:
            k = (self.y-H)
            k_T = np.transpose(k, axes=[0,2,1])
            J = (1/(2*self.n)) * (k_T @ k)
            
        l2 = ((1-self.beta)/2) * (self.w_list @ w_list_T)
        l1 = self.beta * np.linalg.norm(self.w_list, ord=1, axis=(1,2))[:,np.newaxis,np.newaxis]
        J = (J+(self.reg_rate/self.n)*(l1+l2)).ravel()
        
        w = self.w_list
        
        
        plt.figure(figsize=size, tight_layout=True)
            
        ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=2)
        
        plt.plot(J, 'r')
        plt.ylabel('error', fontsize=13)
        plt.text(0.85, 0.9,
                 f'$Err = {round(J[-1], 2)}$', 
                 fontsize=14,
                 bbox={'facecolor':'white', 
                       'edgecolor':'gray', 
                       'boxstyle':'round', 
                       'alpha':1},
                 transform=ax1.transAxes )
    
        ax2 = plt.subplot2grid((5, 1), (2, 0), rowspan=3)
        
        for i in range(w.shape[2]):
            plt.plot(w[...,i], label='$w_{%d}$' % i)    
        plt.legend(frameon=True, 
                   bbox_to_anchor=(1.0, 1.0), 
                   framealpha=.4,
                   fontsize=11,
                   ncol=2)
        plt.text(0.85, 0.92, 
                 f'$\gamma$ = {self.reg_rate}', 
                 fontsize=14,
                 bbox={'facecolor':'white', 
                       'edgecolor':'gray', 
                       'boxstyle':'round', 
                       'alpha':1},
                 transform=ax2.transAxes)
        plt.xlabel('epochs', fontsize=13)
        plt.ylabel('weights', fontsize=13)
        sns.despine()
        
        plt.show()
        
    
    
    
    def _metrics(self, threshold):
        
        if self.logistic:
            classes = np.array((self.prediction >= threshold), dtype=np.int)
            self.classes = classes
        
            TP = classes[(classes == 1) & (classes == self.y_test)].size
            FP = classes[(classes == 1) & (classes != self.y_test)].size
            TN = classes[(classes == 0) & (classes == self.y_test)].size
            FN = classes[(classes == 0) & (classes != self.y_test)].size
            
            return np.array([TP, FP, FN, TN])
        
        else:
            RSS = (self.y_test-self.prediction).T @ (self.y_test-self.prediction)
            TSS = (self.y_test-self.y_test.mean()).T @ (self.y_test-self.y_test.mean())
            
            self.R_2 = np.round(1 - (RSS/TSS), 3) 
            
            MSE = (1/self.n) * ((self.y_test-self.prediction).T @ (self.y_test-self.prediction))

            self.RMSE = np.round(np.sqrt(MSE), 3)
            
            return np.array([self.R_2, self.RMSE])
    
    
    

    def metrics(self, threshold = None):
        
        if self.logistic:
            assert threshold is not None,\
            'Please assign a valid threshold value'
            
            if (not hasattr(self, '_conf_matr')) or (threshold != self.threshold):
                self._conf_matr = self._metrics(threshold)
            
            self.threshold = threshold
            
            TP, FP, FN, TN = self._conf_matr
            
            eps = 1e-7
            self.a = (TP+TN) / (TP+TN+FP+FN)
            self.p = TP / (TP+FP+eps)
            self.r = TP / (TP+FN+eps)
            self.F1 = (2*self.p*self.r) / (self.p+self.r+eps)
                
            print(f'Accuracy: {round(self.a, 3)}')
            print(f'Precision: {round(self.p, 3)}')
            print(f'Recall: {round(self.r, 3)}')
            print(f'F1 score: {round(self.F1, 3)}', end='\n\n')
            
            self.confusion_matrix
            
        else:
            if not hasattr(self, 'scores'):
                self.R_2, self.RMSE = self._metrics(threshold)
            
            print(f'R^2 = {self.R_2.item()}\n\nRMSE = {self.RMSE.item()}')
                            
        
        
         
    @property
    def confusion_matrix(self):
        
        matrix = self._conf_matr.reshape((2, 2))
        
        plt.figure(figsize=(3,3))
        sns.heatmap(matrix, 
                    annot=True, 
                    fmt='d',  
                    center=0, 
                    annot_kws={'fontsize':14},
                    square=True, 
                    xticklabels=['P','N'],
                    yticklabels=['P','N'])
        
        plt.yticks(rotation=0)
        plt.title('Confusion Matrix', fontsize=14)
        plt.show()
        
                

            
    @property
    def ROC(self):
        
        results = (self._metrics(i) for i in np.arange(0.0, 1.1, 0.01))
        TP, FP, FN, TN = zip(*results)
    
        eps = 1e-7
        TPR = np.array(TP) / (np.array(TP)+np.array(FN)+eps)
        FPR = np.array(FP) / (np.array(FP)+np.array(TN)+eps)
    
        AUC = np.around(np.trapz(TPR, FPR), 2)
    
        plt.figure(figsize=(6, 6))
        plt.plot(FPR, TPR, color='b')
        plt.plot([0,1], [0,1], 'r--')
        plt.text(0.7, 0.2, 
                 f'AUC = {abs(AUC)}', 
                 fontsize=16, 
                 color='k',
                 bbox={'facecolor':'white', 
                       'edgecolor':'gray', 
                       'boxstyle':'round', 
                       'alpha':1})
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC', fontsize=15)
        
        plt.show()
        
        
        
        

        
        
        
        
class CrossValidation:
    
    def __init__(self, model = None, scaler = None):
        
        from copy import copy
        
        self.Model = copy(model)
        self.Scaler = copy(scaler)

        
             
        
    def metrics(self, dataset, folds = 10, threshold = None):
        
        assert 1 < folds <= len(dataset),\
        "folds must be greater than 1 and less than or equal to dataset's length"
        
        Scaler = self.Scaler
        Model = self.Model
        #Model.weights = None
    
        slices = range(0, len(dataset)+1, len(dataset)//folds)
        
        df_train = (dataset.drop(index=range(slices[i], slices[i+1])) for i in range(folds))
        df_test = (dataset.iloc[slices[i]:slices[i+1]] for i in range(folds))
        
        if hasattr(Scaler, 'scaler'): 
            df_train = (Scaler.fitnscale(df, Scaler.scaler, Scaler.num) for df in df_train)                     
            df_test = (Scaler.scale(df) for df in df_test)
            
        if Model.logistic:
            self.threshold = Model.threshold if threshold is None else threshold
            
            Model._conf_matr = [Model.fitnpredict(a, b, Model.target)._metrics(self.threshold) 
                                for a, b in zip(df_train, df_test)]
            Model._conf_matr = np.sum(Model._conf_matr, axis=0)    
        
            Model.metrics(self.threshold)
        
        else:
            self.threshold = None
            
            _ = np.array([(Model.fitnpredict(a, b, Model.target).prediction.ravel(), 
                           Model.y_test.ravel())
                           for a, b in zip(df_train, df_test)])
    
            Model.prediction = _[:,0,0]
            Model.y_test = _[:,1,0]

            Model.metrics(self.threshold)
        
        

        
    
    
               
        
class PreProcessing:
    
    
    def split_train_test(self, 
                         data, 
                         cutoff = 0.8, 
                         seed = None):
        
        n = len(data)
        cut = int(n*cutoff)
        
        data = data.sample(frac=1, random_state=seed)
        
        df_train, df_test = data.iloc[0:cut], data.iloc[cut:]
        
        return df_train, df_test
    
    
    
              
    def fit_scaler(self, 
                   data, 
                   kind = None, 
                   columns = None):
        
        assert kind in ("robust", "standard", "minmax"),\
        'Please choose one of these method: "robust" - "standard" - "minmax"'
        
        self.num = slice(None) if columns is None else columns
        
        self.scaler = kind
        
        if self.scaler == 'robust':
            self.median = data[self.num].agg('median')
            IQR = lambda x: x.quantile(.75) - x.quantile(.25) 
            self.IQR = data[self.num].apply(IQR)
            
        elif self.scaler == 'standard':
            self.mean = data[self.num].agg('mean')
            self.std = data[self.num].agg('std')
            
        elif self.scaler =='minmax':
            self.min = data[self.num].agg('min')
            self.max = data[self.num].agg('max')
            
        return self
    
    
            
            
    def scale(self, data):
        
        df_scaled = data.copy()
            
        if self.scaler == 'robust':
            df_scaled.loc[:,self.num] = (df_scaled[self.num]-self.median) / self.IQR
         
        elif self.scaler == 'standard':
            df_scaled.loc[:,self.num] = (df_scaled[self.num]-self.mean) / self.std
                
        elif self.scaler == 'minmax':
            df_scaled.loc[:,self.num] = (df_scaled[self.num]-self.min) / (self.max-self.min)

        return df_scaled
        
        
    
    
    def fitnscale(self, 
                  data, 
                  kind = None, 
                  columns = None):
        
        df_scaled = self.fit_scaler(data, kind, columns).scale(data)
        
        return df_scaled
    
             
        
        
    def clip_outliers(self, 
                      data, 
                      kind = None, 
                      columns = None):
        
        assert kind in ("robust", "standard"),\
        'Please choose one of these method: "robust" - "standard"'
        
        self.num = slice(None) if columns is None else columns
        
        self.outliers = kind
        
        self.mean = data[self.num].mean()
        
        if kind == 'robust':
            self.IQR = data[self.num].quantile(.75) - data[self.num].quantile(.25)
            self.low = self.mean - 1.5 * self.IQR
            self.up = self.mean + 1.5 * self.IQR
            
            data.loc[:,self.num] = data[self.num].clip(lower=self.low, upper=self.up, axis=1)  
    
        elif kind == 'standard':
            self.std = data[self.num].std()
            s = 3 * self.std
            self.low = self.mean - s
            self.up = self.mean + s
            
            data.loc[:,self.num] = data[self.num].clip(lower=self.low, upper=self.up, axis=1)
            
        return data
    
        
        
        
        
    
        
