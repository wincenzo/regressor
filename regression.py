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
                 logistic = True):
        
        
        self.l_rate = l_rate
        self.stop = stop
        self.reg_rate = reg_rate
        self.beta = beta
        self.epochs = epochs
        self.logistic = logistic
        self._weights_ = None
        
        
        
               
    def _X(self, data):
        
        X = data.drop(self.target, axis=1).to_numpy()
        X = np.insert(X, 0, 1, axis=1)
        
        return X
        
                
        

    def _y(self, data):
    
        y = data[[self.target]].to_numpy() # with double squares-brakets return a column vector
        
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
        weights = np.random.rand(1, X_train.shape[1]) if self._weights_ is None or reset else self._weights_
        #weights = np.ones((1, X_train.shape[1])) if self._weights_ is None or reset else self._weights_
    
        gamma = np.full((1, X_train.shape[1]), r/n)
        gamma[0, 0] = 0
        self.gamma = gamma
    
        w_list = [weights]
        
        for e in range(self.epochs):
            
            H = X_train @ weights.T
            #for numeric stability
            _sigm, sigm_ = 1/(1+np.exp(-H)), np.exp(H)/(1+np.exp(H))
            H = np.where(H>=0, _sigm, sigm_) if log else H
            
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
        
        self._weights_ = weights

        return self
    
        
 
    
    def predict(self, df_test):
        
        X_test = self._X(df_test)
        self.y_test = self._y(df_test)
        
        H = X_test @ self._weights_.T
        _sigm, sigm_ = 1/(1+np.exp(-H)), np.exp(H)/(1+np.exp(H))
        
        self.prediction = np.where(H>=0, _sigm, sigm_ ) if self.logistic else H
        
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
                                   'weights': self._weights_.ravel()})
        
        col = lambda x: 'color:red' if x == 0 else "color:''"
        parameters = parameters.style.applymap(col, subset=['weights'])
        
        return parameters
    
        
    
    
    def graph(self, size = (11,11)):
       
        w_list_T = np.transpose(self.w_list, axes=[0,2,1])
            
        H = self.X @ w_list_T
        H = 1 / (1 + np.exp(-H)) if self.logistic else H
            
        J = (-1/self.n) * (self.y.T @ np.log(H) + (1-self.y).T @ np.log(1-H))
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
        
        plt.show()
        
    
    
    
    def _metrics(self, threshold, full = True):
            
        classes = np.array((self.prediction >= threshold), dtype=np.int)
    
        TP = classes[(classes == 1) & (classes == self.y_test)].size
        FP = classes[(classes == 1) & (classes != self.y_test)].size
        TN = classes[(classes == 0) & (classes == self.y_test)].size
        FN = classes[(classes == 0) & (classes != self.y_test)].size
        
        if full:
            eps = 1e-7
            self.a = (TP+TN) / (TP+TN+FP+FN)
            self.p = TP / (TP+FP+eps)
            self.r = TP / (TP+FN+eps)
            self.F1 = (2*self.p*self.r) / (self.p+self.r+eps) 
            
        return np.array([[TP, FP], [FN, TN]])
    
    
    

    def metrics(self, threshold):
        
        if (not hasattr(self, '_conf_matr')) or (threshold != self.threshold):
            self._conf_matr = self._metrics(threshold)
            self.threshold = threshold
            
        print(f'Accuracy: {round(self.a, 3)}')
        print(f'Precision: {round(self.p, 3)}')
        print(f'Recall: {round(self.r, 3)}')
        print(f'F1 score: {round(self.F1, 3)}', end='\n\n')
        
        self.confusion_matrix
        
        
         
    @property
    def confusion_matrix(self):
        plt.figure(figsize=(3,3))
        sns.heatmap(self._conf_matr, 
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
        
        results = (self._metrics(i, False).ravel() for i in np.arange(0.0, 1.1, 0.01))
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
        
        

        
    def cross_val(self, 
                  dataset, 
                  folds = 10, 
                  threshold = 0.5):
        
        reg_CV = Regressor(
            l_rate=self.l_rate,
            stop=self.stop,
            reg_rate=self.reg_rate,
            beta=self.beta,
            epochs=self.epochs,
            logistic=self.logistic)
        
        
        assert 1 < folds <= len(dataset), "folds must be greater than 1 and less than or equal to dataset's length"
        
        slices = range(0, len(dataset)+1, len(dataset)//folds)
        
        df_train = (dataset.drop(index=range(slices[i], slices[i+1])) for i in range(folds))
        df_test = (dataset.iloc[slices[i]:slices[i+1]] for i in range(folds))
        
        
        if hasattr(self, 'scaler'):
            df_train = (reg_CV.fitnscale(df, self.scaler, self.num) for df in df_train)                     
            df_test = (reg_CV.scale(df) for df in df_test)
        
        
        reg_CV._conf_matr = np.sum([
            reg_CV.fitnpredict(a, b, self.target)._metrics(self.threshold, False) for a, b in zip(df_train, df_test)], 
            axis=0)
        
        (TP, FP), (FN, TN) = reg_CV._conf_matr
    
        eps = 1e-7
        reg_CV.a = (TP+TN) / (TP+TN+FP+FN)
        reg_CV.p = TP / (TP+FP+eps)
        reg_CV.r = TP / (TP+FN+eps)
        reg_CV.F1 = (2*reg_CV.p*reg_CV.r) / (reg_CV.p+reg_CV.r)
        
        reg_CV.threshold = self.threshold
        reg_CV.metrics(self.threshold)

        
        
        
        
        
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
        
        assert kind != None, 'Please choose a method: "robust" - "standard" - "minmax"'
        
        self.num = slice(None) if columns is None else columns
        
        self.scaler = kind
        
        if self.scaler == 'robust':
            self.median = data[self.num].agg('median')
            IQR = lambda x: x.quantile(.75) - x.quantile(.25) 
            self.IQR = data[self.num].apply(IQR)
            
        if self.scaler == 'standard':
            self.mean = data[self.num].agg('mean')
            self.std = data[self.num].agg('std')
            
        if self.scaler =='minmax':
            self.min = data[self.num].agg('min')
            self.max = data[self.num].agg('max')
            
        return self
    
    
            
            
    def scale(self, data):
        
        df_scaled = data.copy()
            
        if self.scaler == 'robust':
            df_scaled.loc[:,self.num] = (df_scaled.loc[:,self.num]-self.median) / self.IQR

            return df_scaled
                
        if self.scaler == 'standard':
            df_scaled.loc[:,self.num] = (df_scaled.loc[:,self.num]-self.mean) / self.std
       
            return df_scaled 
                
        if self.scaler == 'minmax':
            df_scaled.loc[:,self.num] = (df_scaled.loc[:,self.num]-self.min) / (self.max-self.min)

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
        
        assert kind != None, 'Please choose a method: "robust" - "standard"'
        
        self.num = slice(None) if columns is None else columns
        
        self.outliers = kind
        
        self.mean = data[self.num].mean()
        
        if kind == 'robust':
            self.IQR = data[self.num].quantile(.75) - data[self.num].quantile(.25)
            self.low = self.mean - 1.5 * self.IQR
            self.up = self.mean + 1.5 * self.IQR
            
            data.loc[:,self.num] = data.loc[:,self.num].clip(lower=self.low, upper=self.up, axis=1)
            
            return data
    
        if kind == 'standard':
            self.std = data[self.num].std()
            s = 3 * self.std
            self.low = self.mean - s
            self.up = self.mean + s
            
            data.loc[:,self.num] = data.loc[:,self.num].clip(lower=self.low, upper=self.up, axis=1)
            
            return data
    
        
        
        
        
    
        