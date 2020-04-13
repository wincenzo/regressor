import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



class Regressor:
    
    def __init__(self,
                 target = None,
                 l_rate = .5,
                 stop = 1e-3,
                 reg_rate = 0.,
                 beta = .5,
                 epochs = 1000,
                 logistic = True):
        
        self.target = target
        self.l_rate = l_rate
        self.stop = stop
        self.reg_rate = reg_rate
        self.beta = beta
        self.epochs = epochs
        self.logistic = logistic
        self._weights_ = None
        
        
    
                   
    def matrix(self, dataset):
        
        X = dataset.drop(self.target, axis=1).to_numpy()
        X = np.insert(X, 0, 1, axis=1)
        y = dataset[[self.target]].to_numpy() #double square brakets return a column vector 

        return X, y   
    
    

 
    def split(self, dataset, cutoff = .8, seed=None):
        
        n = len(dataset)
        cut = int(n*cutoff)
        
        dataset = dataset.sample(frac=1, random_state=seed)
        
        df_train, df_test = dataset.iloc[0:cut], dataset.iloc[cut:]
        
        X_train, y_train = self.matrix(df_train)
        X_test, y_test = self.matrix(df_test)
        
        self.features = list(dataset.columns)
          
        return  X_train, y_train, X_test, y_test
    
    
   
    
    def fit(self, X_train, y_train, reset = False):
        
        self.n = len(X_train)
        self.X, self.y = X_train, y_train
        
        n = self.n
        l = self.l_rate
        r = self.reg_rate
        b = self.beta
        s = self.stop
        log = self.logistic
        
        weights = np.ones((1, X_train.shape[1])) if self._weights_ is None or reset else self._weights_
    
        gamma = np.full((1, X_train.shape[1]), r/n)
        gamma[0, 0] = 0
        self.gamma = gamma
    
        w_list = [weights]
        
        for e in range(self.epochs):
            
            H = X_train @ weights.T
            H = 1 / (1 + np.exp(-H)) if log else H
            
            dJ = (1/n) * ((H-y_train).T @ X_train)
            _weights = weights - (l*dJ)

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
            
        w_list = np.array(w_list)
        
        self.w_list = w_list
        self._weights_ = weights

        return self
    
    
    
    
    @property
    def weights(self):
        
        self.features.remove(self.target)
        self.features.insert(0, 'bias')
        weights = pd.DataFrame({'features': self.features, 'weights': self._weights_.ravel()})
        
        return weights
    
        
    
    
    def graph(self, size = (11,10)):
       
        w_T = np.transpose(self.w_list, axes=[0,2,1])
            
        H = self.X @ w_T
        H = 1 / (1 + np.exp(-H)) if self.logistic else H
            
        J = (-1/self.n) * (self.y.T @ np.log(H) + (1-self.y).T @ np.log(1-H))
        l2 = ((1-self.beta)/2) * (self.w_list @ w_T)
        l1 = self.beta * np.linalg.norm(self.w_list, ord=1, axis=(1,2))[:,np.newaxis,np.newaxis]
        J = (J+(self.reg_rate/self.n)*(l1+l2)).ravel()
        
        plt.figure(figsize=size, tight_layout=True)
            
        plt.subplot2grid((5, 1), (0, 0), rowspan=2)
        plt.plot(J, 'r')
        plt.xlabel('epochs', fontsize=13)
        plt.ylabel('error', fontsize=13)

            
        w = self.w_list
        
        plt.subplot2grid((5, 1), (2, 0), rowspan=3)
        for i in range(w.shape[2]):
            plt.plot(w[...,i], label=f'W_{i}')
            
        plt.legend(frameon=True, bbox_to_anchor=(1.1, 1.0), framealpha=.6)
        plt.title(f'$\gamma$ = {self.reg_rate}', fontsize=14)
        plt.xlabel('epochs', fontsize=13)
        plt.ylabel('weights', fontsize=13)
        
        plt.show()
        
        
 
    
    def predict(self, X_test):
    
        self.prediction =  1 / (1 + np.exp(-(X_test @ self._weights_.T)))
        
        return self
    
    

    
    def fit_predict(self, X_train, y_train, X_test):
        
        weights = self.fit(X_train, y_train)
        prediction = 1 / (1 + np.exp(-(X_test @ self._weights_.T)))
        
        return  weights, output
    
    
    
    
    def _metrics(self, y_test, threshold = 0.5):
            
        classification = np.array((self.prediction >= threshold), dtype=np.int)
    
        TP = classification[(classification == 1) & (classification == y_test)].size
        FP = classification[(classification == 1) & (classification != y_test)].size
        TN = classification[(classification == 0) & (classification == y_test)].size
        FN = classification[(classification == 0) & (classification != y_test)].size

        epsilon = 1e-7
        self.accuracy = (TP+TN) / (TP+TN+FP+FN)
        self.precision = TP / (TP+FP+epsilon)
        self.recall = TP / (TP+FN+epsilon)
        self.F1 = (2*self.precision*self.recall) / (self.precision+self.recall+epsilon) 
            
        return np.array([[TP, FP], [FN, TN]])
    
    

 
    def metrics(self, y_test, threshold = 0.5):
        
        self.confusion_matrix = self._metrics(y_test, threshold)
        
        print(f'Accuracy: {self.accuracy}\nPrecision: {self.precision}\nRecall: {self.recall}\nF1 score: {self.F1}', end='\n\n')
        
        plt.figure(figsize=(3,3))
        sns.heatmap(self.confusion_matrix, 
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
        
            


    def ROC(self, test):
        
        results = (self._metrics(test, i).ravel() for i in np.arange(0.0, 1.1, 0.01))
        TP, FP, FN, TN = zip(*results)
    
        epsilon = 1e-7
        TPR = np.array(TP) / (np.array(TP) + np.array(FN) + epsilon)
        FPR = np.array(FP) / (np.array(FP) + np.array(TN) + epsilon)
    
        AUC = np.around(np.trapz(TPR, FPR), 2)
    
        plt.figure(figsize=(6, 6))
        plt.plot(FPR, TPR, color='b')
        plt.plot([0,1], [0,1], 'r--')
        plt.text(0.7, 0.2, f'AUC = {abs(AUC)}', fontsize=16, color='k')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC', fontsize=15)
        plt.show()
        
        

        
    def cross_val(self, dataset, folds = 10, threshold = 0.5):
        
        reg_CV = Regressor(
            target=self.target,
            l_rate=self.l_rate,
            stop=self.stop,
            reg_rate=self.reg_rate,
            beta=self.beta,
            epochs=self.epochs,
            logistic=self.logistic)
        
        assert 1 < folds <= len(dataset), 'folds must be greater than 1 and less than or equal to dataset length'
        
        slices = range(0, len(dataset)+1, len(dataset)//folds)
        df_train = (dataset.drop(index=range(slices[i], slices[i+1])) for i in range(folds))
        df_test = (dataset.iloc[slices[i]:slices[i+1]] for i in range(folds))
        
        train = map(reg_CV.matrix, df_train)
        test = map(reg_CV.matrix, df_test)

        conf_matr = np.sum([reg_CV.fit(a, b).predict(c)._metrics(d, threshold) for (a, b), (c, d) in zip(train, test)], 
                           axis=0)
        
        (TP, FP), (FN, TN) = conf_matr
    
        epsilon = 1e-7
        a = (TP+TN) / (TP+TN+FP+FN)
        p = TP / (TP+FP+epsilon)
        r = TP / (TP+FN+epsilon)
        F1 = (2*p*r) / (p+r)
    
        print(f'Accuracy: {a}\nPrecision: {p}\nRecall: {r}\nF1 score: {F1}', end='\n\n')
        
        plt.figure(figsize=(3,3))
        sns.heatmap(conf_matr, 
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
              