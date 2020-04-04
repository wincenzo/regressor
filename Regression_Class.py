import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
        self.start = None
        
    
                   
    def matrix(self, dataset):
        
        X = dataset.drop(self.target, axis=1).to_numpy()
        X = np.insert(X, 0, 1, axis=1)
        y = dataset[[self.target]].to_numpy()

        return X, y   
    

 
    def split(self, dataset, cutoff = .8):
        
        n = len(dataset)
        
        df_train, df_test = dataset.iloc[0:int(n*cutoff)], dataset.iloc[int(n*cutoff):]
        
        X_train, y_train = self.matrix(df_train)
        X_test, y_test = self.matrix(df_test)
          
        return  X_train, y_train, X_test, y_test
    
   
    
    def fit(self, X, y, graph = False, reset = False):
        
        n = len(X)
        
        np.random.seed(3479)
        weights = np.random.rand(1, X.shape[1]) if self.start is None or reset else self.start
    
        Gamma = np.full((1, X.shape[1]), self.reg_rate/n)
        Gamma[0, 0] = 0
    
        W_list = [weights]
        for i in range(self.epochs):
            H = X @ weights.T
            H = 1 / (1 + np.exp(-H)) if self.logistic else H
 
            dJ = (1/n) * ((H-y).T @ X)
            update = weights - (self.l_rate * dJ)

            soft_thresholding = lambda W, K: np.sign(W) * np.maximum(np.abs(W)-K, 0)
            shrinkage = 1 / (1 + self.l_rate*(1-self.beta)*Gamma)
            _weights = shrinkage * soft_thresholding(update, self.l_rate*self.beta*Gamma)
            
            W_list.append(_weights)
        
            delta = weights - _weights
            weights = _weights
        
            if (abs(delta) <= self.stop).all():
                break
        else:
            print('max epochs reached')
            
        self.weights = weights
        self.start = weights
    
        W_list = np.array(W_list)
  
        if graph:
            plt.figure(figsize=(15, 10))
            for i in range(W_list.shape[2]):
                plt.plot(W_list[...,i], label=f'W_{i}')
            plt.legend(ncol=3, frameon=True, loc='upper right')
            plt.title(f'Weights\' evolution;  $\gamma$ = {self.reg_rate}', fontsize=15)
            plt.xlabel('epochs', fontsize=13)
            plt.ylabel('parameters\' values', fontsize=13)
            #plt.savefig('C:\\Users\\wince\\Desktop\\param.png')
            plt.show()

        return self
    
 
    
    def predict(self, test):
    
        self.prediction =  1 / (1 + np.exp(-(test @ self.weights.T)))
        
        return self
    

    
    def fit_predict(self, X_train, y_train, X_test):
        
        weights = self.fit(X_train, y_train, graph=False)
        prediction = 1 / (1 + np.exp(-(X_test @ self.weights.T)))
        
        return  weights, output
    
    
    
    def _metrics(self, test, threshold = 0.5):
            
        classification = np.array((self.prediction >= threshold), dtype=np.int)
    
        TP = len(classification[(classification == 1) & (classification == test)])
        FP = len(classification[(classification == 1) & (classification != test)])
        TN = len(classification[(classification == 0) & (classification == test)])
        FN = len(classification[(classification == 0) & (classification != test)])

        epsilon = 1e-7
        self.accuracy = (TP+TN) / (TP+TN+FP+FN)
        self.precision = TP / (TP+FP+epsilon)
        self.recall = TP / (TP+FN+epsilon)
        self.F1 = (2*self.precision*self.recall) / (self.precision+self.recall+epsilon) 

        self.confusion_matrix = np.array([[TP, FP], [TN, FN]])
            
        return self.confusion_matrix
    

 
    def metrics(self, test, threshold = 0.5):
        
        self._metrics(test, threshold)
        
        print(f'Accuracy: {self.accuracy}\nPrecision: {self.precision}\nRecall: {self.recall}\nF1 score: {self.F1}', end='\n\n')
        print(f'Confusion Matrix:\n {self.confusion_matrix}')
            


    def ROC(self, test):
        
        results = (self._metrics(test, i).reshape(4) for i in np.arange(0.0, 1.1, 0.01))
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
        
        slices = np.arange(0, len(dataset)+1, len(dataset)//folds)
        df_train = (dataset.drop(index=range(slices[i], slices[i+1])) for i in range(folds))
        df_test = (dataset.iloc[slices[i]:slices[i+1]] for i in range(folds))
        
        train = map(reg_CV.matrix, df_train)
        test = map(reg_CV.matrix, df_test)

        results = sum([reg_CV.fit(a, b).predict(c)._metrics(d, threshold) for (a, b), (c, d) in zip(train, test)])
        (TP, FP), (TN, FN) = results
    
        epsilon = 1e-7
        a = (TP+TN) / (TP+TN+FP+FN)
        p = TP / (TP+FP+epsilon)
        r = TP / (TP+FN+epsilon)
        F1 = (2*p*r) / (p+r)
    
        confusion_matrix = np.array([[TP, FP], [TN, FN]])
    
        print(f'Accuracy: {a}\nPrecision: {p}\nRecall: {r}\nF1 score: {F1}\n\nConfusion Matrix:\n {confusion_matrix}')
