'''
medianclf.py

Description
-----------
This module contains the implementation of the Median Single Electrode Energy classifier.

Dependencies
------------
numpy
pandas
scipy.signal

'''
import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)

import numpy as np

class medianclf():
    def __init__(self):
        pass

    def fit(self, X, y):
        if X.shape[1] != 1:
            raise ValueError("Shape tem que ter apenas uma caracteristica")
        if np.unique(y) != [0,1]:
            raise ValueError("Y tem que ser 0 e 1")
        X_ = X.copy().reshape(-1)
        self.median = np.median(X_)
        y_predict = np.array([0 if X_[i] < self.median else 1 for i in range(len(X_))])
        self.signal =  np.sum((y == y_predict).astype(int))/len(y)
        if self.signal < 0.5:
            self.signal = -1
        else:
            self.signal = 1           

    def predict(self, X):
        X_ = X.copy().reshape(-1)
        y_predict = np.array([False if X_[i] < self.median else True for i in range(len(X_))])
        if self.signal == -1:
            y_predict = not y_predict
        y_predict = y_predict.astype(int)
        return y_predict
    
    def fit_predict(self, X, y):
       self.fit(X, y)
       return self.predict(X)