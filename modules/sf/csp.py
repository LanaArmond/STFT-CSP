'''
csp.py

Description
-----------
This module contains the implementation of the Common Spatial Patterns filter.

Dependencies
------------
eegdata on modules/core
numpy
scipy

'''

import numpy as np
import scipy as sp
from modules.core.eegdata import eegdata
from modules.utils.data_transform import data_transform

class csp:
    ''' Common Spatial Patterns filter

    Description
    -----------
    This class implements the Common Spatial Patterns filter.

    Attributes
    ----------
    n_electrodes : int
        The number of electrodes.
    m_pairs : int
        The number of pairs of features.
    W : np.ndarray
        The spatial filters.

    Methods
    -------
    fit(X, y):
        Fits the filter to the data.
    transform(X):
        Transforms the input data into the selected feature space.
    fit_transform(X, y):
        Fits the filter to the data and transforms the input data into the selected feature space.

    '''
    def __init__(self, m_pairs: int = 2):
        ''' Initializes the class.

        Parameters
        ----------
        m_pairs : int
            The number of pairs of features.

        returns
        -------
        None

        '''

        self.m_pairs = m_pairs

    def fit(self, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
        ''' Fits the filter to the data.
        
        Parameters
        ----------
        data : eegdata or dict
            The input data.
        
        returns
        -------
        np.ndarray
            The spatial filters.
            
        '''
    
        X = data.copy()
        y = labels.copy()

        if len(X.shape) == 3:
            X = np.expand_dims(X, axis=1)

        self.bands = X.shape[1]
        self.n_electrodes = X.shape[2]

        self.W = np.zeros((self.bands, self.n_electrodes, self.n_electrodes))

        # unique values of y
        y_unique = np.unique(y)
        if len(y_unique) != 2:
            raise ValueError("y must have exactly two unique classes.")

        sigma = np.zeros((len(y_unique), self.bands, self.n_electrodes, self.n_electrodes))

        for i in range(len(y)):
            for band_ in range(self.bands):
                if y[i] == y_unique[0]:
                    sigma[0, band_] += (X[i, band_] @ X[i, band_].T)
                else:
                    sigma[1, band_] += (X[i, band_] @ X[i, band_].T)

        for band_ in range(self.bands):
            sigma[0, band_] /= np.sum(y == y_unique[0])
            sigma[1, band_] /= np.sum(y == y_unique[1])

        sigma_tot = np.zeros((self.bands, self.n_electrodes, self.n_electrodes))
        for band_ in range(self.bands):
            sigma_tot[band_] = sigma[0, band_] + sigma[1, band_]

        # generalized eigenvalues problem
        W = np.zeros((self.bands, self.n_electrodes, self.n_electrodes))
        for band_ in range(self.bands):
            try:
                _, W[band_] = sp.linalg.eigh(sigma[0, band_], sigma_tot[band_])
            except:
                 # Caso ocorra um erro (ex.: matriz singular), usa a identidade
                W[band_] = np.eye(self.n_electrodes)
                

        W = np.array(W)
        first_aux = W[:, :, :self.m_pairs]
        last_aux = W[:, :, -self.m_pairs:]

        self.W = np.concatenate((first_aux, last_aux), axis=2)

        return self

    def transform(self, data: dict) -> dict:
        ''' Transforms the input data into the selected feature space.
        
        Parameters
        ----------
        data : eegdata or dict
            The input data.
            
        returns
        -------
        np.ndarray
            The transformed data.
            
        '''

        X = data.copy()

        if len(X.shape) == 3:
            X = np.expand_dims(X, axis=1)

        if X.shape[1] != self.bands:
            raise ValueError("The number of bands in the input data is different from the number of bands in the fitted data.")
        
        X = [np.transpose(self.W[band_]) @ X[:, band_] for band_ in range(self.bands)]
        X = np.swapaxes(np.array(X), 0, 1)
        return X
    
        #X_ = [np.transpose(self.W) @ X_[i] for i in range(len(X_))]

    def fit_transform(self, data: dict, labels) -> dict:
        ''' Fits the filter to the data and transforms the input data into the selected feature space.

        Parameters
        ----------
        data : eegdata or dict
            The input data.
        
        returns
        -------
        np.ndarray
            The transformed data.
        
        '''

        return self.fit(data, labels).transform(data)


class csp_eegdata(data_transform):

    def __init__(self, m_pairs: int = 2):

        self.csp_ = csp(m_pairs=m_pairs)
        
    def func_transform(self, data, **kwargs):
        return self.csp_.transform(data, **kwargs)

    def func_fit_transform(self, data, **kwargs):
        return self.csp_.fit_transform(data, **kwargs)