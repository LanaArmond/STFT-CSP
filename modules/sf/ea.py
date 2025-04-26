'''
euclideanalignment.py

Description
-----------
This module contains the implementation of the Euclidean Alignment method for Spatial filter.

Dependencies
------------
numpy
scipy

'''

import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)

from scipy.linalg import fractional_matrix_power
import numpy as np
from modules.utils.data_transform import data_transform

class ea:
    ''' Euclidean Alignment method for Spatial filter
    
    Description
    -----------
    This class implements the Euclidean Alignment method for Spatial filter.
    It transform the input data from the target domain, aligning the data in a way that the reference matrix becomes an identity matrix

    It's important to notice that the original implementation of the Euclidean Alignment only has one reference
    matrix for each subject. However, this implementantion uses a reference matrix for each bandpass of the subject
    
    Attributes
    ----------
    target_transformation: list-like, size (n_bands)
        List containing the reference matrix for each band of the target subject


    Methods
    -------
    __init__(random_state):
        Initializes the class.
    calc_r(data)
        Return the list of reference matrix from the data
    full_r(self, data):
        Return the list of reference matrix to be used in the transformation of the data
    fit(self, data = None):
        Create and stores the reference matrix
    transform(self, data):
        Align target subject data
    fit_transform(self, data):
        Calculates the reference matrices of the target subject,
        and then aligns all received input data
    '''
    def __init__(self):   
        self.target_transformation = None

    def calc_r(self, data):
        ''' Return the list of reference matrix from the data
        
        Description
        -----------
        This method calculates the reference matrix from the input data. Because the package organize the 
        input data as (n_trials, n_bands, n_electodes, n_times), each band has its own reference matrix
        
        Parameters
        ----------
        data : array-like, shape (n_trials, n_bands, n_electodes, n_times)
            The input data from a subject.
            
        Returns
        -------
        list_r : list-like, size (n_bands), containing array-like, shape (n_electodes, n_electodes)
            The list of reference matrix from the data.
        
        '''
        list_r = []
        for band in range(data.shape[1]):
            r = np.zeros((data.shape[2], data.shape[2]))
            for trial in range(data.shape[0]):
                product = np.dot(data[trial][band], data[trial][band].T)
                r += product
            r = r / data.shape[0]
            list_r.append(r)
        return np.array(list_r)
    
    def full_r(self, data):
        ''' Return the list of reference matrix to be used in the transformation of the data
        
        Description
        -----------
        This method call calc_r, and then raises all matrices to the power of -1/2,
        to transform the input data
        
        Parameters
        ----------
        data : array-like, shape (n_trials, n_bands, n_electodes, n_times)
            The input data from a subject.
            
        Returns
        -------
        list_r_inv : list-like, size (n_bands), containing array-like, shape (n_electodes, n_electodes)
            The list of reference matrix to the power of -1/2 from the data.
        
        '''
        list_r = self.calc_r(data)
        list_r_inv = [fractional_matrix_power(r, -0.5) for r in list_r]
        return np.array(list_r_inv)

    def fit(self, data = None):
        ''' Create and stores the reference matrix
        
        Description
        -----------
        This method call full_r, and then store it inside the class
        
        Parameters
        ----------
        data : array-like, shape (n_trials, n_bands, n_electodes, n_times)
            The input data from a subject.
            
        Returns
        -------
        np.ndarray
            The spatial filters.
        
        '''
        self.target_transformation = self.full_r(data)
        return self

    def transform(self, data):
        ''' Align target subject data
        
        Description
        -----------
        This method aligns the target subject's data by multiplying it
        by the reference matrix for each band.
        
        Parameters
        ----------
        target_data : array-like, shape (n_trials, n_bands, n_electodes, n_times)
            The input data from the target subject before the transformation.
            
        Returns
        -------
        target_data : array-like, shape (n_trials, n_bands, n_electodes, n_times)
            The input data from the target subject after the transformation.
        '''
        X = data.copy()

        for band in range(X.shape[1]):
            for trial in range(X.shape[0]):
                X[trial][band] = np.dot(self.target_transformation[band], X[trial][band])

        return X

    def fit_transform(self, data):
         ''' Calculates the reference matrices of the target subject,
        and then aligns all received input data
        
        Description
        -----------
        This method receives the input data from the target subject, and then calculates the 
        reference matrix of all it. After that, the alignment is performed, multiplying each 
        data with its respective matrix of reference. This method returns the input data from the target subject after the transformation.
        
        Parameters
        ----------
        target_data : array-like, shape (n_trials, n_bands, n_electodes, n_times)
            The input data from the target subject before the transformation.
            
        Returns
        -------
        target_data : array-like, shape (n_trials, n_bands, n_electodes, n_times)
            The input data from the target subject after the transformation.
        
        '''
         return self.fit(data).transform(data)

class ea_eegdata(data_transform):
     
	def __init__(self):
          
		self.ea_ = ea()		

	def func_transform(self, data, **kwargs):
		return self.ea_.transform(data, **kwargs)

	def func_fit_transform(self, target, source, **kwargs):
		return self.ea.fit_transform(target, source, **kwargs)
