'''
logpower.py

Description
-----------
This module contains the implementation of the logpower feature extractor.

Dependencies
------------
eegdata on modules/core
typing
numpy

'''

import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)

import numpy as np
from typing import Union, List, Optional

from modules.core.eegdata import eegdata

class logpower():
    ''' Logpower feature extractor
    
    Description
    -----------
    This class implements the logpower feature extractor.
    
    Attributes
    ----------
    None
    
    Methods
    -------
    transform(data, flating=False):
        Transforms the input data into the logpower feature space.
        
    '''

    def __init__(self, flating: Optional[bool] = False):
        ''' Initializes the class.
        
        Description
        -----------
        This method initializes the class. It does not receive any parameters and does not
        return anything.
        
        Parameters
        ----------
        flating : bool, optional
        
        Returns
        -------
        None
        
        '''
        
        if type(flating) != bool:
            raise ValueError ("Has to be a boolean type value")
        else:
            self.flating = flating

    def fit(self, data: Union[eegdata, dict]) -> object:
        ''' That method does nothing.
        
        Description
        -----------
        This method does nothing.
        
        Parameters
        ----------
        data : eegdata
            The input data.
            
        Returns
        -------
        None
        
        '''
        if type(data) != eegdata and type(data) != dict:
            raise ValueError ("Has to be a eegdata or dict type")         
        return self

    def transform(self, data: Union[eegdata, dict]) -> dict:
        ''' Transforms the input data into the logpower feature space.
        
        Description
        -----------
        This method transforms the input data into the logpower feature space. It returns a
        dictionary with the transformed data.
        
        Parameters
        ----------
        data : eegdata or dict
            The input data.
            
        Returns
        -------
        output : dict
            The transformed data.
            
        '''
        if type(data) != eegdata and type(data) != dict:
            raise ValueError ("Has to be a eegdata or dict type") 
        if type(data) == eegdata:
            X = data.data
            y = data.labels
        else:                
            X = data['X']
            y = data['y']

        many_trials = len(X.shape) == 4
        if not many_trials:
            X = X[np.newaxis, :, :, :]

        output = []
        trials_, bands_, channels_, _ = X.shape

        for trial_ in range(trials_):
            output.append([])
            for band_ in range(bands_):
                output[trial_].append([])
                for channel_ in range(channels_):
                    if X[trial_, band_, channel_, :].std() == 0:
                        output[trial_][band_].append(0)
                    else:
                        output[trial_][band_].append(np.log(np.mean(X[trial_, band_, channel_, :]**2)))

        output = np.array(output)
        
        if self.flating:
            output = output.reshape(output.shape[0], -1)

        if not many_trials:
            output = output[0]

        return {'X': output, 'y': y}
    
    def fit_transform(self, data: Union[eegdata, dict]) -> dict:
        ''' Fits the model to the input data and transforms it into the logpower feature space.
        
        Description
        -----------
        This method fits the model to the input data and transforms it into the logpower feature
        space. It returns a dictionary with the transformed data.
        
        Parameters
        ----------
        data : eegdata or dict
            The input data.
            
        Returns
        -------
        output : dict
            The transformed data.
            
        '''
        
        return self.fit(data).transform(data)