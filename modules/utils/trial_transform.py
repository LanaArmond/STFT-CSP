'''
trial_transform.py

Description
-----------
This module contains the implementation of the trial_transform class.

Dependencies
------------
eeegdata on modules/core
abc

'''

from abc import abstractmethod

class trial_transform():
    ''' trial_transform class
    
    Description
    -----------
    This class implements the trial_transform class.
    
    Attributes
    ----------
    None
    
    '''
    
    def __init__(self):
        ''' Initializes the class.

        Parameters
        ----------
        None

        returns
        -------
        None

        '''

        pass

    def transform(self, data, **kwargs):
        ''' Transforms the data.
        
        Description
        -----------
        This method applies the transformation to the data.
        
        Parameters
        ----------
        data : eegdata
            The input data.
        **kwargs
            The keyword arguments.
        
        returns
        -------
        eegdata
            The transformed data.
            
        '''

        return data.apply_to_trial(self.func, **kwargs)
    
    @abstractmethod
    def func(self, trial, **kwargs):
        ''' The transformation function.
        
        Description
        -----------
        This method applies the transformation to a single trial.
        
        Warning
        -------
        This method must be implemented by the child class.
        
        Parameters
        ----------
        trial : np.ndarray
            The input data.
        **kwargs
            The keyword arguments.
        
        returns
        -------
        np.ndarray
            The transformed data.
            
        '''
        pass