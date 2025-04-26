'''
data_transform.py

Description
-----------
This module contains the implementation of the data_transform class.

Dependencies
------------
eeegdata on modules/core
abc

'''

from abc import abstractmethod

class data_transform():
    ''' data_transform class
    
    Description
    -----------
    This class implements the data_transform class.
    
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

        return data.apply_to_data(self.func_transform, **kwargs)
    
    @abstractmethod
    def func_transform(self, data, **kwargs):
        ''' The transformation function.
        
        Description
        -----------
        This method applies the transformation to a single data.
        
        Warning
        -------
        This method must be implemented by the child class.
        
        Parameters
        ----------
        data : np.ndarray
            The input data.
        **kwargs
            The keyword arguments.
        
        returns
        -------
        np.ndarray
            The transformed data.
            
        '''
        pass

    def fit_transform(self, data, **kwargs):
        return data.apply_to_data(self.func_fit_transform, **kwargs)
    
    @abstractmethod
    def func_fit_transform(self, data, **kwargs):
        pass