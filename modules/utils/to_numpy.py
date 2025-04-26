'''
to_numpy.py

Description
-----------
This module contains the implementation of the to_numpy transformer.

Dependencies
------------
None

'''

import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)

class to_numpy():
    ''' to_numpy transformer

    Description
    -----------
    This class implements the to_numpy transformer.

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

    def fit(self, data):
        ''' Fits the transformer to the data.

        Description
        -----------
        This method does nothing.

        Parameters
        ----------
        data : eegdata
            The input data.

        returns
        -------
        self
            The transformer.

        '''

        return self
    
    def transform(self, data):
        ''' Transforms the input data into a numpy dictionary.
        
        Description
        -----------
        This method transforms the input data into a numpy dictionary.
        
        Parameters
        ----------
        data : eegdata
            The input data.
        
        returns
        -------
        dict
            The transformed data.
            
        '''

        return {'X': data.data, 
                'y': data.labels}
    
    def fit_transform(self, data):
        ''' Fits the transformer to the data and transforms it.
        
        Description
        -----------
        This method does nothing.
        
        Parameters
        ----------
        data : eegdata
            The input data.
        
        returns
        -------
        dict
            The transformed data.
        
        '''
        
        return self.transform(data)