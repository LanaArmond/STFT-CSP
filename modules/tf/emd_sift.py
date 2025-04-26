'''
emd_sift.py

Description
-----------
This module contains the implementation of the Empirical Mode Decomposition (EMD) 
applied to EEG data. The EMD function decomposes the input signals into Intrinsic Mode Functions (IMFs).

Dependencies
------------
numpy, emd, os, sys, copy, matplotlib

'''

import numpy as np
import emd

import os, sys, copy
project_directory = os.getcwd()
sys.path.append(project_directory)

from modules.utils.trial_transform import trial_transform

def EMD(X, n_imfs=5):
    
    '''
    Empirical Mode Decomposition (EMD) function to decompose signals into Intrinsic Mode Functions (IMFs).

    Parameters
    ----------
    X : np.ndarray
        Input EEG data. Expected shape is (n_trials, 1, n_electrodes, n_samples).
    n_imfs : int, optional
        Number of IMFs to extract, by default 5.

    Returns
    -------
    np.ndarray
        Decomposed IMFs with shape (n_trials, n_imfs, n_electrodes, n_samples).

    Raises
    ------
    ValueError
        If the input data does not have exactly one band (shape[1] != 1).
    '''
        
    many_trials = len(X.shape) == 4
    if not many_trials:
        X = X[np.newaxis, :, :, :]

    # verify if the data has only one band
    if X.shape[1] != 1:
        raise ValueError('The input data must have only one band.')

    imfs = []

    for trial_ in range(X.shape[0]):
        imfs.append([])
        for electrode_ in range(X.shape[2]):
            try:
                imfs_ = emd.sift.sift(X[trial_][0][electrode_], max_imfs=None).T
            except:
                imfs_ = np.random.rand(n_imfs, X.shape[3]) * 1e-6

            if len(imfs_) > n_imfs:
                imfs_[n_imfs-1] = np.sum(imfs_[n_imfs-1:], axis=0)
                imfs_ = imfs_[:n_imfs]
            elif len(imfs_) < n_imfs:
                imfs_ = np.concatenate((imfs_, np.zeros((n_imfs-len(imfs_), X.shape[3]))), axis=0)
            
            imfs[-1].append(imfs_)

    imfs = np.array(imfs)
    imfs = np.transpose(imfs, (0, 2, 1, 3))
    
    if not many_trials:
        imfs = imfs[0]

    return imfs


class EMD_eegdata(trial_transform):
    
    '''
    EMD_eegdata class to apply EMD on EEG data using the trial_transform base class.

    Methods
    -------
    func(trial, **kwargs):
        Applies EMD to the given trial.
    '''
    
    def func(self, trial, **kwargs):
        
        '''
        Applies EMD to the given trial.

        Parameters
        ----------
        trial : np.ndarray
            The EEG data trial to decompose.
        **kwargs : dict
            Additional arguments to pass to the EMD function.

        Returns
        -------
        np.ndarray
            The decomposed IMFs of the trial.
        '''

        return EMD(trial, **kwargs)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Create a sample data
    X = np.random.rand(10, 1, 3, 2*128)

    # Filter the data
    X_ = EMD(X)

    # Plot the data
    for trial_ in range(10):
        for band_ in range(5):
            for electrode_ in range(3):
                plt.plot(X[trial_, 0, electrode_], 'b')
                plt.plot(X_[trial_, band_, electrode_], 'r')
                plt.show()

    print('Done')
