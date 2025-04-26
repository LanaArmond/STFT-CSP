'''
stft.py

Description
-----------
This module contains the implementation of the Short-Time Fourier Transform (STFT) for EEG data. 
The STFT function transforms the input signals into the time-frequency domain.

Dependencies
------------
numpy, scipy, modules.core.eegdata, modules.utils.trial_transform

'''

import os, sys, copy
project_directory = os.getcwd()
sys.path.append(project_directory)

import numpy as np
from scipy.signal import stft

from modules.core.eegdata import eegdata
from modules.utils.trial_transform import trial_transform

def STFT(X, sfreq, nperseg=None, noverlap=None, nfft=None, window='hann', return_onesided=True, freqs_per_bands='auto'):
    
    '''
    Computes the Short-Time Fourier Transform (STFT) of the input EEG data.

    Parameters
    ----------
    X : np.ndarray
        Input EEG data. Expected shape is (n_trials, n_bands, n_electrodes, n_samples).
    sfreq : float
        Original sampling frequency of the input data.
    nperseg : int, optional
        Length of each segment. Defaults to the sampling frequency.
    noverlap : int, optional
        Number of points to overlap between segments. Defaults to nperseg-1.
    nfft : int, optional
        Number of points in the FFT. Defaults to nperseg.
    window : str, optional
        Desired window to use. Default is 'hann'.
    return_onesided : bool, optional
        If True, return a one-sided spectrum for real data. Default is True.
    freqs_per_bands : str or int, optional
        Number of frequencies to average per band. Default is 'auto'.

    Returns
    -------
    np.ndarray
        Transformed data in the time-frequency domain.
    '''

    many_trials = len(X.shape) == 4
    if not many_trials:
        X = X[np.newaxis, :, :, :]

    if X.shape[1] != 1:
        raise ValueError('The input data must have only one band.')
    
    if nperseg is None:
        nperseg = sfreq

    if nfft is None:
        nfft = nperseg
    
    if noverlap is None:
        noverlap = nperseg-1
        
    if freqs_per_bands == 'auto':
        freqs_per_bands = 4
    X_ = []
    for trial_ in range(X.shape[0]):
        X_.append([])
        for electrode_ in range(X.shape[2]):
            frequencies, time_points, Zxx = stft(X[trial_, 0, electrode_], fs=sfreq, nperseg=nperseg, noverlap=noverlap, nfft=nfft, window=window, return_onesided=return_onesided)
            Zxx = np.abs(Zxx).astype(float)
            if freqs_per_bands == 'auto':
                Zxx = [Zxx[2*i] + Zxx[2*i+1]/2 for i in range(len(Zxx)//2)]
            else:
                Zxx = [np.mean(Zxx[i:i+freqs_per_bands], axis=0) for i in range(0, len(Zxx)//2, freqs_per_bands)]
            X_[-1].append(Zxx)
    X_ = np.array(X_)
    X_ = np.transpose(X_, (0, 2, 1, 3))[:, :, :, 1:]

    if not many_trials:
        X_ = X_[0]

    return X_

class STFT_eegdata(trial_transform):
    
    '''
    STFT_eegdata class to apply STFT on EEG data using the trial_transform base class.

    Methods
    -------
    func(trial, **kwargs):
        Applies STFT to the given trial.
    '''

    def func(self, trial, **kwargs):
        
        '''
        Applies STFT to the given trial.

        Parameters
        ----------
        trial : np.ndarray
            The EEG data trial to transform.
        **kwargs : dict
            Additional arguments to pass to the STFT function.

        Returns
        -------
        np.ndarray
            Transformed data in the time-frequency domain.
        '''
        
        return STFT(trial, **kwargs)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Create a sample data
    X = np.random.rand(10, 1, 3, 2*128)

    # Filter the data
    X_ = STFT(X, sfreq=128)

    # Plot the data
    for trial_ in range(10):
        for band_ in range(5):
            for electrode_ in range(3):
                plt.plot(X[trial_, 0, electrode_], 'b')
                plt.plot(X_[trial_, band_, electrode_], 'r')
                plt.show()

    print('Done')
