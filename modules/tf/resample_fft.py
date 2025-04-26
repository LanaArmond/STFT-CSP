'''
resample_fft.py

Description
-----------
This module contains the implementation of FFT-based resampling for EEG data. 
The fft_resample function uses the Fast Fourier Transform (FFT) to resample the input signals to a new sampling frequency.

Dependencies
------------
numpy, scipy, modules.core.eegdata, modules.utils.trial_transform

'''

import numpy as np
from scipy.signal import resample

from modules.core.eegdata import eegdata
from modules.utils.trial_transform import trial_transform

def fft_resample(X, sfreq, new_sfreq):
    
    '''
    Resamples the input EEG data to a new sampling frequency using FFT.

    Parameters
    ----------
    X : np.ndarray
        Input EEG data. Expected shape is (n_trials, n_bands, n_electrodes, n_samples).
    sfreq : float
        Original sampling frequency of the input data.
    new_sfreq : float
        New sampling frequency to resample the data.

    Returns
    -------
    dict
        A dictionary containing the resampled data and the new sampling frequency.
        {'data': np.ndarray, 'sfreq': float}

    '''
    
    many_trials = len(X.shape) == 4
    if not many_trials:
        X = X[np.newaxis, :, :, :]

    duration = X.shape[-1]/sfreq
    old_times = np.arange(0, duration, 1./sfreq)
    new_times = np.arange(0, duration, 1./new_sfreq)

    X_ = []
    for trial_ in range(X.shape[0]):
        X_.append([])
        for band_ in range(X.shape[1]):
            X_[-1].append([])
            for electrode_ in range(X.shape[2]):
                new_signal = resample(X[trial_, band_, electrode_], int(new_times.shape[0]))
                X_[-1][-1].append(new_signal)

    if not many_trials:
        X_ = X_[0]

    return {'data': np.array(X_), 'sfreq': new_sfreq}
    
class fft_resample_eegdata(trial_transform):
    
    '''
    fft_resample_eegdata class to apply FFT-based resampling on EEG data using the trial_transform base class.

    Methods
    -------
    func(trial, **kwargs):
        Applies FFT-based resampling to the given trial.
    '''

    def func(self, trial, **kwargs):
        
        '''
        Applies FFT-based resampling to the given trial.

        Parameters
        ----------
        trial : np.ndarray
            The EEG data trial to resample.
        **kwargs : dict
            Additional arguments to pass to the fft_resample function.

        Returns
        -------
        dict
            A dictionary containing the resampled data and the new sampling frequency.
            {'data': np.ndarray, 'sfreq': float}
        '''
        
        return fft_resample(trial, **kwargs)
    
