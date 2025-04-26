'''
resample_cubic.py

Description
-----------
This module contains the implementation of cubic resampling for EEG data. 
The cubic_resample function uses cubic splines to resample the input signals to a new sampling frequency.

Dependencies
------------
numpy, scipy, modules.core.eegdata, modules.utils.trial_transform

'''

import numpy as np
from scipy.interpolate import CubicSpline

from modules.core.eegdata import eegdata
from modules.utils.trial_transform import trial_transform

def cubic_resample(X, sfreq, new_sfreq):
    
    '''
    Resamples the input EEG data to a new sampling frequency using cubic splines.

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
                cubic_spline = CubicSpline(old_times, X[trial_, band_, electrode_])
                new_signal = cubic_spline(new_times)
                X_[-1][-1].append(new_signal)

    if not many_trials:
        X_ = X_[0]

    return {'data': np.array(X_), 'sfreq': new_sfreq}

class cubic_resample_eegdata(trial_transform):
    
    '''
    cubic_resample_eegdata class to apply cubic resampling on EEG data using the trial_transform base class.

    Methods
    -------
    func(trial, **kwargs):
        Applies cubic resampling to the given trial.
    '''

    def func(self, trial, **kwargs):
        
        '''
        Applies cubic resampling to the given trial.

        Parameters
        ----------
        trial : np.ndarray
            The EEG data trial to resample.
        **kwargs : dict
            Additional arguments to pass to the cubic_resample function.

        Returns
        -------
        dict
            A dictionary containing the resampled data and the new sampling frequency.
            {'data': np.ndarray, 'sfreq': float}
        '''
        
        return cubic_resample(trial, **kwargs)
    