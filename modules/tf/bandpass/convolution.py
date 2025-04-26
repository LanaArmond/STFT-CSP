''' 
bandpass.py

Description
-----------
This module contains the implementation of the bandpass filter.

Dependencies
------------
eegdata on modules/core
trial_transform on modules/utils
numpy
scipy

'''

import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)

import numpy as np

from modules.core.eegdata import eegdata
from modules.utils.trial_transform import trial_transform

def bandpass_conv(X, sfreq, low_cut=4, high_cut=40, transition=None, window_type='hamming', kind='same'):
    ''' Bandpass filter using convolution.

    Description
    -----------
    This function implements a bandpass filter using convolution.

    Parameters
    ----------
    X : np.ndarray
        The input data.
    sfreq : int
        The sampling frequency.
    low_cut : int
        The low cut frequency.
    high_cut : int
        The high cut frequency.
    transition : int or float or list
        The transition bandwidth. If int or float, the same value is used for both low and high cut frequencies.
    window_type : str
        The window type for the filter.
    kind : str
        The mode for the convolution.

    returns
    -------
    np.ndarray
        The filtered data.

    '''
    
    many_trials = len(X.shape) == 4
    if not many_trials:
        X = X[np.newaxis, :, :, :]

    if transition is None:
        transition = (high_cut - low_cut) / 2
    if isinstance(transition, int):
        transition = float(transition)
    if isinstance(transition, float):
        transition = [transition, transition]

    NL = int(4 * sfreq / transition[0])
    NH = int(4 * sfreq / transition[1])


    hlpf = np.sinc(2 * high_cut / sfreq * (np.arange(NH) - (NH - 1) / 2))
    if window_type=='hamming':
        hlpf *= np.hamming(NH)
    elif window_type=='blackman':
        hlpf *= np.blackman(NH)
    hlpf /= np.sum(hlpf)

    hhpf = np.sinc(2 * low_cut / sfreq * (np.arange(NL) - (NL - 1) / 2))
    if window_type=='hamming':
        hhpf *= np.hamming(NL)
    elif window_type=='blackman':
        hhpf *= np.blackman(NL)
    hhpf = -hhpf
    hhpf[(NL - 1) // 2] += 1

    kernel = np.convolve(hlpf, hhpf)
    if len(kernel) > X.shape[-1] and kind == 'same':
        kind = 'valid'

    X_ = []
    for trial_ in range(X.shape[0]):
        X_.append([])
        for band_ in range(X.shape[1]):
            X_[-1].append([])
            for electrode_ in range(X.shape[2]):
                filtered = np.convolve(X[trial_, band_, electrode_], kernel, mode=kind)
                X_[-1][-1].append(filtered)

    X_ = np.array(X_)

    if not many_trials:
        X_ = X_[0]

    return X_

class bandpass_conv_eegdata(trial_transform):
    ''' Bandpass filter using convolution.

    Description
    -----------
    This class implements a bandpass filter using convolution.

    attributes
    ----------
    None

    methods
    -------
    func : np.ndarray
        The filtered data.

    '''

    def func(self, trial, **kwargs):
        ''' Filters the data.

        Parameters
        ----------
        trial : np.ndarray
            The input data.
        
        returns
        -------
        np.ndarray
            The filtered data.

        '''
        
        return bandpass_conv(trial, **kwargs)