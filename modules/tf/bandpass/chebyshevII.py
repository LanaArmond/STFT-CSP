import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)

import numpy as np

from modules.core.eegdata import eegdata
from modules.utils.trial_transform import trial_transform

def chebyshevII(X, sfreq, low_cut=4, high_cut=40, btype='bandpass', order=4, rs='auto'):
    ''' Bandpass filter using Chebyshev type II filter.

    Description
    -----------
    This function implements a bandpass filter using Chebyshev type II filter.

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
    btype : str
        The type of filter. It can be 'lowpass', 'highpass', 'bandpass', or 'bandstop'.
    order : int
        The order of the filter.
        For Chebyshev type II filter, the order must be even.
    rs : int
        The minimum attenuation in the stop band.
        If 'auto', the value is set to 40 for bandpass filters and 20 for other filters.

    returns
    -------
    np.ndarray
        The filtered data.

    '''
    from scipy.signal import cheby2, filtfilt

    Wn = [low_cut, high_cut]
    
    if rs == 'auto':
        if btype == 'bandpass':
            rs = 40
        else:
            rs = 20

    many_trials = len(X.shape) == 4
    if not many_trials:
        X = X[np.newaxis, :, :, :]

    X_ = []
    for trial_ in range(X.shape[0]):
        X_.append([])
        for band_ in range(X.shape[1]):
            X_[-1].append([])
            for electrode_ in range(X.shape[2]):
                filtered = filtfilt(*cheby2(order, rs, Wn, btype, fs=sfreq), X[trial_, band_, electrode_])
                X_[-1][-1].append(filtered)

    X_ = np.array(X_)

    if not many_trials:
        X_ = X_[0]

    return X_

class chebyshevII_eegdata(trial_transform):
    ''' Bandpass filter using Chebyshev type II filter.

    Description
    -----------
    This class implements a bandpass filter using Chebyshev type II filter.

    Parameters
    ----------
    sfreq : int
        The sampling frequency.
    low_cut : int
        The low cut frequency.
    high_cut : int
        The high cut frequency.
    btype : str
        The type of filter. It can be 'lowpass', 'highpass', 'bandpass', or 'bandstop'.

    '''

    def func(self, trial, **kwargs):
        ''' Filters the data.

        Parameters
        ----------
        trial : eegdata
            The input data.

        returns
        -------
        eegdata
            The filtered data.

        '''
        return chebyshevII(trial, **kwargs)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Create a sample data
    X = np.random.rand(10, 5, 3, 2*128)
    
    # Filter the data
    import scipy.signal

    # # f contains the frequency components
    # # S is the PSD
    # (f, S) = scipy.signal.periodogram(X, 2*128)

    # print(S[0,0,0,:])
    # for trial_ in range(2):
    #     for band_ in range(2):
    #         for electrode_ in range(2):
    #             plt.scatter(f[1:], S[trial_, band_, electrode_,1:])
    #             plt.xlabel('frequency [Hz]')
    #             plt.ylabel('PSD [V**2/Hz]')
    #             plt.show()

    X_ = chebyshevII(X, 2*128, low_cut=4, high_cut=40)
    # if np.any(X_ > 40):
    #     print("false")
    # print(X_.shape)
    # # Plot the data
    # (f, S) = scipy.signal.periodogram(X_, 2*128)
    # print(S[0,0,0,:])
    # for trial_ in range(2):
    #     for band_ in range(2):
    #         for electrode_ in range(2):
    #             plt.scatter(f[1:], S[trial_, band_, electrode_,1:])
    #             plt.xlabel('frequency [Hz]')
    #             plt.ylabel('PSD [V**2/Hz]')
    #             plt.show()   
    for trial_ in range(10):
        for band_ in range(5):
            for electrode_ in range(3):
                plt.plot(X[trial_, band_, electrode_], 'b')
                plt.plot(X_[trial_, band_, electrode_], 'r')
                plt.show()

    print('Done')
