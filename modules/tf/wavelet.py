import os, sys, copy
project_directory = os.getcwd()
sys.path.append(project_directory)

import numpy as np
import pywt

from modules.core.eegdata import eegdata
from modules.utils.trial_transform import trial_transform

def wavelet(X, levels=5):
    ''' Wavelet transform.

    Description
    -----------
    This function implements the wavelet transform.

    Parameters
    ----------
    X : np.ndarray
        The input data.
    levels : int
        The number of levels.

    returns
    -------
    np.ndarray
        The transformed data.

    '''

    many_trials = len(X.shape) == 4
    if not many_trials:
        X = X[np.newaxis, :, :, :]


    widths = np.arange(1, levels+1)
    X_ = []
    for trial_ in range(X.shape[0]):
        X_.append([])
        for electrode_ in range(X.shape[2]):
            coef_, freqs_ = pywt.cwt(X[trial_, 0, electrode_], widths, 'morl')
            X_[-1].append(coef_)

    X_ = np.array(X_)
    X_ = np.transpose(X_, (0, 2, 1, 3))
    
    if not many_trials:
        X_ = X_[0]

    return X_

class wavelet_eegdata(trial_transform):

    def func(self, trial, **kwargs):
        return wavelet(trial, **kwargs)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Create a sample data
    X = np.random.rand(10, 1, 3, 2*128)

    # Filter the data
    X_ = wavelet(X, levels=5)

    # Plot the data
    for trial_ in range(10):
        for band_ in range(5):
            for electrode_ in range(3):
                plt.plot(X[trial_, 0, electrode_], 'b')
                plt.plot(X_[trial_, band_, electrode_], 'r')
                plt.show()

    print('Done')
