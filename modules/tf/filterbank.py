
import numpy as np
from modules.tf.bandpass.convolution import bandpass_conv
from modules.tf.bandpass.chebyshevII import chebyshevII
from modules.utils.trial_transform import trial_transform

def filterbank(X, sfreq, low_cut=[4,8,12,16,20,24,28,32,36], high_cut=[8,12,16,20,24,28,32,36,40], kind_bp='conv', **kwargs):
    ''' Filterbank.

    Description
    -----------
    This function implements a filterbank.

    Parameters
    ----------
    X : np.ndarray
        The input data with shape (n_trials, n_bands, n_channels, n_samples).
    sfreq : int
        The sampling frequency.
    low_cut : int or list
        The low cut frequency.
    high_cut : int or list
        The high cut frequency.
    type : str
        The type of filter to use. Options are 'conv' and 'fft'.
    kwargs : dict
        Additional arguments to be passed to the filter function.

    returns
    -------
    np.ndarray
        The filtered data.

    '''
    
    many_trials = len(X.shape) == 4
    if not many_trials:
        X = X[np.newaxis, :, :, :]

    # verify if the data has only one band
    if X.shape[1] != 1:
        raise ValueError('The input data must have only one band.')
    
    # verify if the low_cut and high_cut have the same length
    if len(low_cut) != len(high_cut):
        raise ValueError('The low_cut and high_cut must have the same length.')

    X_ = []
    for trial_ in range(X.shape[0]):
        X_.append([])
        for i in range(len(low_cut)):
            if kind_bp == 'conv':                
                X__ = bandpass_conv(X[trial_],
                                    sfreq, 
                                    low_cut=low_cut[i], 
                                    high_cut=high_cut[i],
                                    kind='same',
                                    **kwargs)[0]
                X_[-1].append(X__)
            elif kind_bp == 'chebyshevII':
                X__ = chebyshevII(X[trial_],
                                    sfreq, 
                                    low_cut=low_cut[i], 
                                    high_cut=high_cut[i],
                                    **kwargs)[0]
                X_[-1].append(X__)

    X_ = np.array(X_)

    if not many_trials:
        X_ = X_[0]

    return X_

class filterbank_eegdata(trial_transform):
    
    def func(self, trial, **kwargs):

        return filterbank(trial, **kwargs)