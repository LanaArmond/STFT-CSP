import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)
from datasets.BCICIV2a import loading

from modules.tf.stft import STFT, STFT_eegdata
import numpy as np
import pytest
import scipy.signal
import pandas as pd

class TestClassSTFT: 
    @pytest.fixture
    def eegData2a(self):
        bciciv2a = loading(verbose='DEBUG')
        return bciciv2a
    
    def test_STFT(self):
     # Create a sample data
        X = np.random.rand(10, 1, 3, 2*128)

        try:
            X_ = STFT(X, sfreq=128)
            assert True
        except ValueError:
            assert False

        assert not np.array_equal(X_, X)

        X = np.random.rand(10, 2, 3, 2*128)

        try:
            X_ = STFT(X, sfreq=128)
            assert False
        except ValueError:
            assert True
        
        X = np.random.rand(10, 1, 3, 2*128)

        try:
            X_ = STFT(X, sfreq=128, freqs_per_bands=6)
            assert True
        except ValueError:
            assert False

        X = np.random.rand(10, 1, 3, 2*128)
        try:
            X_ = STFT(X, sfreq=2*128, nperseg=2*128, noverlap=2*128-1, nfft=2*128)
            assert True
        except ValueError:
            assert False



    def test_STFT_eegdata(self,eegData2a):
        eeg = eegData2a.copy()
        conv = STFT_eegdata()
        try:
            conv.transform(eeg,sfreq = eeg.sfreq)#receive object eegdata.
            assert True
        except ValueError:
            assert False         
        pass