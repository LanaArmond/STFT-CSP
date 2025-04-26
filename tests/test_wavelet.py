import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)
from datasets.BCICIV2a import loading

from modules.tf.wavelet import wavelet, wavelet_eegdata
import numpy as np
import pytest
import scipy.signal
import pandas as pd

class TestClassSTFT: 
    @pytest.fixture
    def eegData2a(self):
        bciciv2a = loading(verbose='DEBUG')
        return bciciv2a
    
    def test_wavelet(self):
     # Create a sample data
        X = np.random.rand(10, 1, 3, 2*128)

        try:
            X_ = X_ = wavelet(X, levels=5)
            assert True
        except ValueError:
            assert False

        assert not np.array_equal(X_, X)

        X = np.random.rand(1, 3, 2*128)

        try:
            X_ = wavelet(X,levels=5)
            assert True
        except ValueError:
            assert False

    def test_wavelet_eegdata(self,eegData2a):
        eeg = eegData2a.copy()
        conv = wavelet_eegdata()
        try:
            conv.transform(eeg,levels=5)#receive object eegdata.
            assert True
        except ValueError:
            assert False         
        pass