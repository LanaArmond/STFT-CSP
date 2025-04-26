import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)
from datasets.BCICIV2a import loading

from modules.tf.resample_fft import fft_resample, fft_resample_eegdata

import numpy as np
import pytest
import scipy.signal
import pandas as pd

class TestClassSTFT: 
    @pytest.fixture
    def eegData2a(self):
        bciciv2a = loading(verbose='DEBUG')
        return bciciv2a
    
    def test_fft_resample(self):
     # Create a sample data
        X = np.random.rand(10, 1, 3, 2*128)

        try:
            X_ = X_ = fft_resample(X, 2*128,128)
            assert True
        except ValueError:
            assert False
        
        assert not np.array_equal(X_['data'], X)

        X = np.random.rand(1, 3, 2*128)

        try:
            X_ = fft_resample(X,2*128,128)
            assert True
        except ValueError:
            assert False

    def test_fft_resample_eegdata(self,eegData2a):
        eeg = eegData2a.copy()
        conv = fft_resample_eegdata()
        try:
            conv.transform(eeg,sfreq = eeg.sfreq, new_sfreq= eeg.sfreq*2)#receive object eegdata.
            assert True
        except ValueError:
            assert False         
        pass