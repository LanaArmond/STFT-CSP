import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)

from modules.tf.filterbank import filterbank, filterbank_eegdata
from datasets.BCICIV2a import loading

import numpy as np
import pytest
import scipy.signal
import pandas as pd

class TestClassfilterbank: 
    @pytest.fixture
    def eegData2a(self):
        bciciv2a = loading(verbose='DEBUG')
        return bciciv2a
    
    def test_filterbank(self):
     # Create a sample data
        X = np.random.rand(10, 1, 3, 2*128)

        try:
            X_ = filterbank(X,2*128)
            assert True
        except ValueError:
            assert False

        assert not np.array_equal(X_, X)

        X = np.random.rand(10, 2, 3, 2*128)

        try:
            X_ = filterbank(X,2*128)
            assert False
        except ValueError:
            assert True

        low_cut=[1,2,3,4] 
        high_cut=[1,2,3]

        try:
            X_ = filterbank(X, 2*128, low_cut=low_cut, high_cut=high_cut)
            assert False
        except ValueError:
            assert True
                    

    def test_filterbank_eegdata(self,eegData2a):
        eeg = eegData2a.copy()
        conv = filterbank_eegdata()
        try:
            conv.transform(eeg,sfreq = eeg.sfreq)#receive object eegdata.
            assert True
        except ValueError:
            assert False         
        pass