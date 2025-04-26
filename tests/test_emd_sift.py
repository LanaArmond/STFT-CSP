import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)
from datasets.BCICIV2a import loading

from modules.tf.emd_sift import EMD, EMD_eegdata
import numpy as np
import pytest
import scipy.signal
import pandas as pd

class TestClassEMD: 
    @pytest.fixture
    def eegData2a(self):
        bciciv2a = loading(verbose='DEBUG')
        return bciciv2a
    
    def test_EMD(self):
     # Create a sample data
        X = np.random.rand(10, 1, 3, 2*128)

        try:
            X_ = EMD(X)
            assert True
        except ValueError:
            assert False

        assert not np.array_equal(X_, X)

        X = np.random.rand(10, 2, 3, 2*128)

        try:
            X_ = EMD(X)
            assert False
        except ValueError:
            assert True

    def test_EMD_eegdata(self,eegData2a):
        eeg = eegData2a.copy()
        conv = EMD_eegdata()
        try:
            conv.transform(eeg)#receive object eegdata.
            assert True
        except ValueError:
            assert False         
        pass