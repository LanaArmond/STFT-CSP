import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)

from datasets.BCICIV2a import loading
import numpy as np
import pytest
import scipy.signal
import pandas as pd

class TestClassBCICIV2a: 
    
    def test_BCICIV2a(self):
        bciciv2a = None
        try:
            bciciv2a = loading(verbose='DEBUG')
            assert True
        except ValueError:
            assert False
        try:
            bciciv2a = loading()
            assert True
        except ValueError:
            assert False

        try:
            bciciv2a = loading(subject=10)
            assert False
        except ValueError:
            assert True

        try:
            bciciv2a = loading(subject=1.3)
            assert False
        except ValueError:
            assert True

        try:
            bciciv2a = loading(events_dict=2)
            assert False
        except ValueError:
            assert True

        array = np.zeros(4)
        try:
            bciciv2a = loading(session_list=array)
            assert False
        except ValueError:
            assert True

        try:
            bciciv2a = loading(run_list=array)
            assert False
        except ValueError:
            assert True
        
        try:
            bciciv2a = loading(verbose=array)
            assert False
        except ValueError:
            assert True

        return bciciv2a
