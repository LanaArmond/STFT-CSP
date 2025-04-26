import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)

from datasets.BCICIV2b import loading
import numpy as np
import pytest
import scipy.signal
import pandas as pd

class TestClassBCICIV2b: 
    
    def test_BCICIV2b(self):
        bciciv2b = None
        try:
            bciciv2b = loading(verbose='DEBUG')
            assert True
        except ValueError:
            assert False
        try:
            bciciv2b = loading()
            assert True
        except ValueError:
            assert False

        try:
            bciciv2b = loading(subject=10)
            assert False
        except ValueError:
            assert True

        try:
            bciciv2b = loading(subject=1.3)
            assert False
        except ValueError:
            assert True

        try:
            bciciv2b = loading(events_dict=2)
            assert False
        except ValueError:
            assert True

        array = np.zeros(4)
        try:
            bciciv2b = loading(session_list=array)
            assert False
        except ValueError:
            assert True

        try:
            bciciv2b = loading(run_list=array)
            assert False
        except ValueError:
            assert True
        
        try:
            bciciv2b = loading(verbose=array)
            assert False
        except ValueError:
            assert True

        return bciciv2b
