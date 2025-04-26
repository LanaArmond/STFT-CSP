import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)

from datasets.CBCIC import loading
import numpy as np
import pytest
import scipy.signal
import pandas as pd

class TestClassCBCIC: 
    
    def test_CBCIC(self):
        cbcic = None
        try:
            cbcic = loading(verbose='DEBUG')
            assert True
        except ValueError:
            assert False
        try:
            cbcic = loading()
            assert True
        except ValueError:
            assert False

        try:
            cbcic = loading(subject=10)
            assert False
        except ValueError:
            assert True

        try:
            cbcic = loading(subject=1.3)
            assert False
        except ValueError:
            assert True

        try:
            cbcic = loading(events_dict=2)
            assert False
        except ValueError:
            assert True

        array = np.zeros(4)
        try:
            cbcic = loading(session_list=array)
            assert False
        except ValueError:
            assert True

        try:
            cbcic = loading(run_list=array)
            assert False
        except ValueError:
            assert True
        
        try:
            cbcic = loading(verbose=array)
            assert False
        except ValueError:
            assert True

        return cbcic
