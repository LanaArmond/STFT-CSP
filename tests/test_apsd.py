import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)

from modules.fe.apsd import apsd 
from datasets.BCICIV2a import loading
import numpy as np
import pytest
import scipy.signal
import pandas as pd

class TestClassApsd: 
    @pytest.fixture
    def eegData2a(self):
        bciciv2a = loading(verbose='DEBUG')
        return bciciv2a

    def test_apsd(self):
        fe = None
        try:
            fe = apsd()
            assert True 
        except ValueError:
            assert False

        factor = np.zeros(5)
        try:
            fe = apsd(flating=factor)
            print(fe.flating)
            print(type(fe.flating))
            assert False 
        except ValueError:
            assert True

        fe = apsd()
        X = np.random.rand(10, 1, 3, 2*128)
        try:
            fe.fit(X)
            assert False 
        except ValueError:
            assert True

        fe = apsd()
        X = np.random.rand(40, 1, 3, 2*128)
        y = np.random.randint(3,size=40)
        data = {"X": X, "y":y}
        try:
            fe.fit(data)
            assert True 
        except ValueError:
            assert False

        fe = apsd()
        X = np.random.rand(40, 1, 3, 2*128)
        y = np.random.randint(3,size=40)
        data = {"X": X, "y":y}
        try:
            fe.transform(data)
            assert True 
        except ValueError:
            assert False
        
    
        fe = apsd()
        X = np.random.rand(40, 1, 3, 2*128)
        y = np.random.randint(3,size=40)
        data = {"X": X, "y":y}
        try:
            fe.fit_transform(data)
            assert True 
        except ValueError:
            assert False

        fe = apsd()
        X = np.random.rand(40, 1, 3, 2*128)
        # y = np.random.randint(3,size=40)
        # data = {"X": X, }
        try:
            fe.fit_transform(X)
            assert False 
        except (ValueError):
            assert True

        fe = apsd()
        X = np.random.rand(40, 1, 3, 2*128)
        # y = np.random.randint(3,size=40)
        # data = {"X": X, }
        try:
            fe.transform(X)
            assert False 
        except (ValueError):
            assert True