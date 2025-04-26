import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)

from modules.fe.logpower import logpower 
from datasets.BCICIV2a import loading
import numpy as np
import pytest
import scipy.signal
import pandas as pd

class TestClassLogpower: 
    @pytest.fixture
    def eegData2a(self):
        bciciv2a = loading(verbose='DEBUG')
        return bciciv2a

    def test_logpower(self):
        fe = None
        try:
            fe = logpower()
            assert True 
        except ValueError:
            assert False

        factor = np.zeros(5)
        try:
            fe = logpower(flating=factor)
            print(fe.flating)
            print(type(fe.flating))
            assert False 
        except ValueError:
            assert True

        fe = logpower()
        X = np.random.rand(10, 1, 3, 2*128)
        try:
            fe.fit(X)
            assert False 
        except ValueError:
            assert True

        fe = logpower()
        X = np.random.rand(40, 1, 3, 2*128)
        y = np.random.randint(3,size=40)
        data = {"X": X, "y":y}
        try:
            fe.fit(data)
            assert True 
        except ValueError:
            assert False

        fe = logpower()
        X = np.random.rand(40, 1, 3, 2*128)
        y = np.random.randint(3,size=40)
        data = {"X": X, "y":y}
        try:
            fe.transform(data)
            assert True 
        except ValueError:
            assert False
        
    
        fe = logpower()
        X = np.random.rand(40, 1, 3, 2*128)
        y = np.random.randint(3,size=40)
        data = {"X": X, "y":y}
        try:
            fe.fit_transform(data)
            assert True 
        except ValueError:
            assert False

        fe = logpower()
        X = np.random.rand(40, 1, 3, 2*128)
        # y = np.random.randint(3,size=40)
        # data = {"X": X, }
        try:
            fe.fit_transform(X)
            assert False 
        except (ValueError):
            assert True

        fe = logpower()
        X = np.random.rand(40, 1, 3, 2*128)
        # y = np.random.randint(3,size=40)
        # data = {"X": X, }
        try:
            fe.transform(X)
            assert False 
        except (ValueError):
            assert True