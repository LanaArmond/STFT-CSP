from modules.core.kfold import kfold, find_key_with_value
import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)

from modules.core.eegdata import eegdata

from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from typing import Union, List, Optional
import logging
from datasets.BCICIV2a import BCICIV2a

class TestClassKfold:
    kfoldObj = None
    n_splits = int
    shuffle = bool
    pre_folding = dict
    pos_folding = dict
    window_size = float
    
    def test_init_kfold(self):
        kfoldObj = kfold(n_splits=self.n_splits,
                    shuffle=self.shuffle,
                    pre_folding=self.pre_folding,
                    pos_folding=self.pos_folding,
                    window_size=self.window_size)
        self.kfoldObj = kfoldObj
        assert  type(kfoldObj) == kfold
    #EEG info
    eeg = BCICIV2a().load()
    #exec parameters
    target = eeg
    train_window = list
    test_window = list
    def test_exec_kfold(self):
        kfoldObj = kfold(n_splits=self.n_splits,
                    shuffle=self.shuffle,
                    pre_folding=self.pre_folding,
                    pos_folding=self.pos_folding,
                    window_size=self.window_size)
        result = kfoldObj.exec(target=self.target,
                               train_window=self.train_window,
                               test_window=self.test_window)
        assert  isinstance(result, pd.DataFrame) == True
    def test_find_key_with_value(self):
        dictionaryUsed = {
                            "BCI": "Project",
                            "Method": "Loreta",
                            "year": 2002
                        }
        valueToSearchFor1 = "Project"
        valueToSearchFor2 = "sLoreta"
        result1 =  find_key_with_value(dictionary=dictionaryUsed, i=valueToSearchFor1)
        result2 =  find_key_with_value(dictionary=dictionaryUsed, i=valueToSearchFor2)
        key1 = "BCI"
        key2 = "Method"
        assert result1 == key1 and result2 != key2