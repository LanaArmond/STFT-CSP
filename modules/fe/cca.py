

import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)

import numpy as np
from typing import Union, List, Optional

from modules.core.eegdata import eegdata

class cca():
    

    def __init__(self, flating: Optional[bool] = False):
        if type(flating) != bool:
            raise ValueError ("Has to be a boolean type value")
        else:
            self.flating = flating
        self.sfreq = None

    def fit(self, data: Union[eegdata, dict], sfreq: int) -> object:
        
        if type(sfreq) == float and (sfreq - int(sfreq)) == 0:
            self.sfreq = sfreq
        if type(sfreq) != int or sfreq <= 0:
            raise ValueError("sfreq must be a positive integer")
        else:
            self.sfreq = sfreq
           
        if type(data) != eegdata and type(data) != dict:
            raise ValueError ("Has to be a eegdata or dict type")        
        return self

    def transform(self, data: Union[eegdata, dict]) -> dict:
        if type(data) != eegdata and type(data) != dict:
            raise ValueError ("Has to be a eegdata or dict type") 
        
        if type(data) == eegdata:
            X = data.data
            y = data.labels
        else:                
            X = data['X']
            y = data['y']

        many_trials = len(X.shape) == 4
        if not many_trials:
            X = X[np.newaxis, :, :, :]

        output = []
        trials_, bands_, channels_, _ = X.shape

        for trial_ in range(trials_):
            output.append([])
            for band_ in range(bands_):
                output[trial_].append([])
                for channel_ in range(channels_):
                    output[trial_][band_].append(np.log(np.mean(X[trial_, band_, channel_, :]**2)))

        output = np.array(output)
        
        if self.flating:
            output = output.reshape(output.shape[0], -1)

        if not many_trials:
            output = output[0]

        return {'X': output, 'y': y}
    
    def fit_transform(self, data: Union[eegdata, dict], sfreq: float) -> dict:
        
        
        return self.fit(data, sfreq).transform(data)