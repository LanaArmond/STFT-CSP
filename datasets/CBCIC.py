'''
CBCIC.py

Description
-----------
This code is used to load EEG data from the CBCIC dataset. It modifies the data to fit the requirements of the eegdata class, which is used to store and process EEG data.

Dependencies
------------
numpy
pandas
scipy
mne 

'''

#https://sites.google.com/view/bci-comp-wcci/
#Rever mistura de T com E
import numpy as np
import pandas as pd
import scipy
import mne

import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)

from modules.core.eegdata import *


def loading(subject: int=1, session_list: list=None, run_list: list=None, events_dict: dict={'right-hand': 1, 'left-hand': 2}, verbose: dict='INFO'):

    """
    Description
    -----------
    
    Load EEG data from the CBCIC dataset. 
    It modifies the data to fit the requirements of the eegdata class, which is used to store and process EEG data. 

    Parameters
    ----------
        subject : int
            index of the subject to retrieve the data from
        session_list : list, optional
            list of session codes
        run_list : list, optional
            list of run numbers
        events_dict : dict
            dictionary mapping event names to event codes
        verbose : str
            verbosity level


    Returns:
    ----------
        eegdata: An instance of the eegdata class containing the loaded EEG data.

    """
    if type(subject) != int:
        raise ValueError("Has to be a int type value")
    if subject > 9:
        raise ValueError("Has to be an existing subject")
    if type(session_list) != list and session_list != None:
        raise ValueError("Has to be an List or None type")
    if type(run_list) != list and run_list != None:
        raise ValueError("Has to be an List or None type")
    if type(events_dict) != dict:
        raise ValueError("Has to be an dict type")
    if type(verbose) != type(str()):
        raise ValueError("Has to be an string type")

    sfreq = 512.
    events = {'get_start': [0, 3],
            'beep_sound': [2],
            'cue': [3, 8],
            'task_exec': [3, 8]}
    ch_names = ["F3", "FC3", "C3", "CP3", "P3", "FCz", "CPz", "F4", "FC4", "C4", "CP4", "P4"]
    ch_names = np.array(ch_names)
    tmin = 0.

    """
    'sfreq' is set to 512. This represents the sampling frequency of the EEG data.
    'events' is a dictionary that maps event names to their corresponding time intervals.
    'ch_names' is a list of channel names.
    'tmin' is set to 0, representing the starting time of the EEG data.
    """

    if session_list is None:
        session_list = ['T', 'E']

    rawData, rawLabels = [], []

    """
    If 'session_list' is not provided, it is set to ['01T', '02T', '03T', '04E', '05E'].
    'rawData' and 'rawLabels' are empty lists that will store the EEG data and labels for each session.
    """

    for sec in session_list:
        raw=scipy.io.loadmat('Datasets/Data/CBCIC/parsed_P%02d%s.mat'%(subject, sec))
        rawData_ = raw['RawEEGData']
        rawLabels_ = np.reshape(raw['Labels'], -1)
        rawData_ = np.reshape(rawData_, (rawData_.shape[0], 1, rawData_.shape[1], rawData_.shape[2]))
        rawData.append(rawData_)
        rawLabels.append(rawLabels_)
        """
        For each session in 'session_list', the EEG data is loaded from the .mat file
        'rawData_' and 'rawLabels_' store the EEG data and labels for the current session
        'rawData_' is reshaped into a 4-dimensional array to match the expected eegdata format
        'rawData_' and 'rawLabels_' are appended to 'rawData' and 'rawLabels', respectively
        """
    
    data, labels = np.concatenate(rawData), np.concatenate(rawLabels)
    
    """
    'data' and 'labels' are created by concatenating the EEG data and labels from all sessions
    """

    return eegdata(data=data, 
                   sfreq=sfreq, 
                   labels=labels, 
                   labels_dict=events_dict,
                   events=events, 
                   ch_names=ch_names, 
                   tmin=tmin,
                   verbose=verbose)
    """
    An instance of the eegdata class is created with the loaded EEG data and relevant information.
    """

if __name__  == "__main__":

    set_global_verbose_eegdata('DEBUG')
    cbcic = loading(verbose="DEBUG")
    """
    The main block of the code sets the global verbosity level for the eegdata class to 'DEBUG'.
    The loading function is called with verbose='DEBUG' to load the CBCIC dataset
    """
