'''
BCICIV2b.py

Description
-----------
This code is used to load EEG data from the BCICIV2b dataset. It modifies the data to fit the requirements of the eegdata class, which is used to store and process EEG data. 

Dependencies
------------
numpy
pandas
scipy
mne 

'''
import numpy as np
import pandas as pd
import scipy
import mne

import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)

from modules.core.eegdata import *

def loading(subject: int=1, session_list: list=None, run_list: list=None, events_dict: dict={'left-hand': 1, 'right-hand': 2}, verbose: str='INFO'):

    """
    Description
    -----------
    
    Load EEG data from the BCICIV2b dataset. 
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

    sfreq = 250.
    events = {'get_start': [0, 3],
                'beep_sound': [2],
                'cue': [3, 4],
                'task_exec': [4, 7],
                'break': [7, 8.5]}
    ch_names = ['C3', 'Cz', 'C4']
    ch_names = np.array(ch_names)
    tmin = 0.

    """
    'sfreq' is set to 250. This represents the sampling frequency of the EEG data.
    'events' is a dictionary that maps event names to their corresponding time intervals.
    'ch_names' is a list of channel names.
    'tmin' is set to 0, representing the starting time of the EEG data.
    """

    if session_list is None:
        session_list = ['01T', '02T', '03T', '04E', '05E']

    rawData, rawLabels = [], []

    """
    If 'session_list' is not provided, it is set to ['01T', '02T', '03T', '04E', '05E'].
    'rawData' and 'rawLabels' are empty lists that will store the EEG data and labels for each session.
    """
    for sec in session_list:
        raw=mne.io.read_raw_gdf('Datasets/Data/BCICIV2b/B%02d%s.gdf'%(subject, sec), preload=True, verbose='ERROR')
        raw_data = raw.get_data()[:3]
        annotations = raw.annotations.to_data_frame()
        first_timestamp = pd.to_datetime(annotations['onset'].iloc[0])
        annotations['onset'] = (pd.to_datetime(annotations['onset']) - first_timestamp).dt.total_seconds()
        annotations['description'] = annotations['description'].astype(int)
        new_trial_time = np.array(annotations[annotations['description']==768]['onset'])

        times_ = np.array(raw.times)
        rawData_ = []
        for trial_ in new_trial_time:
            idx_ = np.where(times_ == trial_)[0][0]
            rawData_.append(raw_data[:, idx_:idx_+2125])
        rawData_ = np.array(rawData_)
        rawLabels_ = np.array(scipy.io.loadmat('Datasets/Data/BCICIV2b/B%02d%s.mat'%(subject, sec))['classlabel']).reshape(-1)

        rawData.append(rawData_)
        rawLabels.append(rawLabels_)

    """
    For each session in the 'session_list', the raw EEG data is loaded using mne.io.read_raw_gdf.
    The data is filtered to include only the first three channels.
    The 'annotations' (relevant timestamps) are extracted and converted to a DataFrame.
    The onset times are normalized to start from zero.
    The event descriptions are converted to integers.
    The 'new_trial_time' is obtained by extracting the onset times of the '768' event ('768' is the code for new trial in the dataset).
    The 'times_' array is obtained from the raw data.
    The EEG data is extracted for each trial based on the 'new_trial_time'.
    The data is reshaped to include only the relevant channels and time points.
    The class labels are loaded from the corresponding .mat file.
    The raw data and labels are appended to the 'rawData' and 'rawLabels' lists.
    """

    data, labels = np.concatenate(rawData), np.concatenate(rawLabels)
    idx_labels = np.isin(labels, list(map(events_dict.get, events_dict.keys())))
    data, labels = data[idx_labels], labels[idx_labels]

    data = np.reshape(data, (data.shape[0], 1, data.shape[1], data.shape[2]))

    """
    The 'data' and 'labels' are concatenated across sessions.
    The 'labels' are filtered based on the event codes specified in the 'events_dict'.
    The trial data is reshaped into a 4-dimensional array to match the expected eegdata format.
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
    bciciv2b = loading(verbose="DEBUG")

    """
    The main block of the code sets the global verbosity level for the eegdata class to 'DEBUG'.
    The loading function is called with verbose='DEBUG' to load the BCICIV2b dataset
    """