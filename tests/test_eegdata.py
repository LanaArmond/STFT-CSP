import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)

from modules.core.eegdata import eegdata
from datasets.BCICIV2a import loading
import numpy as np
import pytest

import pandas as pd

class TestClassEegdata: 
    
    # eeg = None
    # id = int
    # data = np.array
    # sfreq = int 
    # n_trials : int
    # n_bands : int
    # n_channels : int
    # n_times : int
    # tmin : float
    # timestamps : np.ndarray
    # events : dict
    # labels : np.ndarray
    # labels_dict : dict
    # ch_names : np.ndarray
    # local_logger : logging.Logger
    # verbose : str
    
    @pytest.fixture
    def eegData2a(self):
        bciciv2a = loading(verbose='DEBUG')
        return bciciv2a
    
    def test_init_eegdata(self,eegData2a):
        eeg = eegData2a.copy()
        
        assert  type(eeg) == eegdata

        dataUsed = eeg.data
        try:
            eegaux = eegdata(data=dataUsed, 
                    sfreq=eeg.sfreq, 
                    labels=eeg.labels, 
                    labels_dict=eeg.labels_dict,
                    events=eeg.events, 
                    ch_names=eeg.ch_names, 
                    tmin=eeg.tmin,
                    verbose=None, #
                                )
            assert True
        except ValueError:
            assert False
        try:
            eegaux = eegdata(data=dataUsed, 
                    sfreq=eeg.sfreq, 
                    labels=eeg.labels, 
                    labels_dict=eeg.labels_dict,
                    events=eeg.events, 
                    ch_names=eeg.ch_names, 
                    tmin=eeg.tmin,
                    verbose="None", #
                                )
            assert False
        except ValueError:
            assert True

        try:
            eegaux = eegdata(data=dataUsed, 
                    sfreq=eeg.sfreq, 
                    labels=eeg.labels, 
                    labels_dict=eeg.labels_dict,
                    events=eeg.events, 
                    ch_names=eeg.ch_names, 
                    tmin=eeg.tmin,
                    verbose="Debug", #
                                )
            assert False
        except ValueError:
            assert True
        
        try:
            eegaux = eegdata(data=dataUsed, 
                    sfreq=eeg.sfreq, 
                    labels=eeg.labels, 
                    labels_dict=eeg.labels_dict,
                    events=eeg.events, 
                    ch_names=eeg.ch_names, 
                    tmin=eeg.tmin,
                    verbose="DEBUG", #
                                )
            assert True
        except ValueError:
            assert False

        dataUsed2 = {
                            "BCI": "Project",
                            "Method": "Loreta",
                            "year": 2002
                        }
        try:
            eegaux = eegdata(data=dataUsed2, 
                    sfreq=eeg.sfreq, 
                    labels=eeg.labels, 
                    labels_dict=eeg.labels_dict,
                    events=eeg.events, 
                    ch_names=eeg.ch_names, 
                    tmin=eeg.tmin,
                    verbose=eeg.verbose, #
                                )
            assert False
        except ValueError:
            assert True
        
        dataUsed2 = np.random.randn(5, 3, 3, 2,1)
        # assert type(dataUsed2.dtype) != np.float64
        try:
            eegaux = eegdata(data=dataUsed2, 
                    sfreq=eeg.sfreq, 
                    labels=eeg.labels, 
                    labels_dict=eeg.labels_dict,
                    events=eeg.events, 
                    ch_names=eeg.ch_names, 
                    tmin=eeg.tmin,
                    verbose=eeg.verbose, #
                                )
            assert False
        except ValueError:
            assert True
        dataUsed2 = np.random.randn(576, 1, 22, 1875)
        
        try:
            eegaux = eegdata(data=dataUsed2, 
                    sfreq=eeg.sfreq, 
                    labels=eeg.labels, 
                    labels_dict=eeg.labels_dict,
                    events=eeg.events, 
                    ch_names=eeg.ch_names, 
                    tmin=eeg.tmin,
                    verbose=eeg.verbose, #
                                )
            assert True
        except ValueError:
            assert False
        try:
            eegaux = eegdata(data=eeg.data, 
                    sfreq=1.23, 
                    labels=eeg.labels, 
                    labels_dict=eeg.labels_dict,
                    events=eeg.events, 
                    ch_names=eeg.ch_names, 
                    tmin=eeg.tmin,
                    verbose=eeg.verbose, #
                                )
            assert False
        except ValueError:
            assert True
        try:
            eegaux = eegdata(data=eeg.data, 
                    sfreq=-1, 
                    labels=eeg.labels, 
                    labels_dict=eeg.labels_dict,
                    events=eeg.events, 
                    ch_names=eeg.ch_names, 
                    tmin=eeg.tmin,
                    verbose=eeg.verbose, #
                                )
            assert False
        except ValueError:
            assert True

        try:
            eegaux = eegdata(data=eeg.data, 
                    sfreq=0, 
                    labels=eeg.labels, 
                    labels_dict=eeg.labels_dict,
                    events=eeg.events, 
                    ch_names=eeg.ch_names, 
                    tmin=eeg.tmin,
                    verbose=eeg.verbose, #
                                )
            assert False
        except ValueError:
            assert True
        #
        # O label n pode ser None, mas ele aceita ser none e dps dá erropois pede o shape dele la na frente.   
        try:
            eegaux = eegdata(data=eeg.data, 
                    sfreq=eeg.sfreq, 
                    labels=None, 
                    labels_dict=eeg.labels_dict,
                    events=eeg.events, 
                    ch_names=eeg.ch_names, 
                    tmin=eeg.tmin,
                    verbose=eeg.verbose, #
                                )
            assert True
        except ValueError:
            assert False
            
        try:
            eegaux = eegdata(data=eeg.data, 
                    sfreq=eeg.sfreq, 
                    labels={
                            "BCI": "Project",
                            "Method": "Loreta",
                            "year": 2002
                        }, 
                    labels_dict=eeg.labels_dict,
                    events=eeg.events, 
                    ch_names=eeg.ch_names, 
                    tmin=eeg.tmin,
                    verbose=eeg.verbose, #
                                )
            assert False
        except ValueError:
            assert True
        
        try:
            eegaux = eegdata(data=eeg.data, 
                    sfreq=eeg.sfreq, 
                    labels=2, 
                    labels_dict=eeg.labels_dict,
                    events=eeg.events, 
                    ch_names=eeg.ch_names, 
                    tmin=eeg.tmin,
                    verbose=eeg.verbose, #
                                )
            assert False
        except ValueError:
            assert True
        try:
            eegaux = eegdata(data=eeg.data, 
                    sfreq=eeg.sfreq, 
                    labels=np.random.randn(2,2), 
                    labels_dict=eeg.labels_dict,
                    events=eeg.events, 
                    ch_names=eeg.ch_names, 
                    tmin=eeg.tmin,
                    verbose=eeg.verbose, #
                                )
            assert False
        except ValueError:
            assert True

        try:
            eegaux = eegdata(data=eeg.data, 
                    sfreq=eeg.sfreq, 
                    labels=np.random.randn(2), 
                    labels_dict=eeg.labels_dict,
                    events=eeg.events, 
                    ch_names=eeg.ch_names, 
                    tmin=eeg.tmin,
                    verbose=eeg.verbose, #
                                )
            assert False
        except ValueError:
            assert True

        try:
            eegaux = eegdata(data=eeg.data, 
                    sfreq=eeg.sfreq, 
                    labels=eeg.labels, 
                    labels_dict=np.random.randn(2),
                    events=eeg.events, 
                    ch_names=eeg.ch_names, 
                    tmin=eeg.tmin,
                    verbose=eeg.verbose, #
                                )
            assert False
        except ValueError:
            assert True
        
        try:
            eegaux = eegdata(data=eeg.data, 
                    sfreq=eeg.sfreq, 
                    labels=eeg.labels, 
                    labels_dict={
                            'left-hand': 0, 'right-hand': 1, 'both-feet': 2
                        } ,
                    events=eeg.events, 
                    ch_names=eeg.ch_names, 
                    tmin=eeg.tmin,
                    verbose=eeg.verbose, #
                                )
            assert False
        except ValueError:
            assert True

        try:
            eegaux = eegdata(data=eeg.data, 
                    sfreq=eeg.sfreq, 
                    labels=eeg.labels, 
                    labels_dict=eeg.labels_dict,
                    events=None, 
                    ch_names=eeg.ch_names, 
                    tmin=eeg.tmin,
                    verbose=eeg.verbose, #
                                )
            assert True
        except ValueError:
            assert False

        try:
            eegaux = eegdata(data=eeg.data, 
                    sfreq=eeg.sfreq, 
                    labels=eeg.labels, 
                    labels_dict=eeg.labels_dict,
                    events=np.random.randn(2), 
                    ch_names=eeg.ch_names, 
                    tmin=eeg.tmin,
                    verbose=eeg.verbose, #
                                )
            assert False
        except ValueError:
            assert True
        
        try:
            eegaux = eegdata(data=eeg.data, 
                    sfreq=eeg.sfreq, 
                    labels=eeg.labels, 
                    labels_dict=eeg.labels_dict,
                    events={
                            "BCI": 1,
                            "Method": 2,
                            "year": 3,
                            "month": 4
                        },
                    ch_names=eeg.ch_names, 
                    tmin=eeg.tmin,
                    verbose=eeg.verbose, #
                                )
            assert True
        except ValueError:
            assert False


        try:
            eegaux = eegdata(data=eeg.data, 
                    sfreq=eeg.sfreq, 
                    labels=eeg.labels, 
                    labels_dict=eeg.labels_dict,
                    events=eeg.events, 
                    ch_names={
                            "BCI": 1,
                            "Method": 2,
                            "year": 3,
                            "month": 4
                        }, 
                    tmin=eeg.tmin,
                    verbose=eeg.verbose, #
                                )
            assert False
        except ValueError:
            assert True

        try:
            eegaux = eegdata(data=eeg.data, 
                    sfreq=eeg.sfreq, 
                    labels=eeg.labels, 
                    labels_dict=eeg.labels_dict,
                    events=eeg.events, 
                    ch_names=2, 
                    tmin=eeg.tmin,
                    verbose=eeg.verbose, #
                                )
            assert False
        except ValueError:
            assert True

        try:
            eegaux = eegdata(data=eeg.data, 
                    sfreq=eeg.sfreq, 
                    labels=eeg.labels, 
                    labels_dict=eeg.labels_dict,
                    events=eeg.events, 
                    ch_names=np.random.randn(2,2), 
                    tmin=eeg.tmin,
                    verbose=eeg.verbose, #
                                )
            assert False
        except ValueError:
            assert True

        try:
            eegaux = eegdata(data=eeg.data, 
                    sfreq=eeg.sfreq, 
                    labels=eeg.labels, 
                    labels_dict=eeg.labels_dict,
                    events=eeg.events, 
                    ch_names=np.random.randn(2), 
                    tmin=eeg.tmin,
                    verbose=eeg.verbose, #
                                )
            assert False
        except ValueError:
            assert True


        try:
            eegaux = eegdata(data=eeg.data, 
                    sfreq=eeg.sfreq, 
                    labels=eeg.labels, 
                    labels_dict=eeg.labels_dict,
                    events=eeg.events, 
                    ch_names=eeg.ch_names, 
                    tmin=None,
                    timestamp=None,
                    verbose=eeg.verbose, #
                                )
            assert True
        except ValueError:
            assert False

        try:
            eegaux = eegdata(data=eeg.data, 
                    sfreq=eeg.sfreq, 
                    labels=eeg.labels, 
                    labels_dict=eeg.labels_dict,
                    events=eeg.events, 
                    ch_names=eeg.ch_names, 
                    tmin=eeg.tmin,
                    timestamp=eeg.timestamps,
                    verbose=eeg.verbose, #
                                )
            assert False
        except ValueError:
            assert True

        try:
            eegaux = eegdata(data=eeg.data, 
                    sfreq=eeg.sfreq, 
                    labels=eeg.labels, 
                    labels_dict=eeg.labels_dict,
                    events=eeg.events, 
                    ch_names=eeg.ch_names, 
                    timestamp=eeg.timestamps,
                    verbose=eeg.verbose, #
                                )
            assert True
        except ValueError:
            assert False
        try:
            eegaux = eegdata(data=eeg.data, 
                    sfreq=eeg.sfreq, 
                    labels=eeg.labels, 
                    labels_dict=eeg.labels_dict,
                    events=eeg.events, 
                    ch_names=eeg.ch_names, 
                    timestamp={
                            "BCI": 1,
                            "Method": 2,
                            "year": 3,
                            "month": 4
                        },
                    verbose=eeg.verbose, #
                                )
            assert False
        except ValueError:
            assert True

        try:
            eegaux = eegdata(data=eeg.data, 
                    sfreq=eeg.sfreq, 
                    labels=eeg.labels, 
                    labels_dict=eeg.labels_dict,
                    events=eeg.events, 
                    ch_names=eeg.ch_names, 
                    timestamp=np.random.randn(2,2),
                    verbose=eeg.verbose, #
                                )
            assert False
        except ValueError:
            assert True
        try:
            eegaux = eegdata(data=eeg.data, 
                    sfreq=eeg.sfreq, 
                    labels=eeg.labels, 
                    labels_dict=eeg.labels_dict,
                    events=eeg.events, 
                    ch_names=eeg.ch_names, 
                    timestamp=np.arange(2),
                    verbose=eeg.verbose, #
                                )
            assert False
        except ValueError:
            assert True
        try:
            eegaux = eegdata(data=eeg.data, 
                    sfreq=eeg.sfreq, 
                    labels=eeg.labels, 
                    labels_dict=eeg.labels_dict,
                    events=eeg.events, 
                    ch_names=eeg.ch_names, 
                    timestamp=np.arange(eeg.data.shape[3]),
                    verbose=eeg.verbose, #
                                )
            assert True
        except ValueError:
            assert False

        try:
            eegaux = eegdata(data=eeg.data, 
                    sfreq=eeg.sfreq, 
                    labels=eeg.labels, 
                    labels_dict=eeg.labels_dict,
                    events=eeg.events, 
                    ch_names=eeg.ch_names, 
                    tmin=np.arange(eeg.data.shape[3]),
                    verbose=eeg.verbose, #
                                )
            assert False
        except ValueError:
            assert True
        try:
            eegaux = eegdata(data=eeg.data, 
                    sfreq=eeg.sfreq, 
                    labels=eeg.labels, 
                    labels_dict=eeg.labels_dict,
                    events=eeg.events, 
                    ch_names=eeg.ch_names, 
                    tmin=3,
                    verbose=eeg.verbose, #
                                )
            assert True
        except ValueError:
            assert False

        dataUsed2 = np.random.randn(1,22, 1875)
        eegaux = eegdata(data=dataUsed2, 
                    sfreq=eeg.sfreq, 
                    labels=None, 
                    labels_dict=eeg.labels_dict,
                    events=eeg.events, 
                    ch_names=eeg.ch_names, 
                    tmin=eeg.tmin,
                    verbose=eeg.verbose, #
                                )
        try:
            eegaux = eegdata(data=dataUsed2, 
                    sfreq=eeg.sfreq, 
                    labels=None, 
                    labels_dict=eeg.labels_dict,
                    events=eeg.events, 
                    ch_names=eeg.ch_names, 
                    tmin=eeg.tmin,
                    verbose=eeg.verbose, #
                                )
            assert True
        except ValueError:
            assert False

        dataUsed2 = np.random.randn( 22, 1875)
        try:
            eegaux = eegdata(data=dataUsed2, 
                    sfreq=eeg.sfreq, 
                    labels=None, 
                    labels_dict=eeg.labels_dict,
                    events=eeg.events, 
                    ch_names=eeg.ch_names, 
                    tmin=eeg.tmin,
                    verbose=eeg.verbose, #
                                )
            assert True
        except ValueError:
            assert False

        dataUsed2 = np.random.randn(1875)
        try:
            eegaux = eegdata(data=dataUsed2, 
                    sfreq=eeg.sfreq, 
                    labels=None, 
                    labels_dict=eeg.labels_dict,
                    events=eeg.events, 
                    ch_names=None, 
                    tmin=eeg.tmin,
                    verbose=eeg.verbose, #
                                )
            assert True
        except ValueError:
            assert False
            

    def test_delete_eegdata(self,eegData2a):
        eeg = eegData2a.copy()
        resultOfDeletion = eeg.__delete__()
        print(type(eeg))
        assert resultOfDeletion == None and type(eeg) == eegdata
    
    def test_str_eegdata(self,eegData2a):
        eeg = eegData2a.copy()
        id = eeg.id
        string = f"eegdata<id={id}>"
        assert  string == eeg.__str__()

    def test_get_data(self,eegData2a):
        eeg = eegData2a
        eeg2 =eeg.get_data()
        eeg3 =eeg.get_data(inplace=True)
        assert type(eeg2) == eegdata and eeg3 == eegData2a
        try:
            eeg4 = eeg.get_data(idx = {
                            "BCI": "Project",
                            "Method": "Loreta",
                            "year": 2002
                        })
            assert False
        except ValueError:
            assert True
        createArray = np.array([1.2, 2.3, 3.4, 4.5, 5.0])
        try:
            
            eeg4 = eeg.get_data(idx = createArray)
            assert False
        except ValueError:
            assert True

    def test_id_setter(self,eegData2a):
        eeg = eegData2a.copy()
        value = eeg.id
        eeg.id = 404
        nvalue = eeg.id
        assert value == eeg.id and nvalue != 404

    def test_data_setter(self,eegData2a):
        eeg = eegData2a.copy()
        datares = eeg.data 
        createArray = np.array([1.2, 2.3, 3.4, 4.5, 5.0])
        try:
            eeg.data={
                            "BCI": "Project",
                            "Method": "Loreta",
                            "year": 2002
                        }
            assert False
        except ValueError:
            assert True
        assert np.array_equal(eeg.data, datares)

        try:
            eeg.data=np.array([1,2,3])
            assert False
        except ValueError:
            assert True
        assert np.array_equal(eeg.data, datares)

        try:
            createArray = np.random.randn(5, 3, 3, 2)
            if np.array_equal(createArray, datares):
                assert False
            eeg.data = createArray
            if np.array_equal(eeg.data, None):
                assert False 
            else:
                if np.array_equal(None, datares):
                    assert False
                else:
                    if np.array_equal(eeg.data, datares):
                        assert False
                    else:
                        assert True
        except ValueError:
            assert True

        eeg = eegData2a.copy()

        try:
            createArray = np.random.randn(5, 3, 3)
            if np.array_equal(createArray, datares):
                assert False
            eeg.data = createArray
            if np.array_equal(eeg.data, None):
                assert False 
            else:
                if np.array_equal(None, datares):
                    assert False
                else:
                    if np.array_equal(eeg.data, datares):
                        assert False
                    else:
                        assert True
        except ValueError:
            assert True

        eeg = eegData2a.copy()

        try:
            createArray = np.random.randn(5, 3)
            if np.array_equal(createArray, datares):
                assert False
            eeg.data = createArray
            if np.array_equal(eeg.data, None):
                assert False 
            else:
                if np.array_equal(None, datares):
                    assert False
                else:
                    if np.array_equal(eeg.data, datares):
                        assert False
                    else:
                        assert True
        except ValueError:
            assert True

        eeg = eegData2a.copy()

        try:
            createArray = np.random.randn(5)
            if np.array_equal(createArray, datares):
                assert False
            eeg.data = createArray
            if np.array_equal(eeg.data, None):
                assert False 
            else:
                if np.array_equal(None, datares):
                    assert False
                else:
                    if np.array_equal(eeg.data, datares):
                        assert False
                    else:
                        assert True
        except ValueError:
            assert True

        eeg = eegData2a.copy()

        try:
            createArray = np.random.randn(5, 3, 3, 2, 5)
            if np.array_equal(createArray, datares):
                assert False
            eeg.data = createArray
            if np.array_equal(eeg.data, None):
                assert False 
            else:
                if np.array_equal(None, datares):
                    assert False
                else:
                    if np.array_equal(eeg.data, datares):
                        assert False
                    else:
                        assert False
        except ValueError:
            assert True
        assert np.array_equal(eeg.data, datares)

        try:
            eeg.data=createArray
            if np.array_equal(createArray, datares):
                assert False
            else:
                if np.array_equal(eeg.data, None):
                    assert False 
                else:
                    if np.array_equal(None, datares):
                        assert False
                    else:
                        if np.array_equal(eeg.data, datares):
                            assert False
                        else:
                            assert True
        except ValueError:
            assert True

    def test_ntrials_setter(self,eegData2a):
        eeg = eegData2a.copy()
        value = eeg.n_trials
        eeg.n_trials = 404
        nvalue = eeg.n_trials
        assert value == eeg.n_trials and nvalue != 404

    def test_nbands_setter(self,eegData2a):
        eeg = eegData2a.copy()
        value = eeg.n_bands
        eeg.n_bands = 404
        nvalue = eeg.n_bands
        assert value == eeg.n_bands and nvalue != 404

    def test_nchannels_setter(self,eegData2a):
        eeg = eegData2a.copy()
        value = eeg.n_channels
        eeg.n_channels = 404
        nvalue = eeg.n_channels
        assert value == eeg.n_channels and nvalue != 404

    def test_ntimes_setter(self,eegData2a):
        eeg = eegData2a.copy()
        value = eeg.n_times
        eeg.n_times = 404
        nvalue = eeg.n_times
        assert value == eeg.n_times and nvalue != 404

    def test_sfreq_setter(self,eegData2a):
        eeg = eegData2a.copy()
        value = eeg.sfreq
        eeg.sfreq = 404
        nvalue = eeg.sfreq
        assert value == eeg.sfreq and nvalue != 404

    def test_timestamps_setter(self,eegData2a):
        eeg = eegData2a.copy()
        value = eeg.timestamps
        createArray = np.array([1.2, 2.3, 3.4, 4.5, 5.0])
        eeg.timestamps = createArray
        nArray = eeg.timestamps
        assert np.array_equal(value, eeg.timestamps) and not np.array_equal(nArray, createArray)

    def test_tmin_setter(self,eegData2a):
        eeg = eegData2a.copy()
        value = eeg.tmin
        eeg.tmin = 404
        nvalue = eeg.tmin
        assert value == eeg.tmin and nvalue != 404

    def test_events_setter(self,eegData2a):
        eeg = eegData2a.copy()
        actualDict = eeg.events
        createDict = {
                        "BCI": "Project",
                        "Method": "Loreta",
                        "year": 2002
                    }
        try:
            eeg.events = createDict
            if eeg.events == actualDict:
                assert False
            else:
                assert True
        except ValueError:
            assert False
        eeg = eegData2a.copy()    
        try:
            eeg.events = np.array([1.2, 2.3, 3.4, 4.5, 5.0])
            assert False
        except ValueError:
            assert True
       
        assert actualDict == eeg.events

    def test_labels_setter(self,eegData2a):
        eeg = eegData2a.copy()
        labelres = eeg.labels
        createArray = None
        try:
            createArray = 2
            eeg.labels = createArray
            assert False
        except ValueError:
            assert True

        assert np.array_equal(labelres, eeg.labels) 

        eeg = eegData2a.copy()

        try:
            createArray = np.array([[1.2, 2.3, 3.4, 4.5, 5.0],[1.2, 2.3, 3.4, 4.5, 5.0]])
            eeg.labels = createArray
            assert False
        except ValueError:
            assert True  
        assert np.array_equal(labelres, eeg.labels)  

        eeg = eegData2a.copy() 
        
        try:
            createArray = np.array([1.2, 2.3, 3.4, 4.5, 5.0])
            eeg.labels = createArray
            assert False
        except ValueError:
            assert True   

        assert np.array_equal(labelres, eeg.labels) 

    def test_labelsDict_setter(self,eegData2a):
        eeg = eegData2a
        value = eeg.labels_dict
        createDict = {"BCI": "Project"}
        try:
            eeg.labels_dict = None
            assert False
        except ValueError:
            assert True
        assert value == eeg.labels_dict
        try:
            eeg.labels_dict = createDict
            assert False
        except ValueError:
            assert True
        assert value == eeg.labels_dict
        try:
            eeg.labels_dict = 2
            assert False
        except ValueError:
            assert True
        assert value == eeg.labels_dict

    def test_chNames_setter(self,eegData2a):
        eeg = eegData2a.copy()
        value = eeg.ch_names
        try:
            createArray = np.random.randn(5, 3, 3, 2, 5)
            eeg.ch_names = createArray
            assert False
        except ValueError:
            assert True
        assert np.array_equal(value, eeg.ch_names) 
        try:
            createArray = np.random.randn(2)
            eeg.ch_names = createArray
            assert False
        except ValueError:
            assert True
        assert np.array_equal(value, eeg.ch_names)   

    def test_verbose_setter(self,eegData2a):
        eeg = eegData2a.copy()
        value = eeg.verbose
        eeg.verbose = "404"
        assert value == eeg.verbose
        
    def test_concatenate(self,eegData2a): #concatena o 2a pra testar, verificar o tamanho dos trials 
        eeg = eegData2a.copy()
        # eeg = eegdata.concatenate(eeg)
        createArray = np.random.randn(2)
        try: 
            eeg = eegdata.concatenate(createArray)
            assert False
        except ValueError:
            assert True

        eeg = eegData2a.copy()
        createArray = 2
        try:
            eeg = eegdata.concatenate(createArray)
            assert False
        except ValueError:
            assert True

        eeg = eegData2a.copy()
        createArray = list(np.array([],dtype=eegdata))
        try:
            eeg = eegdata.concatenate(createArray)
            assert False
        except ValueError:
            assert True

        eeg = eegData2a.copy()
        createArray = [eeg, 2, "str"]
        try:
            eeg = eegdata.concatenate(createArray)
            assert False
        except ValueError:
            assert True

        eeg = eegData2a.copy()
        createCh_names = ["CH_%d"%i for i in range(eeg.data.shape[2])]
        eegaux = eegdata(data=eeg.data, 
                   sfreq=eeg.sfreq, 
                   labels=eeg.labels, 
                   labels_dict=eeg.labels_dict,
                   events=eeg.events, 
                   ch_names=createCh_names, 
                   tmin=eeg.tmin,
                   verbose=eeg.verbose, #
                            )
        try:
            eeg = eegdata.concatenate([eeg,eegaux])
            assert False
        except ValueError:
            assert True
        
        eeg = eegData2a.copy()
        eegaux = eegdata(data=eeg.data, 
                   sfreq=10, 
                   labels=eeg.labels, 
                   labels_dict=eeg.labels_dict,
                   events=eeg.events, 
                   ch_names=eeg.ch_names, 
                   tmin=eeg.tmin,
                   verbose=eeg.verbose, #
                            )
        try:  
            eeg = eegdata.concatenate([eeg,eegaux])
            assert False
        except ValueError:
            assert True
        
        eeg = eegData2a.copy()
        dataUsed = eeg.data[:,:,[0,1,2],:]
        eegaux = eegdata(data=dataUsed, 
                   sfreq=eeg.sfreq, 
                   labels=eeg.labels, 
                   labels_dict=eeg.labels_dict,
                   events=eeg.events, 
                   ch_names=eeg.ch_names[:3], 
                   tmin=eeg.tmin,
                   verbose=eeg.verbose, #
                            )
        try:
            eeg = eegdata.concatenate([eeg,eegaux])
            assert False
        except ValueError:
            assert True
        
        eeg = eegData2a.copy()
        dataUsed = eeg.data[:,[0,0],:,:]
        eegaux = eegdata(data=dataUsed, 
                   sfreq=eeg.sfreq, 
                   labels=eeg.labels, 
                   labels_dict=eeg.labels_dict,
                   events=eeg.events, 
                   ch_names=eeg.ch_names, 
                   tmin=eeg.tmin,
                   verbose=eeg.verbose, #
                            ) 
        try:
            eeg = eegdata.concatenate([eeg,eegaux])
            assert False
        except ValueError:
            assert True

        eeg = eegData2a.copy()
        eventsUsed = {"BCI": [0.1]}
        eegaux = eegdata(data=eeg.data, 
                   sfreq=eeg.sfreq, 
                   labels=eeg.labels, 
                   labels_dict=eeg.labels_dict,
                   events=eventsUsed, 
                   ch_names=eeg.ch_names, 
                   tmin=eeg.tmin,
                   verbose=eeg.verbose, 
                            ) 
        try:
            eeg = eegdata.concatenate([eeg,eegaux])
            assert False
        except ValueError:
            assert True

        eeg = eegData2a.copy()

        eeg1 = eegdata.concatenate(eeg)

        assert eeg.id != eeg1.id
        assert np.array_equal(eeg1.data, eeg.data)
        assert eeg.sfreq == eeg1.sfreq
        assert eeg.n_trials == eeg1.n_trials
        assert eeg.n_bands == eeg1.n_bands
        assert eeg.n_channels == eeg1.n_channels
        assert eeg.n_times == eeg1.n_times
        assert eeg.tmin == eeg1.tmin
        assert np.array_equal(eeg1.timestamps, eeg.timestamps)
        assert eeg.events == eeg1.events
        # assert np.array_equal(eeg1.labels, eeg.labels)
        # assert eeg.labels_dict == eeg1.labels_dict  ### ver com o gabriel essa parada daqui
        assert np.array_equal(eeg1.ch_names, eeg.ch_names)
        assert eeg.verbose == eeg1.verbose

        eeg2 = eegdata.concatenate([eeg,eeg])

        assert eeg2.data.shape[0] == 2*eeg.data.shape[0]

    def test_copy_eegdata(self,eegData2a):
        eeg = eegData2a
        eeg1 = eegData2a.copy()
        eeg2 = eegData2a.copy()

        assert eeg2.id != eeg1.id
        assert np.array_equal(eeg1.data, eeg2.data)
        assert eeg2.sfreq == eeg1.sfreq
        assert eeg2.n_trials == eeg1.n_trials
        assert eeg2.n_bands == eeg1.n_bands
        assert eeg2.n_channels == eeg1.n_channels
        assert eeg2.n_times == eeg1.n_times
        assert eeg2.tmin == eeg1.tmin
        assert np.array_equal(eeg1.timestamps, eeg2.timestamps)
        assert eeg2.events == eeg1.events
        assert np.array_equal(eeg1.labels, eeg2.labels)
        assert eeg2.labels_dict == eeg1.labels_dict 
        assert np.array_equal(eeg1.ch_names, eeg2.ch_names)
        assert eeg2.verbose == eeg1.verbose

        assert eeg2.id != eeg.id
        assert np.array_equal(eeg.data, eeg2.data)
        assert eeg2.sfreq == eeg.sfreq
        assert eeg2.n_trials == eeg.n_trials
        assert eeg2.n_bands == eeg.n_bands
        assert eeg2.n_channels == eeg.n_channels
        assert eeg2.n_times == eeg.n_times
        assert eeg2.tmin == eeg.tmin
        assert np.array_equal(eeg.timestamps, eeg2.timestamps)
        assert eeg2.events == eeg.events
        assert np.array_equal(eeg.labels, eeg2.labels)
        assert eeg2.labels_dict == eeg.labels_dict 
        assert np.array_equal(eeg.ch_names, eeg2.ch_names)
        assert eeg2.verbose == eeg.verbose

        
        

    def test_append_eegdata(self,eegData2a):
        eeg = eegData2a.copy()
        try:
            eeg.append(2)
            assert False
        except ValueError:
            assert True
        try:
            eeg.append({"BCI": [0.1]})
            assert False
        except ValueError:
            assert True
        try:
            eeg.append(eeg.data)
            assert False
        except ValueError:
            assert True

        eegaux = eegdata(data=eeg.data, 
                   sfreq=10, 
                   labels=eeg.labels, 
                   labels_dict=eeg.labels_dict,
                   events=eeg.events, 
                   ch_names=eeg.ch_names, 
                   tmin=eeg.tmin,
                   verbose=eeg.verbose, 
                            ) 
        try:
            eeg.append(eegaux)
            assert False
        except ValueError:
            assert True
        eeg = eegData2a.copy()
        dataUsed = eeg.data[:,:,[0,1,2],:]
        eegaux = eegdata(data=dataUsed, 
                   sfreq=eeg.sfreq, 
                   labels=eeg.labels, 
                   labels_dict=eeg.labels_dict,
                   events=eeg.events, 
                   ch_names=eeg.ch_names[:3], 
                   tmin=eeg.tmin,
                   verbose=eeg.verbose, #
                            )
        try:
            eeg.append(eegaux)
            assert False
        except ValueError:
            assert True
        eeg = eegData2a.copy()
        dataUsed = eeg.data[:,[0,0],:,:]
        eegaux = eegdata(data=dataUsed, 
                   sfreq=eeg.sfreq, 
                   labels=eeg.labels, 
                   labels_dict=eeg.labels_dict,
                   events=eeg.events, 
                   ch_names=eeg.ch_names, 
                   tmin=eeg.tmin,
                   verbose=eeg.verbose, #
                            ) 
        try:
            eeg.append(eegaux)
            assert False
        except ValueError:
            assert True

        try:
            eeg.append(eeg)
            assert True
        except ValueError:
            assert False
        #merma coisa que o  concatenate só que com objeto antes
            

    def test_shuffle_eegdata(self,eegData2a):
        eeg = eegData2a.copy()
        eeg1 = eegData2a.copy()
        eeg1.shuffle()
        assert eeg1.data.shape[0] == eeg.data.shape[0]
        assert not np.array_equal(eeg.labels, eeg1.labels)
        assert eeg1.labels_dict == eeg.labels_dict
        keys = list(eeg.labels_dict.keys())
        assert (eeg.labels == eeg.labels_dict[keys[0]]).sum() == (eeg1.labels == eeg1.labels_dict[keys[0]]).sum() 
        assert (eeg.labels == eeg.labels_dict[keys[1]]).sum() == (eeg1.labels == eeg1.labels_dict[keys[1]]).sum()
        assert (eeg.labels == eeg.labels_dict[keys[2]]).sum() == (eeg1.labels == eeg1.labels_dict[keys[2]]).sum()
        assert (eeg.labels == eeg.labels_dict[keys[3]]).sum() == (eeg1.labels == eeg1.labels_dict[keys[3]]).sum() 

    def test_crop_eegdata(self,eegData2a):
        eeg = eegData2a.copy()
        time = [0,0.5]
        timeCropedEEG = None
        #falta tratar o caso do tmin == None
        # timeCropedEEG = eeg.crop(tmin = None, window_size = 2.5, inplace = False )
        # assert timeCropedEEG.timestamps.shape == 0
        # tenho que pegar o tempo do timestamps fazer um corte nele para um intervalo de tempo, pegar esse intervalo manualmente, compara esse resultado com o da func crop
        eegaux = eegdata.concatenate([eeg,eeg])
        timeCropedEEG = eeg.crop(tmin = time, window_size = 2.5
                                 , inplace = False )
        assert timeCropedEEG.data.shape[0] == eeg.data.shape[0] * 2
        assert timeCropedEEG.data.shape[1] == eeg.data.shape[1] 
        assert timeCropedEEG.data.shape[2] == eeg.data.shape[2]
        assert timeCropedEEG.data.shape[3] == eeg.data.shape[3] / 3
        # assert timeCropedEEG.labels.shape == eegaux.labels.shape
        # assert  np.array_equal(timeCropedEEG.labels, eegaux.labels)
        # assert eegaux.timestamps.shape == 0
        assert  np.array_equal(timeCropedEEG.timestamps.shape[0], eegaux.timestamps.shape[0]/3)
        timestampOrigin = eegaux.timestamps[0:timeCropedEEG.timestamps.shape[0]]
        timestampOrigin = np.unique(timestampOrigin)    
        timestampNow = np.unique(timeCropedEEG.timestamps)  
        assert  len(timestampNow) == len(timestampOrigin) 
        assert  np.allclose(timestampNow, timestampOrigin) 
        assert  np.allclose(timeCropedEEG.timestamps, eegaux.timestamps[0:timeCropedEEG.timestamps.shape[0]])
        

        # assert timeCropedEEG.labels == 0


        timeCropedEEG = eeg.crop(tmin = 0, window_size = 2.5
                                 , inplace = False )
        assert timeCropedEEG.data.shape[0] == eeg.data.shape[0] 
        assert timeCropedEEG.data.shape[1] == eeg.data.shape[1] 
        assert timeCropedEEG.data.shape[2] == eeg.data.shape[2]
        assert timeCropedEEG.data.shape[3] == eeg.data.shape[3] / 3
        assert  np.array_equal(timeCropedEEG.timestamps.shape[0], eegaux.timestamps.shape[0]/3)
        timestampOrigin = eegaux.timestamps[0:timeCropedEEG.timestamps.shape[0]]
        timestampOrigin = np.unique(timestampOrigin)    
        timestampNow = np.unique(timeCropedEEG.timestamps)  
        assert  len(timestampNow) == len(timestampOrigin) 
        assert  np.allclose(timestampNow, timestampOrigin) 
        assert  np.allclose(timeCropedEEG.timestamps, eegaux.timestamps[0:timeCropedEEG.timestamps.shape[0]])

        timeCropedEEG = eeg.crop(tmin = np.array(time), window_size = 2.5
                                 , inplace = False )
        assert timeCropedEEG.data.shape[0] == eeg.data.shape[0] *2
        assert timeCropedEEG.data.shape[1] == eeg.data.shape[1] 
        assert timeCropedEEG.data.shape[2] == eeg.data.shape[2]
        assert timeCropedEEG.data.shape[3] == eeg.data.shape[3] / 3
        assert  np.array_equal(timeCropedEEG.timestamps.shape[0], eegaux.timestamps.shape[0]/3)
        timestampOrigin = eegaux.timestamps[0:timeCropedEEG.timestamps.shape[0]]
        timestampOrigin = np.unique(timestampOrigin)    
        timestampNow = np.unique(timeCropedEEG.timestamps)  
        assert  len(timestampNow) == len(timestampOrigin) 
        assert  np.allclose(timestampNow, timestampOrigin) 
        assert  np.allclose(timeCropedEEG.timestamps, eegaux.timestamps[0:timeCropedEEG.timestamps.shape[0]])

        try:
            timeCropedEEG = eeg.crop(tmin = {"BCI": 1}, window_size = 2.5
                                 , inplace = False )
            assert False
        except ValueError:
            assert True

        try:
            timeCropedEEG = eeg.crop(tmin = np.random.randn(2,2), window_size = 2.5
                                 , inplace = False )
            assert False
        except ValueError:
            assert True

        try:
            timeCropedEEG = eeg.crop(tmin = 0, window_size = np.array(2.5)
                                 , inplace = False )
            assert False
        except ValueError:
            assert True
        
        try:
            timeCropedEEG = eeg.crop(tmin = 0, window_size = -2.5
                                 , inplace = False )
            assert False
        except ValueError:
            assert True

        try:
            timeCropedEEG = eeg.crop(tmin = 0, window_size = 10
                                 , inplace = False )
            assert False
        except ValueError:
            assert True

        try:
            timeCropedEEG = eeg.crop(tmin = 5, window_size = 3
                                 , inplace = False )
            assert False
        except ValueError:
            assert True
        



        
    def test_apply_to_trial(self,eegData2a):
        eeg = eegData2a.copy()
        eeg1 = eegData2a.copy()
        def x2 (x, **kwargs):
            return 2*x
        attributesNow = ['sfreq', 'tmin', 'timestamps', 'events', 'ch_names']
        eeg1.apply_to_trial(func=x2,attributes=attributesNow)
        assert np.allclose(eeg1.data[2] , eeg.data[2]*2)
        assert np.allclose(eeg1.data[0,:,:,:] , eeg.data[0,:,:,:]*2)
        assert eeg1.sfreq == eeg.sfreq
        assert eeg1.tmin == eeg.tmin
        assert np.allclose(eeg1.timestamps, eeg.timestamps)

        eeg1 = eegData2a.copy()
        eeg1.apply_to_trial(func=x2,attributes=['sfreq'])
        assert np.allclose(eeg1.data[2] , eeg.data[2]*2)
        assert np.allclose(eeg1.data[0,:,:,:] , eeg.data[0,:,:,:]*2)
        assert eeg1.sfreq == eeg.sfreq
        assert eeg1.tmin == eeg.tmin
        assert np.allclose(eeg1.timestamps, eeg.timestamps)

        eeg1 = eegData2a.copy()
        eeg1.apply_to_trial(func=x2)
        assert np.allclose(eeg1.data[2] , eeg.data[2]*2)
        assert np.allclose(eeg1.data[0,:,:,:] , eeg.data[0,:,:,:]*2)
        assert eeg1.sfreq == eeg.sfreq
        assert eeg1.tmin == eeg.tmin
        assert np.allclose(eeg1.timestamps, eeg.timestamps)
        
        try:
            eeg1.apply_to_trial(func=x2,attributes=['ola'])
            assert False
        except ValueError:
            assert True
        eeg1.apply_to_trial(func=x2,attributes={"sfreq":"tmin"})# era pra isso passar?
        try:
            eeg1.apply_to_trial(func=x2,attributes={"oi":"sfreq"})
            assert False
        except ValueError:
            assert True
        #criar uma função e aplicar ela no apply to trial, pois ele aplica a função passada nos dados.
        #função que transforma trials, tipo pega um trials e multiplica ele por 2
        # tipo x = banda( o 1 do data.shape do BCICIVa2),eletrodo,tempo e o resultado é mesmo shape de eletrodo,tempo porém valor só multiplicou por 2 todos os valores dentro da matriz
        pass

    def test_apply_to_data(self,eegData2a):
        eeg = eegData2a.copy()
        eeg1 = eegData2a.copy()
        def x2 (x, **kwargs):
            return 2*x
        attributesNow = ['labels', 'sfreq', 'tmin', 'timestamps', 'events', 'ch_names']
        eeg1.apply_to_data(func=x2,attributes=attributesNow)
        assert np.allclose(eeg1.data[2] , eeg.data[2]*2)
        assert np.allclose(eeg1.data[:,:,:,:] , eeg.data[:,:,:,:]*2)
        assert eeg1.sfreq == eeg.sfreq
        assert eeg1.tmin == eeg.tmin
        assert np.allclose(eeg1.timestamps, eeg.timestamps)
        assert eeg1.data.shape == eeg.data.shape
        assert len(eeg1.labels) == len(eeg.labels)
        assert np.array_equal(eeg1.labels, eeg.labels)

        eeg1 = eegData2a.copy()
        eeg1.apply_to_data(func=x2,attributes=['sfreq'])
        assert np.allclose(eeg1.data[2] , eeg.data[2]*2)
        assert np.allclose(eeg1.data[0,:,:,:] , eeg.data[0,:,:,:]*2)
        assert eeg1.sfreq == eeg.sfreq
        assert eeg1.tmin == eeg.tmin
        assert np.allclose(eeg1.timestamps, eeg.timestamps)
        assert eeg1.data.shape == eeg.data.shape
        assert len(eeg1.labels) == len(eeg.labels)
        assert np.array_equal(eeg1.labels, eeg.labels)

        eeg1 = eegData2a.copy()
        eeg1.apply_to_data(func=x2)
        assert np.allclose(eeg1.data[2] , eeg.data[2]*2)
        assert np.allclose(eeg1.data[:,:,:,:] , eeg.data[:,:,:,:]*2)
        assert eeg1.sfreq == eeg.sfreq
        assert eeg1.tmin == eeg.tmin
        assert np.allclose(eeg1.timestamps, eeg.timestamps)
        assert eeg1.data.shape == eeg.data.shape
        assert len(eeg1.labels) == len(eeg.labels)
        assert np.array_equal(eeg1.labels, eeg.labels)

        try:
            eeg1.apply_to_data(func=x2,attributes=['ola'])
            assert False
        except ValueError:
            assert True
        eeg1.apply_to_data(func=x2,attributes={"sfreq":"tmin"})# era pra isso passar?
        try:
            eeg1.apply_to_data(func=x2,attributes={"oi":"sfreq"})
            assert False
        except ValueError:
            assert True


        eeg1 = eegData2a.copy()
        eeg1.apply_to_data(func=x2)
        eeg2 = eegData2a.copy()
        eeg2.apply_to_trial(func=x2)
        #era pra dar igual?
        assert np.allclose(eeg1.data[2] , eeg2.data[2])
        assert np.allclose(eeg1.data[:,:,:,:] , eeg2.data[:,:,:,:])
        assert eeg1.sfreq == eeg2.sfreq
        assert eeg1.tmin == eeg2.tmin
        assert np.allclose(eeg1.timestamps, eeg2.timestamps)
        assert eeg1.data.shape == eeg2.data.shape
        assert len(eeg1.labels) == len(eeg2.labels)
        assert np.array_equal(eeg1.labels, eeg2.labels)
        assert np.allclose(eeg1.data , eeg2.data)
        #criar uma função e aplicar ela no apply to data, pois ele aplica a função passada nos dados.
        #função que transforma trials, tipo pega os trials e multiplica eles por 2
        # tipo x = trials(o referente do labels basicamente),banda( o 1 do data.shape do BCICIVa2),eletrodo,tempo e o resultado é mesmo shape de eletrodo,tempo porém valor só multiplicou por 2 todos os valores dentro da matriz
        