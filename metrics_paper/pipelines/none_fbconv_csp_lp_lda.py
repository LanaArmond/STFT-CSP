import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)

from modules.tf.filterbank import filterbank_eegdata
from modules.sf.csp import csp_eegdata
from modules.fe.logpower import logpower
from modules.fs.mibif import MIBIF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

clf = LinearDiscriminantAnalysis()

tf = filterbank_eegdata()
sf = csp_eegdata()
fe = logpower(flating=True)
fs = MIBIF(8, paired=False, clf=clf)

pre_folding = {'tf': (tf, {'attributes': ['sfreq']}),
               }
pos_folding = {'sf': (sf, {'attributes': ['labels']}),
               'fe': (fe, {}),
               'fs': (fs, {}),
               'clf': (clf, {})}
