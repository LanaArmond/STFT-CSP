import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)

from modules.tf.stft import STFT_eegdata

from modules.fe.logpower import logpower
from modules.fs.mibif import MIBIF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

clf = LinearDiscriminantAnalysis()

tf = STFT_eegdata()

fe = logpower(flating=True)
fs = MIBIF(8, paired=False, clf=clf)

pre_folding = {'tf': (tf, {'attributes': ['sfreq']}),
               }
pos_folding = {
               'fe': (fe, {}),
               'fs': (fs, {}),
               'clf': (clf, {})}
