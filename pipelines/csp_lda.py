import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)

from modules.tf.bandpass.chebyshevII import chebyshevII_eegdata
from modules.sf.csp import csp_eegdata
from modules.fe.logpower import logpower
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

clf = LinearDiscriminantAnalysis()

tf = chebyshevII_eegdata()
sf = csp_eegdata()
fe = logpower(flating=True)

pre_folding = {'tf': (tf, {'attributes': ['sfreq']}),
               }
pos_folding = {'sf': (sf, {'attributes': ['labels']}),
               'fe': (fe, {}),
               'clf': (clf, {})}
