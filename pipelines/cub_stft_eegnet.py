import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)

from modules.tf.stft import STFT_eegdata
from modules.utils.to_numpy import to_numpy
from modules.clf.mbeegnet import MBEEGNet
from modules.tf.resample_cubic import cubic_resample_eegdata

clf = MBEEGNet()

tf = cubic_resample_eegdata()
tf2 = STFT_eegdata()
transform = to_numpy()

pre_folding = {'tf': (tf, {'attributes': ['sfreq'], 'new_sfreq': 128}),"tf2": (tf2, {'attributes': ['sfreq']})}
pos_folding = {'to_np': (transform, {}),
               'clf': (clf, {})}
