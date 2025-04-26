import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)


from modules.sf.ea import ea_eegdata
from modules.fe.logpower import logpower
from modules.fs.mibif import MIBIF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

clf = LinearDiscriminantAnalysis()


sf = ea_eegdata()
fe = logpower(flating=True)
fs = MIBIF(8, paired=False, clf=clf)

pre_folding = {
               }
pos_folding = {'sf': (sf, {}),
               'fe': (fe, {}),
               'fs': (fs, {}),
               'clf': (clf, {})}
