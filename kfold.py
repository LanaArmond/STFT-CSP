from modules.core.eegdata import set_global_verbose_eegdata
set_global_verbose_eegdata("WARNING")
from modules.core.kfold import kfold

import numpy as np
import json
import importlib.util

# get string as input argument
import sys
import os
if len(sys.argv) < 2:
    print("Usage: python load_jsonexec.py <config_file>")
    sys.exit(1)

config_file = sys.argv[1]
if not os.path.exists(config_file):
    print(f"File {config_file} not found")
    sys.exit(1)

with open(config_file, 'r') as f:
    data = json.load(f)

spec = importlib.util.spec_from_file_location(data['pipeline'], data['pipeline'])
pipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipe)

pre_folding, pos_folding = pipe.pre_folding, pipe.pos_folding

spec = importlib.util.spec_from_file_location(data['dataset']['name'], data['dataset']['name'])
datasets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(datasets)

target = datasets.loading(subject=data['dataset']['subject'], 
                          session_list=data['dataset']['session_list'], 
                          run_list=data['dataset']['run_list'], 
                          events_dict=data['dataset']['events_dict'], 
                          verbose='INFO')

n_splits = data['n_splits']
window_size = data['window_size']
train_window = data['train_window']
t_min, t_max = data['test_window']['t_min'], data['test_window']['t_max']
test_window = list(np.round(np.arange(t_min, t_max-window_size+data['test_window']['step'], data['test_window']['step']), 1))


kfold_ = kfold(n_splits=n_splits, shuffle=True, pre_folding=pre_folding, pos_folding=pos_folding, window_size=window_size)
results = kfold_.exec(target, train_window=train_window, test_window=test_window)
results['tmin'] = results['tmin']

results.to_csv(config_file[:-5]+'.output', index=False, columns=['fold', 'tmin', 'true_label', *target.labels_dict.keys()])