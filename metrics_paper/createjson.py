import json

data = {
	"pipeline": "",
	"dataset": {
		"name": "datasets\\PhysionetMI.py",
		"subject": None, 
		"session_list": None,
		"run_list": None, 
		"events_dict": {
			"left-hand": 1, 
			"right-hand": 2
		}
	},
	"n_splits": 5,
	"window_size": 2.0,
	"train_window": [0.5],
	"test_window": {
		"t_min": 0,
		"t_max": 4,
		"step": 0.1
	}
}

pipeline = ['metrics_paper\\pipelines\\none_cheb_csp_lp_lda.py',        # 1
            'metrics_paper\\pipelines\\none_conv_csp_lp_lda.py',
            'metrics_paper\\pipelines\\none_emd_csp_lp_lda.py',
            'metrics_paper\\pipelines\\none_fbcheb_csp_lp_lda.py', 
            'metrics_paper\\pipelines\\none_fbconv_csp_lp_lda.py',
            'metrics_paper\\pipelines\\none_none_csp_lp_lda.py',
            'metrics_paper\\pipelines\\none_stft_csp_lp_lda.py', 
            'metrics_paper\\pipelines\\none_wt_csp_lp_lda.py',            

            'metrics_paper\\pipelines\\none_cheb_none_lp_lda.py',       # 9
            'metrics_paper\\pipelines\\none_conv_none_lp_lda.py', 
            'metrics_paper\\pipelines\\none_emd_none_lp_lda.py',
            'metrics_paper\\pipelines\\none_fbcheb_none_lp_lda.py', 
            'metrics_paper\\pipelines\\none_fbconv_none_lp_lda.py',
            'metrics_paper\\pipelines\\none_none_none_lp_lda.py',
            'metrics_paper\\pipelines\\none_stft_none_lp_lda.py', 
            'metrics_paper\\pipelines\\none_wt_none_lp_lda.py',

            'metrics_paper\\pipelines\\none_cheb_ea_lp_lda.py',         # 17
            'metrics_paper\\pipelines\\none_conv_ea_lp_lda.py', 
            'metrics_paper\\pipelines\\none_emd_ea_lp_lda.py',
            'metrics_paper\\pipelines\\none_fbcheb_ea_lp_lda.py',      
            'metrics_paper\\pipelines\\none_fbconv_ea_lp_lda.py',
            'metrics_paper\\pipelines\\none_none_ea_lp_lda.py',
            'metrics_paper\\pipelines\\none_stft_ea_lp_lda.py', 
            'metrics_paper\\pipelines\\none_wt_ea_lp_lda.py',

            'metrics_paper\\pipelines\\cub_cheb_eegnet.py',             # 25
            'metrics_paper\\pipelines\\cub_conv_eegnet.py', 
            'metrics_paper\\pipelines\\cub_emd_eegnet.py',
            'metrics_paper\\pipelines\\cub_fbcheb_eegnet.py', 
            'metrics_paper\\pipelines\\cub_fbconv_eegnet.py', 
            'metrics_paper\\pipelines\\cub_none_eegnet.py',
            'metrics_paper\\pipelines\\cub_stft_eegnet.py',
            'metrics_paper\\pipelines\\cub_wt_eegnet.py',

            'metrics_paper\\pipelines\\fft_cheb_eegnet.py',             # 33
            'metrics_paper\\pipelines\\fft_conv_eegnet.py', 
            'metrics_paper\\pipelines\\fft_emd_eegnet.py',
            'metrics_paper\\pipelines\\fft_fbcheb_eegnet.py', 
            'metrics_paper\\pipelines\\fft_fbconv_eegnet.py', 
            'metrics_paper\\pipelines\\fft_none_eegnet.py',
            'metrics_paper\\pipelines\\fft_stft_eegnet.py', 
            'metrics_paper\\pipelines\\fft_wt_eegnet.py',

            'metrics_paper\\pipelines\\none_cheb_eegnet.py',            # 41
            'metrics_paper\\pipelines\\none_conv_eegnet.py', 
            'metrics_paper\\pipelines\\none_emd_eegnet.py',
            'metrics_paper\\pipelines\\none_fbcheb_eegnet.py', 
            'metrics_paper\\pipelines\\none_fbconv_eegnet.py', 
            'metrics_paper\\pipelines\\none_none_eegnet.py',
            'metrics_paper\\pipelines\\none_stft_eegnet.py', 
            'metrics_paper\\pipelines\\none_wt_eegnet.py',
            ]


for pipe in range(len(pipeline)):
    data['pipeline'] = pipeline[pipe]
    for sub in range(109):
        data['dataset']['subject'] = sub+1
        with open(f"config_phy_{pipe+1}_{sub+1}.json", 'w') as f:
            json.dump(data, f)