import numpy as np
from behave_analysis.utils.settings_objects import Settings_analyze_behave as Settings

settings_ab = Settings(
    escape_stim_type="audio",
    show_plots=False,
    homings_use_boris=False,
    redo_compute=True,
    homings_classifiction_manual_gates={'speed_peak': {'dir': '>=', 'threshold': 15.0}, 
                 'speed_mean': {'dir': '>=', 'threshold': 10.0}, 
                 'net_distance': {'dir': '>=', 'threshold': 25.0}, # not sure why this is in cm... in the training data it's in pix
                 'net_dy': {'dir': '>=', 'threshold': 10.0}, 
                 'displacement_vertical_ratio': {'dir': '>=', 'threshold': 0.3},
                 'speed_variance': {'dir': '>=', 'threshold': 30.0},
                 'initial_hdir_change_abs': {'dir': '>=', 'threshold': 0.3},
                #  'hdir_variance': {'dir': '>=', 'threshold': 0.08},
                 }
)
