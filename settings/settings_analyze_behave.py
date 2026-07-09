import numpy as np
from behave_analysis.utils.settings_objects import Settings_analyze_behave as Settings

settings_ab = Settings(
    escape_stim_type="audio",
    show_plots=False,
    homings_use_boris=False,
    redo_compute=False,
    homings_classifiction_manual_gates={'speed_peak': {'dir': '>=', 'threshold': 20.0}, 
                 'speed_mean': {'dir': '>=', 'threshold': 10.0}, 
                 'net_distance': {'dir': '>=', 'threshold': 25.0}, 
                 'net_dy': {'dir': '>=', 'threshold': 10.0}, 
                 'displacement_vertical_ratio': {'dir': '>=', 'threshold': 0.3},
                 'speed_variance': {'dir': '>=', 'threshold': 45.0}}
)
