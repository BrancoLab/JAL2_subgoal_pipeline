"""
This file contains the settings for the process.py script. Any global changes to the way that process is run should be made here.
"""

# Custom libaries
from behave_analysis.utils.settings_objects import Settings_process

# Os Libraries
import os

settings_process = Settings_process(
    create_new_registration=False, # if True forces to redo the clicking registration
    registration="homography",  # 'affine' 'partial affine' or 'homography' (use the least complex needed)
    fisheye_correction_file=os.path.join("sample_data", "fisheye_maps.npy"),  # remove setting if n/a
    size=(1024, 1024),  # (width, height) how big to make the renderings, in pixels
    pixels_per_cm=10,  # for the arena drawn in register.generate_rendered_arena, report here the ratio between size of arena in pixels and actual size in cm
    efizz=True,  # Are you running the pipeline with efizz data or just behaviour? False if just behaviour
    remove_duplicate_spike_times=True, # if True, will remove duplicate spike times from the efizz data in window settings.duplicate_spikes_censored_period_ms
    radius = 460, # radius of the arena in pixels
)
