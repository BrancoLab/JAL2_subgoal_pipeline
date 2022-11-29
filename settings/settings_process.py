"""Import a dataclass object from settings object for the settings process and define it"""

#Custom libaries
from behave_analysis.utils.settings_objects import Settings_process

#Os Libraries
import os

settings_process = Settings_process(

    create_new_registration = True,
    skip_registration = False,
    registration = 'homography', # 'affine' 'partial affine' or 'homography' (use the least complex needed)
    fisheye_correction_file = os.path.join("sample_data", "fisheye_maps.npy"), # remove setting if n/a
    size = (1024,1024), # (width, height) how big to make the renderings, in pixels
    pixels_per_cm = 10, # for the arena drawn in register.generate_rendered_arena, report here the ratio between size of arena in pixels and actual size in cm
    by_experiment=False,
    experiments = ['block pre edge vectors'],
    by_session=True,
    sessions=[0], # This session points to the dank bank index
    all_sessions=False,
    efizz = False # Are you running the pipeline with efizz data or just behaviour? False if just behaviour

)
