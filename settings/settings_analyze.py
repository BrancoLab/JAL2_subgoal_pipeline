from behave_analysis.utils.settings_objects import Settings_analyze_global as Settings_analyze
from settings.analyses import analyses
from pathlib import Path

settings_analyze = Settings_analyze(

analysis = analyses["explore"], #Change explore test within other options in analyses.py
max_num_trials = 6,
max_escape_duration = 9,
post_laser_seconds_to_plot = 0,
min_distance_from_shelter = 10,
escape_initiation_speed = 20,
edge_vector_threshold = 0.68,
binarize_statistics = True,
two_tailed_test = True,
leftside_only = False,
rightside_only = False,
reflect_trajectories = False,
color_by = 'session', # What should the trajectory color be?
# 'default' 'session' 'trial' 'target'  ''  || for all
# 'speed'   'time'    'speed+RT'            || for trajectories
# Note default caused a bug need to fix
    
save_folder = Path('data'),
efiz_file_path = "D:/Electrophysiology_data/1677_NoShelterThenShelter_22MAY31_g0/1677_NoShelterThenShelter_22MAY31_g0_imec0/"

)

