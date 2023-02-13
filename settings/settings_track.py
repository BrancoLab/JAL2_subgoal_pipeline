# Custom lib
"""Ensure when changing dlc setting file you include the path to the config.yaml file for DLC"""

from behave_analysis.utils.settings_objects import Settings_track

settings_track = Settings_track(

    redo_processing_step = True, # This flag does not control DLC
    min_confidence_in_tracking=0.5,
    max_deviation_from_rest_of_points=100, # in pixels
    display_tracking_output=True, # show a plot of tracking data from DLC
    by_experiment=False,
    experiments = ['no laser'],
    by_session=True,
    sessions = [0], # If running one session and this is set to 1 will run with no errors but will not process anything
    all_sessions = False,
    dlc_settings_file = r"D:\DLC\NPX_7-Laurence-2022-05-20\config.yaml", #Change if using a different DLC model, you need the path to include the full yaml such as: r"D:\DLC\NPX_7-Laurence-2022-05-20\config.yaml"
    inverse_fisheye_correction_file = '.\\sample_data\\inverse_fisheye_maps.npy' # remove setting if n/a
    
)
