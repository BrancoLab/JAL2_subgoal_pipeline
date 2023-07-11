# Custom lib
"""Ensure when changing dlc setting file you include the path to the config.yaml file for DLC"""

from behave_analysis.utils.settings_objects import Settings_track

settings_track = Settings_track(

    redo_processing_step = True, # This flag does not control DLC
    min_confidence_in_tracking = 0.9, # Abitraly set, can be changed. The higher the better but will break if too high
    max_deviation_from_rest_of_points = 100, # in pixels
    display_tracking_output = False, # show a plot of tracking data from DLC
    by_experiment = False,
    experiments = ['no laser'],
    by_session = True,
    all_sessions = False,
    # dlc_settings_file = r"D:\DLC\JAL_NPX1-Jasmine-2023-03-22\config.yaml", #Change if using a different DLC model, you need the path to include the full yaml such as: r"D:\DLC\NPX_7-Laurence-2022-05-20\config.yaml"
    dlc_settings_file = r"C:\Users\jreggiani\Documents\DLC\JAL_NPX1-Jasmine-2023-03-22\config.yaml", #Change if using a different DLC model, you need the path to include the full yaml such as: r"D:\DLC\NPX_7-Laurence-2022-05-20\config.yaml"
    inverse_fisheye_correction_file = '.\\sample_data\\inverse_fisheye_maps.npy', # remove setting if n/a
    tracking_file_location = None,
    save_labeled_video = False # if you want to save the video with the DLC dots plotted on it
    
)
