# Custom lib
"""Ensure when changing dlc setting file you include the path to the config.yaml file for DLC"""

from behave_analysis.utils.settings_objects import Settings_track

settings_track = Settings_track(

    redo_processing_step = True, # This flag does not control DLC, but it will regenerate kalman 
    min_confidence_in_tracking = 0.9, # Abitraly set, can be changed. The higher the better but will break if too high
    max_deviation_from_rest_of_points = 75, # in pixels
    display_tracking_output = False, # show a plot of tracking data from DLC
    by_experiment = False,
    experiments = ['no laser'],
    by_session = True,
    all_sessions = False,
    inverse_fisheye_correction_file = '.\\sample_data\\inverse_fisheye_maps.npy', # remove setting if n/a
    tracking_file_location = None,
    # Is will use the output of DLC nothing that has been processed
    save_labeled_video = False, # if you want to save the video with the DLC dots plotted on it
    random_points = 'full_arena', # compute the head angle to extra points in the arena, 'manual' makes you input points in gui, 'full_arena' computes a full grid of points in the arena, [] empty if you don't want angle with randompoints
)
