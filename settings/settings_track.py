# Custom lib

from behave_analysis.utils.settings_objects import Settings_track

settings_track = Settings_track(

    redo_processing_step=True,
    min_confidence_in_tracking=0.5,
    max_deviation_from_rest_of_points=100, # in pixels
    display_tracking_output=False, # show a plot of tracking data from DLC
    by_experiment=False,
    experiments = ['no laser'],
    by_session=True,
    sessions=[1], 
    all_sessions=False,
    dlc_settings_file='D:\\DLC\\NPX_7-Laurence-2022-05-20\\config.yaml', #Change if using a different DLC model
    inverse_fisheye_correction_file = '.\\sample_data\\inverse_fisheye_maps.npy' # remove setting if n/a
    
)
