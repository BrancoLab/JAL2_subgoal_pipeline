from behave_analysis.utils.settings_objects import Settings_track

settings_track = Settings_track(

    redo_processing_step=True,

    min_confidence_in_tracking=0.5,
    max_deviation_from_rest_of_points=100, # in pixels
    display_tracking_output=False, # show a plot of tracking data

    by_experiment=False,
    experiments = ['no laser'],

    by_session=True,
    sessions=[6, 7, 8, 9, 10, 11], 

    all_sessions=False,

    dlc_settings_file='D:\\data\\DLC_nets\\opto-philip-2021-07-26\\config.yaml', #'D:\\data\\DLC_nets\\Barnes-Philip-2020-12-07\\config.yaml', #'D:\\data\\DLC_nets\\Barnes-Philip-2018-11-22\\config.yaml', # 
    inverse_fisheye_correction_file = '.\\sample_data\\inverse_fisheye_maps.npy' # remove setting if n/a
)
