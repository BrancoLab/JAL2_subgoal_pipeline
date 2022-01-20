from behave_analysis.utils.settings_objects import Settings_visualize

settings_visualize = Settings_visualize(

    laser_trials=False,
    escape_trials=True,
    homing_trials = False,
    t_xing_trials = False,
    
    display_trail=True,
    display_tracking=False,

    display_stimulus = True,
    rapid = True,

    seconds_before_audio = 3,
    seconds_before_laser = 3,
    seconds_before_homing = 3,
    seconds_before_threshold_crossing = 3,
    seconds_after_audio = 2,
    seconds_after_laser = 6,
    seconds_after_homing = 3,
    seconds_after_threshold_crossing = 3,

    save_folder = "..\\",

    by_experiment=False,
    experiments = ['no laser'],

    by_session=True,
    sessions=[0,1,2],
)
