from behave_analysis.utils.settings_objects import Settings_homings

settings_homings = Settings_homings(

    fast_speed                    = 15,
    padding_duration              = 1,
    fast_angular_speed            = 90,
    min_change_in_dist_to_shelter = .2,
    max_time_within_session       = 20,
    threat_area_width             = 820, 
    threat_area_height            = 275, 
    subgoal_locations             = [(512-250, 512),(512+250, 512)],

    duration_after_crossing       = 6,

    by_experiment = False,
    experiments = ['block edge vectors'],

    by_session=True,
    sessions=[0]


)