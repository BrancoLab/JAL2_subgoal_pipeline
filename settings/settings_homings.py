from behave_analysis.utils.settings_objects import Settings_homings

settings_homings = Settings_homings(
    fast_speed=15,  # 15 cm/s seems quite slow no? In the paper philip used 10 cm/s
    padding_duration=0.5,  # How long should the homing event be in seconds
    fast_angular_speed=90,  # Turning towards some reference location speed
    min_change_in_dist_to_shelter=0.3,  # (maybe %) How far does the mouse have to move towards the shelter to be considered a homing event
    max_time_within_session=2000,  # How long is session in minutes - Ignore I think
    threat_area_width=820,
    threat_area_height=275,
    cum_threshold=25,  # How many cm does the mouse have to move when considering homing angle
    # Commenting out the hard coded subgoal locations as we click on them in the GUI
    # subgoal_locations             = [(512-250, 512),(512+250, 512)],
    # Threshold script only logic?
    # duration_after_crossing=6,  # THink only used in threshold script
    # by_experiment=False,
    # experiments=["block edge vectors"],
    by_session=True,
    sessions=[0],
)
