import numpy as np

from behave_analysis.utils.settings_objects import Settings_homings

settings_homings = Settings_homings(
    fast_speed=10,  # 15 cm/s seems quite slow no? In the paper philip used 10 cm/s
    padding_duration=0.5,  # How long should the box car filter be in seconds and how long should a homing event be
    fast_angular_speed=np.pi / 4,  # Turning towards some reference location speed in rad/s with a threshold of 60 degrees
    edge_proximity=100,  # 10 pixels is 1 cm, so 100 pixels is 10 cm. Defining subgoal start homings
    min_change_in_dist_to_shelter=0.3,  # (maybe %) How far does the mouse have to move towards the shelter to be considered a homing event
    max_time_within_session=2000,  # How long is session in minutes - Ignore I think
    threat_area_width=820,
    threat_area_height=275,
    cum_threshold=25,  # How many cm does the mouse have to move when considering homing angle
    by_session=True,
    sessions=[0],
    min_frames_between_trials=40,  # 1 second between trials to stop double counting split homings
)
