# Custom lib
"""Ensure when changing dlc setting file you include the path to the config.yaml file for DLC"""

from behave_analysis.utils.settings_objects import Settings_track

settings_track = Settings_track(
    redo_processing_step=True,  # This flag does not control DLC, but it will regenerate kalman
    min_confidence_in_tracking=0.9,  # Abitraly set, can be changed. The higher the better but will break if too high
    max_deviation_from_rest_of_points=75,  # in pixels
    interpolation_max_gap_frames=4,  # only interpolate short internal NaN gaps up to this many frames
    jump_reject_max_speed_cm_s=120.0,  # reject per-bodypart frame-to-frame jumps above this speed
    jump_reject_expand_frames=1,  # also reject this many neighboring frames around each large jump
    use_kalman=True,  # set False to skip kalman and use cleaned trajectories directly
    display_tracking_output=False,  # show a plot of tracking data from DLC
    inverse_fisheye_correction_file=".\\sample_data\\inverse_fisheye_maps.npy",  # remove setting if n/a
    tracking_file_location=None,
    # Is will use the output of DLC nothing that has been processed
    save_labeled_video=False,  # if you want to save the video with the DLC dots plotted on it
    random_points=None,  # compute the head angle to extra points in the arena, 'manual' makes you input points in gui, 'full_arena' computes a full grid of points in the arena, None if you don't want angle with randompoints
    max_deviation_from_arena=95,  # in pixels, points outside of the arena will be set to nan, this is used to correct for tracking points that are outside of the arena
)
