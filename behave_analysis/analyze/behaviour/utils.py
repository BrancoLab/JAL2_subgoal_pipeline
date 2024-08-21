"""
This module serves as a collection of utility functions specifically tailored to support spatial analysis and homing modules in behavioral research. 
It focuses on providing tools for visualizing and interpreting spatial data related to escape and homing behaviors in experimentd.

Key Functions:
1. base_plotting:
   - Purpose: To establish the foundational layout of the experimental arena, including key elements like shelters and barriers, based on tracking data.

2. identify_condition_of_trial:
   - Purpose: To classify each trial into specific conditions (e.g., 'shelter_only', 'barrier_pre_flip', 'barrier_post_flip') based on the combination of 
   video data and session information.

3. plot_trajectories:
   - Purpose: To visually represent the movement paths and speeds of subjects in the arena, using tracking data including onset frames and stimulus durations.
"""

import numpy as np
import matplotlib.pyplot as plt

from behave_analysis.utils.color_funcs import get_color_based_on_speed
from behave_analysis.visualize.visualize_utils import open_tracking_data


def base_plotting(ax, tracking, condition, session=[]) -> None:
    """
    Plot the base layout of the arena, including the shelter and barriers, based on the given condition.

    This function is used to draw the foundational elements of an experimental arena used in behavioral studies.
    It plots the location of shelters and barriers as per the specified condition, providing a visual context
    for analyzing animal behaviors within the arena.

    Parameters:
    - ax (matplotlib.axes.Axes): The matplotlib axes object to plot on.
    - tracking (dict): A dictionary containing tracking data with keys like 'shelter_loc' and 'barrier_loc'.
    - condition (str): The specific condition of the trial (e.g., 'shelter_only', 'barrier_pre_flip', etc.).
    - session (list, optional): Session data, used if tracking data is not provided.

    The function checks for the presence of shelter and barrier locations in the tracking data and plots them
    accordingly. It also outlines the arena based on a predefined radius.

    Raises:
    - AssertionError: If the input ax is not a matplotlib.axes.Axes instance or if tracking data is not in the expected format.
    """

    # Assertions to validate inputs
    assert isinstance(ax, plt.Axes), "ax must be a matplotlib.axes.Axes instance."
    assert isinstance(tracking, dict), "tracking must be a dictionary."
    assert isinstance(condition, str), "condition must be a string."
    assert "shelter_loc" in tracking or "barrier_loc" in tracking, "tracking must contain 'shelter_loc' and/or 'barrier_loc' keys."

    if len(tracking) == 0:
        tracking = open_tracking_data(session)

    arena_radius = 460

    # draw shelter
    if not(condition == "pre_shelter"):
        if "shelter_loc" in tracking.keys():
            for i in [0, 1]:
                plt.plot(
                    [tracking["shelter_loc"][0][0], tracking["shelter_loc"][1][0]],
                    [tracking["shelter_loc"][i][1], tracking["shelter_loc"][i][1]],
                    color=[1, 0, 0],
                )
                plt.plot(
                    [tracking["shelter_loc"][i][0], tracking["shelter_loc"][i][0]],
                    [tracking["shelter_loc"][0][1], tracking["shelter_loc"][1][1]],
                    color=[1, 0, 0],
                )

    # draw barrier logic
    if not np.logical_or(condition == "shelter_only", condition == "pre_shelter"):
        if len(tracking["barrier_loc"]) > 0:
            if np.logical_or(np.logical_or(condition == 'barrier_present',condition == 'all_time'),condition == 'shelter_present'):
                # draw old two-sided barrier
                bar_loc = [tracking["barrier_loc"][0][0], tracking["barrier_loc"][1][0]]

            if condition == "barrier_pre_flip":
                # draw barrier from first point to the edge
                if tracking["barrier_loc"][0][0] < 512:
                    bar_loc = [tracking["barrier_loc"][0][0], 512 + arena_radius]
                else:
                    bar_loc = [512 - arena_radius, tracking["barrier_loc"][0][0]]

            if condition == "barrier_post_flip":
                # draw barrier from second point to the edge
                if tracking["barrier_loc"][1][0] < 512:
                    bar_loc = [tracking["barrier_loc"][1][0], 512 + arena_radius]
                else:
                    bar_loc = [512 - arena_radius, tracking["barrier_loc"][1][0]]

            if not condition == "barrier_removed":
                # draw barrier location onto the arena base
                plt.plot([bar_loc[0], bar_loc[1]], [tracking["barrier_loc"][0][1], tracking["barrier_loc"][1][1]], color=[0, 0, 0])

    # draw arena edge
    a = 512 + (arena_radius * np.cos(np.linspace(0, 2 * np.pi, 150)))
    b = 512 + (arena_radius * np.sin(np.linspace(0, 2 * np.pi, 150)))

    ax.plot(a, b, color=[0, 0, 0])
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")


def identify_condition_of_trial(video_df, session) -> str:
    """
    Determine the experimental condition of a trial based on video and session data.

    This function analyzes the given DataFrame and session object to ascertain the specific condition
    under which a trial occurred. It categorizes the trial into one of several predefined conditions
    based on the status of the shelter and barrier at the time of the trial.

    Parameters:
    - video_df (DataFrame): A DataFrame containing columns 'shelter', 'barrier_present', and 'barrier_flipped'.
                             Each column is expected to have boolean values indicating the status of each element.
    - session (Object): An object representing the session, which should include the 'barrier_flip_time' attribute.

    Returns:
    - str: The identified condition of the trial, which can be 'shelter_only', 'barrier_pre_flip', 'barrier_post_flip', or 'barrier_present'.

    Raises:
    - AssertionError: If the input DataFrame does not contain the expected columns or if the session object is not in the expected format.
    """

    # Assertions to validate inputs
    assert all(
        col in video_df.columns for col in ["shelter", "barrier_present", "barrier_flipped"]
    ), "video_df must contain 'shelter', 'barrier_present', and 'barrier_flipped' columns."
    assert hasattr(session, "barrier_flip_time"), "session must have 'barrier_flip_time' attribute."

    condition = ""

    if video_df["shelter"].to_numpy() == False:
        condition = 'pre_shelter'

    # Check if mouse is in the shelter condition
    if np.logical_and(video_df["shelter"].to_numpy() == True, video_df["barrier_present"].to_numpy() == False):
        condition = "shelter_only"
        if video_df["barrier_flipped"].to_numpy() == True:
            condition = "barrier_removed" # ATTENTION: this only works if barrier is removed after flip!!!

    # Check which barrier condition the mouse is in
    elif np.logical_and(video_df["shelter"].to_numpy() == True, video_df["barrier_present"].to_numpy() == True):
        if session.barrier_flip_time:
            # Check if the barrier has been flipped
            if video_df["barrier_flipped"].to_numpy() == False:
                condition = "barrier_pre_flip"

            # Check if the barrier has been flipped
            if np.logical_and(video_df["barrier_flipped"].to_numpy() == True, video_df["barrier_present"].to_numpy() == True):
                condition = "barrier_post_flip"

        else:
            condition = "barrier_present"

    return condition


def plot_trajectories(onset_frames, stimulus_durations, ax, stim_type, tracking_data):
    """
    Plot the trajectories of an object based on tracking data.

    This function computes and plots the trajectory of an object over a given duration,
    represented by changes in its location. The trajectory is colored based on the object's
    speed. It also calculates and returns the total distance traveled by the object during
    the specified stimulus duration.

    Parameters:
    - onset_frames (int): The frame number where the trajectory starts.
    - stimulus_durations (int): The duration of the stimulus in frames.
    - ax (matplotlib.axes.Axes): The matplotlib axis object to plot the trajectory on.
    - stim_type (str): The type of stimulus
    - tracking_data (DataFrame): A DataFrame containing tracking data, including head location and average velocity.

    Returns:
    - float: The total distance traveled by the object during the stimulus duration.

    Raises:
    - AssertionError: If any of the input parameters are not in the expected format or range.
    """

    # Assertions to validate input
    if not isinstance(onset_frames, np.int64):
        onset_frames = int(onset_frames)
    assert onset_frames >= 0, "onset_frames must be a non-negative integer."
    assert stimulus_durations > 0, "stimulus_durations must be a positive integer."
    assert hasattr(ax, "scatter"), "ax must be a valid matplotlib.axes.Axes object."
    assert isinstance(stim_type, str), "stim_type must be a string."
    assert "head_loc" in tracking_data and "avg_Velocity" in tracking_data, "tracking_data must contain 'head_loc' and 'avg_Velocity'."

    # compute and plot each trajectory
    x_loc = tracking_data["head_loc"][onset_frames : onset_frames + int(stimulus_durations), 0]
    y_loc = tracking_data["head_loc"][onset_frames : onset_frames + int(stimulus_durations), 1]
    speed = tracking_data["avg_Velocity"][onset_frames : onset_frames + int(stimulus_durations)]
    trail_color = np.empty((len(speed), 3))
    distance_travelled = []
    for i, stim_status in enumerate(np.arange(0, stimulus_durations)):
        trail_color[i, :] = get_color_based_on_speed(speed=speed[i], object_to_color="trail", stim_status=stim_status, stim_type=stim_type)
        if i > 0:
            distance_travelled = np.append(distance_travelled, np.sqrt((x_loc[i] - x_loc[i - 1]) ** 2 + (y_loc[i] - y_loc[i - 1]) ** 2))
    ax.scatter(x_loc, y_loc, s=5, c=trail_color / 255)
    return np.sum(distance_travelled)
