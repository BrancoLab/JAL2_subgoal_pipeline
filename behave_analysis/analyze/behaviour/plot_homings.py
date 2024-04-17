"""
This module provides functionality for loading homing trajectory data and plotting these trajectories onto an arena layout. 
It leverages matplotlib for visualization and uses a grid layout to organize multiple homing trajectory plots into a single 
figure or multiple figures, depending on the number of trials.

Functions:
    - load_homings: Loads the homing data from a pickle file specific to a session.
    - plot_homings: Plots the homing trajectories for each trial in the session data. This function manages the creation of one or more figures, 
    each containing a grid of subplots. Each subplot represents a single trial's homing trajectory.

The module requires that the homing data be pre-processed and stored as a pickle file. It utilizes functions from the 
'behave_analysis.analyze.behaviour.utils' module for plotting base arena layout, identifying trial conditions, and plotting individual trajectories.

Note:
    The script is designed to handle multiple trials and conditions, and it dynamically adjusts the number of figures 
    and the layout of subplots based on the number of trials. Each figure can display up to 20 trials in a 4x5 grid layout.
"""

import os

import dill as pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from behave_analysis.analyze.behaviour.utils import base_plotting, identify_condition_of_trial, plot_trajectories
from behave_analysis.utils.data_loading import load_or_extract_homings


def plot_homings(session, tracking_data, video_df) -> None:
    """Plot and visualize the homing trajectories for a given session.

    This function creates one or more figures, each containing a grid of subplots, with each subplot representing
    the homing trajectory of a single trial. The function dynamically determines the number of figures and subplots
    based on the total number of trials, with a maximum of 20 trials per figure laid out in a 4x5 grid.

    Parameters:
    - session (SessionType): An object representing the session. This object is expected to provide necessary
      attributes and methods for determining the file paths and other session-specific details.
    - tracking_data (DataFrame): Data containing tracking information, used for base plotting in each subplot.
    - video_df (DataFrame): Video data frame used to identify the condition of each trial.

    The function utilizes several helper functions to process and visualize the data:
    - `load_homings`: Loads the homing data for the session.
    - `identify_condition_of_trial`: Determines the condition of each trial based on video data.
    - `base_plotting`: Plots the base layout of the arena.
    - `plot_trajectories`: Plots the actual homing trajectories on the subplots.

    Note:
    - Each subplot is a visualization of a single trial's homing trajectory within the session.
    - The function assumes that the `session` object provides access to video frame rate information.
    - The layout of the figures and subplots is subject to change based on the number of trials.
    """
    homings_obj = load_or_extract_homings(session)
    assert homings_obj is not None, "Failed to load homing data."
    assert hasattr(homings_obj, 'onset_frames') and hasattr(homings_obj, 'stimulus_durations'), "Homings object must have 'onset_frames' and 'stimulus_durations'."

    # Configure plot params
    ntrials = len(homings_obj.onset_frames)
    assert ntrials > 0, "Number of trials must be greater than zero."
    nrows = 4
    ncols = 5

    # Determine number of figures
    if ntrials > 20:
        number_of_figures = int(ntrials // 20 + (ntrials % 20 > 0))
    else:
        number_of_figures = 1

    # Plot up to 20 trials per figure
    trial_counter = 0
    for figure in range(number_of_figures):
        fig = plt.figure(figsize=(20, 16))
        # Plot the title of the figure which is homings for a session for one figure
        fig.suptitle(f"Homings for {session.name}, figure {figure + 1} of {number_of_figures}", fontsize=16)
        gs = gridspec.GridSpec(nrows, ncols, wspace=0, hspace=0)
        for row in range(nrows):
            for col in range(ncols):
                if trial_counter == ntrials:
                    break

                # Get homing details
                onset_frame, stimulus_durations = homings_obj.onset_frames[trial_counter], homings_obj.stimulus_durations[trial_counter][0]
                trial_condition = identify_condition_of_trial(video_df.filter(video_df["frames"] == onset_frame), session)

                # Create subplot
                ax = fig.add_subplot(gs[row, col])
                base_plotting(ax, tracking_data, condition=trial_condition)
                plot_trajectories(onset_frame, stimulus_durations * session.video.fps, ax, "homing", tracking_data)

                trial_counter += 1
        
        # Save figure
        fig.savefig(os.path.join(session.base_path, session.processed_path, "homings", f"homings_figure_{figure}.png"))
    plt.show()
