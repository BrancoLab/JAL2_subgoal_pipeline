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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import seaborn as sns
from behave_analysis.utils.arena_plotting import Arena
from behave_analysis.utils.rm_escapes_from_homings import remove_escapes_from_homings_object
from mpl_toolkits.axes_grid1 import make_axes_locatable
from behave_analysis.analyze.behaviour.utils import base_plotting, identify_condition_of_trial, plot_trajectories
from behave_analysis.utils.data_loading import load_or_extract_homings
from behave_analysis.analyze.filtering_data.filtering_functions import identify_conditions


def plot_homings(session, tracking_data, video_df, homings_obj, show_plots=False) -> None:
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
    assert homings_obj is not None, "Failed to load homing data."
    assert hasattr(homings_obj, "onset_frames") and hasattr(
        homings_obj, "stimulus_durations"
    ), "Homings object must have 'onset_frames' and 'stimulus_durations'."

    try:
        escape_path = os.path.join(session.base_path, session.processed_path, "escapes", "escapes_obj.pkl")
        with open(escape_path, "rb") as f:
            escape_object = pickle.load(f)
        homings_obj = remove_escapes_from_homings_object(homings_obj, escape_object)
    except FileNotFoundError:
        raise FileNotFoundError("The escape object file does not exist. Please run the escape analysis first.")

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
    if show_plots:
        plt.show()
    plt.close()


def plot_the_start_of_each_homing(session, homing_object, video_df, tracking_data):
    """Plot the start of each homing so we can characterise behavioural variability

    Args:
    -- homing_info (list): A list of homing dataframes for each homing period
    -- classes (list): A list of classes for each homing period, is the homing target left or right

    Executes:
    -- A plot where the head direction and start location of each homing is plotted coloured by the class"""
    conditions = identify_conditions(session)

    if "barrier_present" in conditions:
        conditions.remove("barrier_present")

    if "shelter_present" in conditions:
        conditions.remove("shelter_present")
        
    if "pre_shelter" in conditions:
        conditions.remove("pre_shelter")

    fig, ax = plt.subplots(nrows=1, ncols=len(conditions), figsize=(20, 20))
    length = 180
    onsets = homing_object.onset_frames

    for i, con in enumerate(conditions):
        sum_homings = 0
        for onset in onsets:
            trial_condition = identify_condition_of_trial(video_df.filter(video_df["frames"] == onset), session)

            if con == "all_time":
                head_direction = video_df.filter(video_df["frames"] == onset)["hdir"][0]
                dx = length * np.cos(head_direction)
                dy = length * -np.sin(head_direction)
                x = video_df.filter(video_df["frames"] == onset)["mouse_x_position"][0]
                y = video_df.filter(video_df["frames"] == onset)["mouse_y_position"][0]
                ax[i].scatter(x, y, color="black")
                ax[i].quiver(x, y, dx, dy, angles="xy", scale_units="xy", scale=2, color="black")
                ax[i].set_title(con)
                Arena(ax=ax[i], shelter_coordinates=tracking_data["shelter_loc"], condition=con, barrier_coordinates=tracking_data["barrier_loc"])
                
                sum_homings += 1

            if trial_condition == con:
                head_direction = video_df.filter(video_df["frames"] == onset)["hdir"][0]
                dx = length * np.cos(head_direction)
                dy = length * -np.sin(head_direction)
                x = video_df.filter(video_df["frames"] == onset)["mouse_x_position"][0]
                y = video_df.filter(video_df["frames"] == onset)["mouse_y_position"][0]
                ax[i].scatter(x, y, color="black")
                ax[i].quiver(x, y, dx, dy, angles="xy", scale_units="xy", scale=2, color="black")
                Arena(ax=ax[i], shelter_coordinates=tracking_data["shelter_loc"], condition=con, barrier_coordinates=tracking_data["barrier_loc"])
            
                sum_homings += 1
                
        # if yaxes not inverted, then invert
        if ax[i].get_ylim()[1] > ax[i].get_ylim()[0]:
            ax[i].invert_yaxis()
    
        ax[i].set_title(f"{con} (n={sum_homings})")
        
    plt.savefig(os.path.join(session.base_path, session.processed_path, "homings", "start_of_homings.png"))
    plt.close()


def plot_the_probability_of_start_locations(session, homing_object, video_df, tracking_data):
    """Conduct a 2d histrogram normalised to count the probability of starting a homing at a given location"""
    conditions = identify_conditions(session)

    if "barrier_present" in conditions:
        conditions.remove("barrier_present")

    if "shelter_present" in conditions:
        conditions.remove("shelter_present")

    fig, ax = plt.subplots(nrows=1, ncols=len(conditions), figsize=(20, 20))
    onset_frames = homing_object.onset_frames

    for i, con in enumerate(conditions):
        start_locs = []
        for onset in onset_frames:
            trial_condition = identify_condition_of_trial(video_df.filter(video_df["frames"] == onset), session)

            if con == "all_time":
                start_locs.append(
                    (
                        video_df.filter(video_df["frames"] == onset)["mouse_x_position"][0],
                        video_df.filter(video_df["frames"] == onset)["mouse_y_position"][0],
                    )
                )

            if trial_condition == con:
                start_locs.append(
                    (
                        video_df.filter(video_df["frames"] == onset)["mouse_x_position"][0],
                        video_df.filter(video_df["frames"] == onset)["mouse_y_position"][0],
                    )
                )

        # Conduct 2D histogram
        x, y = zip(*start_locs)
        center = (512, 512)
        radius = 512

        # Conduct 2D histogram
        bins = np.arange(0, 1000, 50)
        hist, xedges, yedges = np.histogram2d(x, y, bins=bins, density=True)

        # Step 3: Define a circular mask
        center_x = 512
        center_y = 512
        radius = 512

        # Create a grid of the bin centers
        x_bin_centers = (xedges[:-1] + xedges[1:]) / 2
        y_bin_centers = (yedges[:-1] + yedges[1:]) / 2
        x_grid, y_grid = np.meshgrid(x_bin_centers, y_bin_centers)

        # Create the circular mask
        circular_mask = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2 <= radius**2

        # Step 4: Apply the mask and normalize the density within the circle
        circular_density = hist * circular_mask

        # Normalize the circular density so it integrates to 1
        circular_density_normalized = circular_density / circular_density.sum()

        #
        # Define the custom colormap
        cmap = sns.color_palette("viridis", as_cmap=True)
        cmap.set_bad("white")  # Color for no data due to no exploration
        cmap.set_under("grey")  # Color for zero data due to no spikes

        ax[i] = sns.heatmap(
            circular_density_normalized,
            cmap=cmap,
            ax=ax[i],
        )

        # Set the title with the number of points
        ax[i].set_title(f"{con} (n={len(start_locs)})")

        #

        #Arena(ax=ax[i], shelter_coordinates=tracking_data["shelter_loc"], condition=con, barrier_coordinates=tracking_data["barrier_loc"])

        # Set axis range to be between 0 and 1000
        ax[i].set_xlim(0, 1000)
        ax[i].set_ylim(0, 1000)

        # invert y axis
        ax[i].invert_yaxis()

    # Add color bar for last plot
    #plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.show()
