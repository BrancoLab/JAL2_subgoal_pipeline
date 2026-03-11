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
import re
import dill as pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import seaborn as sns
from behave_analysis.utils.arena_plotting import Arena
from behave_analysis.utils.rm_escapes_from_homings import remove_escapes_from_homings_object
from behave_analysis.analyze.behaviour.utils import plot_trajectories
from behave_analysis.analyze.filtering_data.filtering_functions import identify_conditions
from behave_analysis.analyze.behaviour.homings_escapes.homings import cum_distance
from behave_analysis.utils.creating_directories import make_directory
from behave_analysis.database.computer_ID import get_computer_specific_paths

def plot_homings(session, tracking_data, homings_obj, show_plots=False) -> None:
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
                onset_frame, stimulus_durations = (
                    homings_obj.onset_frames[trial_counter],
                    homings_obj.stimulus_durations[trial_counter][0],
                )
                trial_condition = homings_obj.homing_condition[trial_counter]

                # Create subplot
                ax = fig.add_subplot(gs[row, col])
                Arena(
                    ax=ax,
                    shelter_coordinates=tracking_data["shelter_loc"],
                    condition=trial_condition,
                    barrier_coordinates=session.barrier_location,
                )
                # base_plotting(ax, tracking_data, condition=trial_condition, session = session)
                plot_trajectories(onset_frame, stimulus_durations * session.video.fps, ax, "homing", tracking_data)

                trial_counter += 1

        # Save figure
        fig.savefig(os.path.join(session.base_path, session.processed_path, "analyze_behave", f"homings_figure_{figure}.png"))
        if show_plots:
            plt.show()
        plt.close()


def plot_the_start_of_each_run(session, onsets, hdir_at_start, all_conditions, tracking_data, title, show_plots=False):
    """Plot the start of each homing so we can characterise behavioural variability

    Args:
    -- homing_info (list): A list of homing dataframes for each homing period
    -- classes (list): A list of classes for each homing period, is the homing target left or right

    Executes:
    -- A plot where the head direction and start location of each homing is plotted coloured by the class"""

    # paths for saving in summary dir
    ceph_path, _ = get_computer_specific_paths(session_path = '', return_ceph = True)
    ceph_path = os.path.dirname(ceph_path)
    overall_path = make_directory(os.path.join(ceph_path, 'summary_plots', title + '_plots'))
    match = re.search(r'(\d{4}_\d{2}_\d{2})T', session.file_path) # these three lines could be replaced with session.date if process is rerun
    if match:
        date_str = match.group(1)

    conditions = identify_conditions(session)

    if "barrier_pre_flip" in conditions:
        conditions.remove("barrier_present")

    if "shelter_only" in conditions:
        conditions.remove("shelter_present")

    _, ax = plt.subplots(nrows=1, ncols=len(conditions), figsize=(20, 4))
    length = 180

    for i, con in enumerate(conditions):
        sum_homings = 0
        for idx, onset in enumerate(onsets):
            if np.isnan(onset):
                continue
            trial_condition = all_conditions[idx]
            if isinstance(onset,np.ndarray):
                onset = onset[0].astype(int)

            if np.logical_or(con == trial_condition, con == "all_time"):
                dx = length * np.cos(hdir_at_start[idx])
                dy = length * -np.sin(hdir_at_start[idx])
                mouse = tracking_data["avg_loc"][onset - 1]
                ax[i].scatter(mouse[0], mouse[1], color="black")
                ax[i].quiver(mouse[0], mouse[1], dx, dy, angles="xy", scale_units="xy", scale=2, color="black")
                ax[i].set_title(con)
                sum_homings += 1

        Arena(ax=ax[i], shelter_coordinates=tracking_data["shelter_loc"], condition=con, barrier_coordinates=session.barrier_location)
        ax[i].set_title(f"{con} (n={sum_homings})")

    # save figure in session dir
    plt.savefig(os.path.join(session.base_path, session.processed_path, "analyze_behave", str("start_of_"+title+".png")))
    # save in summary dir
    filename = session.mouse + '_' + date_str + '_' + str("start_of_"+title+".png")
    plt.savefig(overall_path + '/' + filename)
    if show_plots: plt.show()
    plt.close()

def plot_the_probability_of_start_locations(session, onset_frames, all_conditions, tracking_data, title, show_plots=False):
    """Conduct a 2d histrogram normalised to count the probability of starting a homing at a given location"""

    # paths for saving in summary dir
    ceph_path, _ = get_computer_specific_paths(session_path = '', return_ceph = True)
    ceph_path = os.path.dirname(ceph_path)
    overall_path = make_directory(os.path.join(ceph_path, 'summary_plots', title + '_plots'))
    match = re.search(r'(\d{4}_\d{2}_\d{2})T', session.file_path) # these three lines could be replaced with session.date if process is rerun
    if match:
        date_str = match.group(1)

    conditions = identify_conditions(session)

    if "barrier_pre_flip" in conditions:
        conditions.remove("barrier_present")

    if "shelter_only" in conditions:
        conditions.remove("shelter_present")

    fig, ax = plt.subplots(nrows=1, ncols=len(conditions), figsize=(20, 4))
    cbar_ax = fig.add_axes([0.91, 0.13, 0.01, 0.75])

    for i, con in enumerate(conditions):
        start_locs = []
        for idx, onset in enumerate(onset_frames):
            if np.isnan(onset):
                continue
            trial_condition = all_conditions[idx]
            if isinstance(onset,np.ndarray):
                onset = onset[0].astype(int)

            if np.logical_or(con == trial_condition, con == "all_time"):
                start_locs.append(
                    (
                        tracking_data["avg_loc"][onset - 1][0],
                        tracking_data["avg_loc"][onset - 1][1],
                    )
                )

        if len(start_locs) > 0:
            # Conduct 2D histogram
            x, y = zip(*start_locs)
            bins = np.arange(0, 1025, 32)
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
            circular_density = (hist.T) * circular_mask

            # Normalize the circular density so it integrates to 1
            circular_density_normalized = circular_density / circular_density.sum()

            #
            # Define the custom colormap
            cmap = sns.color_palette("viridis", as_cmap=True).copy()
            cmap.set_bad("white")  # Color for no data due to no exploration
            cmap.set_under("grey")  # Color for zero data due to no spikes

            ax[i] = sns.heatmap(circular_density_normalized,
                                cmap="coolwarm",
                                cbar_ax=cbar_ax,
                                robust=True,
                                ax=ax[i],
                                mask=(circular_density_normalized == 0),
                                cbar_kws={"label": "Normalised time spent in position"},
                                norm=plt.Normalize(vmin=0, vmax=1))

        # Set the title with the number of points
        ax[i].set_title(f"{con} (n={len(start_locs)})")
        
        Arena(dim = np.amax(ax[i].get_ylim()), ax=ax[i], shelter_coordinates=tracking_data["shelter_loc"], condition=con, barrier_coordinates=session.barrier_location)

    if show_plots: plt.show()
    # save figure in session dir
    plt.savefig(os.path.join(session.base_path, session.processed_path, "analyze_behave", str("start_of_"+title+"_loc_probability.png")))
    # save in summary dir
    filename = session.mouse + '_' + date_str + '_' + str("start_of_"+title+"_loc_probability.png")
    plt.savefig(overall_path + '/' + filename)
    plt.close()

def hist_initial_heading_angle(session, onsets, offsets, head_angle, all_conditions, tracking_data, title, show_plots=False, plotting = True):
    """Finds the cosine similarity between the heading of the mouse when he starts running in the homing and the angle with the three goals
    Assigns the homing heading to the goal it is most similar to
    Doesn't differentiate between below and above the barrier so a lot of shelter targets are actually below the barrier"""

    # paths for saving in summary dir
    ceph_path, _ = get_computer_specific_paths(session_path = '', return_ceph = True)
    ceph_path = os.path.dirname(ceph_path)
    overall_path = make_directory(os.path.join(ceph_path, 'summary_plots', title + '_plots'))
    match = re.search(r'(\d{4}_\d{2}_\d{2})T', session.file_path) # these three lines could be replaced with session.date if process is rerun
    if match:
        date_str = match.group(1)

    conditions = identify_conditions(session)

    if np.logical_and(not 'barrier_present' in conditions, 'shelter_present' in conditions):
        conditions = conditions + ['shelter_only']

    if "barrier_pre_flip" in conditions:
        conditions.remove("barrier_present")

    if "shelter_only" in conditions:
        conditions.remove("shelter_present")
    
    if plotting:    
        _, ax = plt.subplots(nrows=1, ncols=len(conditions), figsize=(15, 5))

    heading_by_cond = {}

    for i, con in enumerate(conditions):
        pref_heading = np.zeros(3)
        sum_homings = 0
        for idx, (onset,offset) in enumerate(zip(onsets,offsets)):
            if np.isnan(onset):
                continue
            trial_condition = all_conditions[idx]
            if isinstance(onset,np.ndarray):
                onset = onset[0].astype(int)

            if np.logical_or(con == trial_condition, con == "all_time"):

                frame_coords = tracking_data["avg_loc"][onset:offset]
                # _, start_frame = cum_distance(onset, offset, frame_coords, session.video.pixels_per_cm, 15)
                _, start_frame = cum_distance(onset, offset, frame_coords, 10, 15)

                # calculate the preference of mouse heading for one of three targets using cosine similarity
                xdist = -tracking_data['head_loc'][start_frame, 0]+tracking_data['barrier_loc'][0][0]
                ydist = -tracking_data['head_loc'][start_frame, 1]+tracking_data['barrier_loc'][0][1]
                bprea = - np.arctan2(ydist, xdist)
                xdist = -tracking_data['head_loc'][start_frame, 0]+tracking_data['barrier_loc'][1][0]
                ydist = -tracking_data['head_loc'][start_frame, 1]+tracking_data['barrier_loc'][1][1]
                bposta = - np.arctan2(ydist, xdist)
                if tracking_data["bod_shelt_dir"][start_frame] < 0: bsa = tracking_data["bod_shelt_dir"][start_frame] + np.pi
                if tracking_data["bod_shelt_dir"][start_frame] > 0:  bsa = tracking_data["bod_shelt_dir"][start_frame] - np.pi

                cosim=[]
                for ang in [bprea,bsa, bposta]:
                    cosim = np.append(cosim,np.cos(ang-head_angle[idx]))
                pref_heading[np.argmax(cosim)] += 1
                sum_homings += 1

        if plotting: 
            ax[i].bar(np.arange(3),pref_heading,color = ['green','red','blue'])
            ax[i].set_ylabel('number of homings')
            ax[i].set_xlabel('homing target')
            ax[i].set_xticks(np.arange(3))
            ax[i].set_xticklabels(['preflip','shelter','postflip'])
            ax[i].set_title(f"{con} (n={sum_homings})")

        heading_by_cond[con] = pref_heading
    
    if plotting:
        plt.tight_layout()
        # save figure in session dir
        plt.savefig(os.path.join(session.base_path, session.processed_path, "analyze_behave", str("hist_"+title+"_heading_angle.png")))
        # save in summary dir
        filename = session.mouse + '_' + date_str + '_' + str("hist_"+title+"_heading_angle.png")
        plt.savefig(overall_path + '/' + filename)
        if show_plots:
            plt.show()
        plt.close()

    return heading_by_cond

def trial_initial_heading_angle(session, onsets, offsets, head_angle, hdir_at_start, all_conditions, tracking_data, title, show_plots=False):
    """On a drawing of the arena it plots the heading of the mouse before the homing begins and as the mouse starts running (after the head turn)
    The heading of the mouse as he is running is colored by the goal it is targeting (using cosine similarity)"""

    # paths for saving in summary dir
    ceph_path, _ = get_computer_specific_paths(session_path = '', return_ceph = True)
    ceph_path = os.path.dirname(ceph_path)
    overall_path = make_directory(os.path.join(ceph_path, 'summary_plots', title + '_plots'))
    match = re.search(r'(\d{4}_\d{2}_\d{2})T', session.file_path) # these three lines could be replaced with session.date if process is rerun
    if match:
        date_str = match.group(1)

    conditions = identify_conditions(session)

    if "barrier_pre_flip" in conditions:
        conditions.remove("barrier_present")

    if "shelter_only" in conditions:
        conditions.remove("shelter_present")

    _, ax = plt.subplots(nrows=1, ncols=len(conditions)+1, figsize=(20, 4))
    length = 180

    for i, con in enumerate(conditions):
        sum_homings = 0
        for idx, (onset,offset) in enumerate(zip(onsets,offsets)):
            if np.isnan(onset):
                continue
            trial_condition = all_conditions[idx]
            if isinstance(onset,np.ndarray):
                onset = onset[0].astype(int)

            if np.logical_or(con == trial_condition, con == "all_time"):
                # add an arrow for where the mouse was facing before the head turn
                dx = length * np.cos(hdir_at_start[idx])
                dy = length * -np.sin(hdir_at_start[idx])
                mouse = tracking_data["avg_loc"][onset - 1]
                ax[i].scatter(mouse[0], mouse[1], color="black")
                ax[i].quiver(mouse[0], mouse[1], dx, dy, angles="xy", scale_units="xy", scale=2, color="black")

                # arrow of which way mouse is facing once it started running, colored by whatever it is targeting
            
                frame_coords = tracking_data["avg_loc"][onset:offset]
                _, start_frame = cum_distance(onset, offset, frame_coords, session.video.pixels_per_cm, 15)
                mouse = tracking_data["head_loc"][start_frame]
                if len(mouse) == 1: mouse = mouse[0] # why did this happen once?
                dx = length * np.cos(head_angle[idx])
                dy = length * -np.sin(head_angle[idx])

                # calculate the preference of mouse heading for one of three targets
                xdist = -tracking_data['head_loc'][start_frame, 0]+tracking_data['barrier_loc'][0][0]
                ydist = -tracking_data['head_loc'][start_frame, 1]+tracking_data['barrier_loc'][0][1]
                bprea = - np.arctan2(ydist, xdist)
                xdist = -tracking_data['head_loc'][start_frame, 0]+tracking_data['barrier_loc'][1][0]
                ydist = -tracking_data['head_loc'][start_frame, 1]+tracking_data['barrier_loc'][1][1]
                bposta = - np.arctan2(ydist, xdist)
                if tracking_data["bod_shelt_dir"][start_frame] < 0: bsa = tracking_data["bod_shelt_dir"][start_frame] + np.pi
                if tracking_data["bod_shelt_dir"][start_frame] > 0:  bsa = tracking_data["bod_shelt_dir"][start_frame] - np.pi

                cosim=[]
                for ang in [bprea,bsa, bposta]:
                    cosim = np.append(cosim,np.cos(ang-head_angle[idx]))
                color = ['green','red','blue']

                ax[i].scatter(mouse[0], mouse[1], color=color[np.argmax(cosim)])
                ax[i].quiver(mouse[0], mouse[1], dx, dy, angles="xy", scale_units="xy", scale=2, color=color[np.argmax(cosim)])

                sum_homings += 1

        Arena(ax=ax[i], shelter_coordinates=tracking_data["shelter_loc"], condition=con, barrier_coordinates=session.barrier_location)
        ax[i].set_title(f"{con} (n={sum_homings})")

    # make a legend for the two types of arrows
    dx = length * np.cos(0)
    dy = length * -np.sin(0)
    ax[len(conditions)].quiver(312, 730, dx, dy, angles="xy", scale_units="xy", scale=2, color="red")
    ax[len(conditions)].text(412,712, title + ' run targets shelter')
    ax[len(conditions)].quiver(312, 630, dx, dy, angles="xy", scale_units="xy", scale=2, color="green")
    ax[len(conditions)].text(412,612, title + ' run targets preflip')
    ax[len(conditions)].quiver(312, 530, dx, dy, angles="xy", scale_units="xy", scale=2, color="blue")
    ax[len(conditions)].text(412,512, title + ' run targets postflip')
    ax[len(conditions)].quiver(312, 430, dx, dy, angles="xy", scale_units="xy", scale=2, color="black")
    ax[len(conditions)].text(412,412,'pre ' + title + ' hdir')
    ax[len(conditions)].axis("off")
    ax[len(conditions)].set_aspect("equal")
    ax[len(conditions)].set_xlim([0, 1024])
    ax[len(conditions)].set_ylim([0, 1024])

    # save figure in session dir
    plt.savefig(os.path.join(session.base_path, session.processed_path, "analyze_behave", str(title+"_heading_angle.png")))
    # save in summary dir
    filename = session.mouse + '_' + date_str + '_' + str(title+"_heading_angle.png")
    plt.savefig(overall_path + '/' + filename)
    if show_plots: plt.show()
    plt.close()

def trajectory_by_target(session, onsets, offsets, head_angle, all_conditions, tracking_data, title, show_plots=False):
    """Plot the trajectory of all homings in each condition, colored by the target in the first 15 cm of the run"""

    # paths for saving in summary dir
    ceph_path, _ = get_computer_specific_paths(session_path = '', return_ceph = True)
    ceph_path = os.path.dirname(ceph_path)
    overall_path = make_directory(os.path.join(ceph_path, 'summary_plots', title + '_plots'))
    match = re.search(r'(\d{4}_\d{2}_\d{2})T', session.file_path) # these three lines could be replaced with session.date if process is rerun
    if match:
        date_str = match.group(1)
        
    conditions = identify_conditions(session)

    if "barrier_pre_flip" in conditions:
        conditions.remove("barrier_present")

    if "shelter_only" in conditions:
        conditions.remove("shelter_present")

    _, ax = plt.subplots(nrows=1, ncols=len(conditions), figsize=(20, 4))

    for i, con in enumerate(conditions):
        sum_homings = 0
        for idx, (onset,offset) in enumerate(zip(onsets,offsets)):
            if np.isnan(onset):
                continue
            trial_condition = all_conditions[idx]
            if isinstance(onset,np.ndarray):
                onset = onset[0].astype(int)

            if np.logical_or(con == trial_condition, con == "all_time"):
                frame_coords = tracking_data["avg_loc"][onset:offset]
                _, start_frame = cum_distance(onset, offset, frame_coords, session.video.pixels_per_cm, 15)

                # calculate the preference of mouse heading for one of three targets - could also use start_frame or the average from start_frame to end_fr
                xdist = -tracking_data['head_loc'][start_frame, 0]+tracking_data['barrier_loc'][0][0]
                ydist = -tracking_data['head_loc'][start_frame, 1]+tracking_data['barrier_loc'][0][1]
                bprea = - np.arctan2(ydist, xdist)
                xdist = -tracking_data['head_loc'][start_frame, 0]+tracking_data['barrier_loc'][1][0]
                ydist = -tracking_data['head_loc'][start_frame, 1]+tracking_data['barrier_loc'][1][1]
                bposta = - np.arctan2(ydist, xdist)
                if tracking_data["bod_shelt_dir"][start_frame] < 0: bsa = tracking_data["bod_shelt_dir"][start_frame] + np.pi
                if tracking_data["bod_shelt_dir"][start_frame] > 0:  bsa = tracking_data["bod_shelt_dir"][start_frame] - np.pi

                cosim=[]
                for ang in [bprea,bsa, bposta]:
                    cosim = np.append(cosim,np.cos(ang-head_angle[idx]))
                color = ['green','red','blue']

                x_loc = tracking_data["head_loc"][onset : offset, 0]
                y_loc = tracking_data["head_loc"][onset : offset, 1]
                ax[i].scatter(x_loc, y_loc, s=3, color=color[np.argmax(cosim)])

        Arena(ax=ax[i], shelter_coordinates=tracking_data["shelter_loc"], condition=con, barrier_coordinates=session.barrier_location)
        ax[i].set_title(f"{con} (n={sum_homings})")

    # save figure in session dir
    plt.savefig(os.path.join(session.base_path, session.processed_path, "analyze_behave", str(title+"_trajectory_by_target.png")))
    # save in summary dir
    filename = session.mouse + '_' + date_str + '_' + str(title+"_trajectory_by_target.png")
    plt.savefig(overall_path + '/' + filename)
    if show_plots: plt.show()
    plt.close()

def trial_speed_hist(session, avg_speed, title, show_plots=False):
    # histogram of homing speed
    # paths for saving in summary dir
    ceph_path, _ = get_computer_specific_paths(session_path = '', return_ceph = True)
    ceph_path = os.path.dirname(ceph_path)
    overall_path = make_directory(os.path.join(ceph_path, 'summary_plots', title + '_plots'))
    match = re.search(r'(\d{4}_\d{2}_\d{2})T', session.file_path) # these three lines could be replaced with session.date if process is rerun
    if match:
        date_str = match.group(1)

    _, ax = plt.subplots(nrows = 1,ncols = 1, figsize = (4,4))

    ax.hist(avg_speed, bins = np.arange(0,200,10))
    ax.set_xlabel('speed (cm/s)')
    ax.set_ylabel('number of homings')

    # save figure in session dir
    plt.savefig(os.path.join(session.base_path, session.processed_path, "analyze_behave", str("hist_speed_of_"+title+".png")))
    # save in summary dir
    filename = session.mouse + '_' + date_str + '_' + str("hist_speed_of_"+title+".png")
    plt.savefig(overall_path + '/' + filename)
    if show_plots: plt.show()
    plt.close()

## ----------------------NON FUNCTIONAL FUNCTIONS

def homing_head_angle_trajectory(session, onsets, offsets, all_conditions, tracking_data, title):
    """Conduct a 2d histrogram normalised to count the probability of starting a homing at a given location"""
    conditions = identify_conditions(session)

    if "barrier_pre_flip" in conditions:
        conditions.remove("barrier_present")

    if "shelter_only" in conditions:
        conditions.remove("shelter_present")

    _, ax = plt.subplots(nrows=1, ncols=len(conditions), figsize=(20, 20))

    max_homing_length = np.amax(offsets-onsets)
    for i, con in enumerate(conditions):
        head_traj = np.zeros((max_homing_length + (2*session.video.fps),1))
        for idx, onset, offset in enumerate(zip(onsets,offsets)):
            trial_condition = all_conditions[idx][0]

            if np.logical_or(con == trial_condition, con == "all_time"):
                head_direction = tracking_data["hdir"][onset - session.video.fps:offset+session.video.fps]
    # plot heatmap of homing head angle trajectories
    # separate homings into conditions
    # extract hdir from 1s before onset through stim duration plus 1s
    # heatmap it