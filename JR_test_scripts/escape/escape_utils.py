import os
import polars as pl
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from numba import njit
import speedystats as ss

from behave_analysis.process.process import Process
from behave_analysis.utils.data_loading import load_or_extract_homings

###------------------------DATA LOADING----------------------

def load(exp):
    """Load necessary data:
    session object, frame by cluster matrix of neural data and behavioral variables"""
    # load session
    session = Process(exp).load_session()
    base_path = os.path.join(session.base_path, session.processed_path)

    # spikeys
    # spike_data = pl.read_csv(os.path.join(base_path, "good_spike_data.csv"))

    # matrix
    frame_by_cluster_matrix = np.load(
        os.path.join(session.base_path, session.processed_path)
        + "\\"
        + "frame_by_good_cluster_matrix.npy"
    )

    # behavior
    video_df = pl.read_csv(os.path.join(base_path, "full_video_dataframe.csv"))
    behave = video_df["speed"].to_numpy()
    y_pos = video_df["mouse_y_position"].to_numpy()
    x_pos = video_df["mouse_x_position"].to_numpy()
    bar = video_df["barrier_present"].to_numpy()
    barflip = video_df["barrier_flipped"].to_numpy()
    escape = video_df["EscapePeriod"].to_numpy()
    outofshelter = video_df["OutofshelterIdx"].to_numpy()

    return session, frame_by_cluster_matrix, behave, y_pos, x_pos, bar, barflip, escape, outofshelter

def load_homing(session, n_frames):
    """Load homing onset and offset frames, and create homing bool"""
    # homing object
    homings = load_or_extract_homings(session)

    # homing bool
    homing_bool = np.zeros(n_frames, dtype=bool)
    onset_frames = homings.onset_frames
    offset_frames = homings.offset_frames
    for onset, offset in zip(onset_frames, offset_frames):
        homing_bool[onset - 1 : offset - 1] = True

    return homings.onset_frames, homings.offset_frames, homing_bool

###------------------------COMPUTE BEHAVIORAL VARIABLES----------------------

def compute_escape_trajectory(xpos, ypos, start = 0, stop = -1):
    # compute cumulative distance travelled at every time point
    distance_travelled = np.zeros_like(xpos)
    all_time = np.arange(len(xpos)+1)
    used_time = all_time[start:stop]
    for n, i in enumerate(used_time):
        if n > 0:
            dist = np.sqrt((xpos[i] - xpos[i - 1]) ** 2 + (ypos[i] - ypos[i - 1]) ** 2)
            distance_travelled[i] = dist + distance_travelled[i-1]
    return distance_travelled


def compute_dist_shelt(x_pos, y_pos, cond, session):
    """This function creates a vector of the distance of the mouse to the shelter at any position.
    The distance is computed as the shortest path between mouse and shelter (around barrier, if necessary)
    INPUTS:
        x_pos, y_pos: vector of the x and y position of the mouse at any given time
        cond: vector of the condition the mouse is in at any given time (0 for shelter_only, 1 for barrier, 2 for flipped_barrier)
        session: session object

    RETURNS:
        dist: a vector of length x_pos of the fistance of the mouse to the shelter.
    """
    dist = np.zeros((len(x_pos)))
    shelter = [
        np.mean([session.shelter_location[0][0], session.shelter_location[1][0]]),
        session.shelter_location[0][1],
    ]
    bar1 = session.barrier_location[0]
    bar2 = session.barrier_location[1]
    # measure the distance of the mouse to a point in the top half of arena
    top_barrier = np.logical_and(cond == 1, y_pos < 512)
    dist[top_barrier] = np.sqrt(
        ((x_pos[top_barrier] - bar1[0]) ** 2) + ((y_pos[top_barrier] - bar1[1]) ** 2)
    )
    top_barrierflip = np.logical_and(cond == 2, y_pos < 512)
    dist[top_barrierflip] = np.sqrt(
        ((x_pos[top_barrierflip] - bar2[0]) ** 2)
        + ((y_pos[top_barrierflip] - bar2[1]) ** 2)
    )
    # measure the distance of the mouse to shelt in the bottom half of arena
    dist = dist + np.sqrt(((x_pos - shelter[0]) ** 2) + ((y_pos - shelter[1]) ** 2))
    return dist

def compute_dist_first_goal(x_pos, y_pos, cond, session):
    """This function creates a vector of the distance of the mouse to the first goal at any position.
    In shelter_only, the first goal is the shelter, but in any barrier condition the first goal is the barrier edge.
    The distance is computed as the shortest path between mouse and the first goal.
    INPUTS:
        x_pos, y_pos: vector of the x and y position of the mouse at any given time
        cond: vector of the condition the mouse is in at any given time (0 for shelter_only, 1 for barrier, 2 for flipped_barrier)
        session: session object

    RETURNS:
        dist: a vector of length x_pos of the fistance of the mouse to the first goal.
    """
    dist = np.zeros((len(x_pos)))
    shelter = [
        np.mean([session.shelter_location[0][0], session.shelter_location[1][0]]),
        session.shelter_location[0][1],
    ]
    bar1 = session.barrier_location[0]
    bar2 = session.barrier_location[1]
    # in barrier conditions, measure distance to barrier (for now we're ignoring which side of the barrier the mouse is on)
    top_barrier = cond == 1
    dist[top_barrier] = np.sqrt(
        ((x_pos[top_barrier] - bar1[0]) ** 2) + ((y_pos[top_barrier] - bar1[1]) ** 2)
    )
    top_barrierflip = cond == 2
    dist[top_barrierflip] = np.sqrt(
        ((x_pos[top_barrierflip] - bar2[0]) ** 2)
        + ((y_pos[top_barrierflip] - bar2[1]) ** 2)
    )
    # shelter only, compute distance to shelter
    shelter_only = cond == 0
    dist[shelter_only] = np.sqrt(((x_pos[shelter_only] - shelter[0]) ** 2) + ((y_pos[shelter_only] - shelter[1]) ** 2))
    return dist

###------------------------PROCESS DATA----------------------

def compress_vars(var, neural_matrix):
    """This function transforms the x-axis of the data from time into a variable of choice (e.g. speed, position, distance to shelter)"""
    # pos is the variable we're basing the compression on
    # neural_matrix is getting compressed with it
    for i, neural_activity in enumerate(neural_matrix):
        # Step 1: Identify change points
        change_points = (
            np.where(np.diff(var) != 0)[0] + 1
        )  # Indices where position changes
        change_points = np.insert(
            change_points, 0, 0
        )  # Include the start of the first segment
        change_points = np.append(
            change_points, len(var)
        )  # Include the end of the last segment

        # Step 2: Compress position and neural activity
        compressed_pos = [var[start] for start in change_points[:-1]]
        compressed_activity = [
            neural_activity[start:end].mean()  # Example: mean activity for each segment
            for start, end in zip(change_points[:-1], change_points[1:])
        ]

        # Outputs
        if i == 0:
            new_pos = np.array(compressed_pos)
            new_activity = np.array(compressed_activity)
        else:
            new_activity = np.vstack((new_activity, compressed_activity))
    return new_activity, new_pos

def discretize_x_axis(var, bin_size=10):
    """Bin the x-axis of the neural data by a variable of choice (e.g. speed, position, distance to shelter)"""
    bins = np.arange(0, np.amax(var), bin_size)
    disc_var = np.digitize(var, bins)
    return disc_var

def firing_by_bin_median_np(var, neural_activity, nbins, remove_empty=False):
    """For each bin of a variable, calculate the median neural activity.
    remove_empty: if True remove bins with no behavioral data.
    THIS VARIANT USES THE MEDIAN
     SLOW!!!
      """
    # from scipy.stats import mode
    angles_firing = np.full(nbins, np.nan)  # Start with NaN to handle empty bins
    for i in range(nbins):
        mask = (var == i)  # Find data points in the current bin
        if np.any(mask):  # Check if the bin has any data
            # angles_firing[i] = mode(neural_activity[mask], nan_policy='omit')[0][0]
            angles_firing[i] = np.median(neural_activity[mask])
    if remove_empty:
        angles_firing = angles_firing[~np.isnan(angles_firing)]  # Remove empty bins
    else:
        angles_firing[np.isnan(angles_firing)] = 0
    return angles_firing

def firing_by_bin_median_ss(var, neural_activity, nbins, remove_empty=False):
    """For each bin of a variable, calculate the median neural activity.
    remove_empty: if True remove bins with no behavioral data.
    THIS VARIANT USES THE MEDIAN
    speedystats!!!
    """
    angles_firing = np.full(nbins, np.nan)  # Start with NaN to handle empty bins
    for i in range(nbins):
        mask = (var == i)  # Find data points in the current bin
        if np.any(mask):  # Check if the bin has any data
            angles_firing[i] = ss.median(neural_activity[mask])
    if remove_empty:
        angles_firing = angles_firing[~np.isnan(angles_firing)]  # Remove empty bins
    else:
        angles_firing[np.isnan(angles_firing)] = 0
    return angles_firing

@njit
def firing_by_bin_median_numba(var, neural_activity, nbins, remove_empty=False):
    """Compute the median firing rate per bin using Numba."""
    var = np.asarray(var)
    neural_activity = np.asarray(neural_activity)

    # Step 1: Pre-allocate a large 2D array to store values
    max_entries = len(var)  # Worst case: all data points in one bin
    bin_storage = np.full((nbins, max_entries), np.nan)  # Fill with NaNs
    bin_counts = np.zeros(nbins, dtype=np.int32)  # Track how many entries in each bin

    # Step 2: Assign values to bins
    for i in range(len(var)):
        bin_idx = var[i]
        if 0 <= bin_idx < nbins:
            count = bin_counts[bin_idx]  # Current count in bin
            bin_storage[bin_idx, count] = neural_activity[i]  # Store value
            bin_counts[bin_idx] += 1  # Increment bin count

    # Step 3: Compute medians
    angles_firing = np.full(nbins, np.nan)
    for i in range(nbins):
        if bin_counts[i] > 0:
            angles_firing[i] = np.median(bin_storage[i, :bin_counts[i]])  # Take median of non-NaN values

    # Step 4: Handle empty bins
    if remove_empty:
        return angles_firing[~np.isnan(angles_firing)]  # Remove NaNs
    else:
        angles_firing[np.isnan(angles_firing)] = 0  # Set empty bins to 0
        return angles_firing

def firing_by_bin_median_pandas(var, neural_activity, nbins, remove_empty=False):
    """For each bin of a variable, calculate the median neural activity.
    remove_empty: if True remove bins with no behavioral data.
    THIS VARIANT USES THE MEDIAN"""
    
    df = pd.DataFrame({'var': var, 'neural_activity': neural_activity})
    grouped = df.groupby('var')['neural_activity'].median()
    
    # Create an output array with NaNs
    angles_firing = np.full(nbins, np.nan)
    
    # Assign the computed medians to their corresponding bins
    angles_firing[grouped.index] = grouped.values
    
    if remove_empty:
        angles_firing = angles_firing[~np.isnan(angles_firing)]
    else:
        angles_firing[np.isnan(angles_firing)] = 0

    return angles_firing

def firing_by_bin(var, neural_activity, nbins, remove_empty = False):
    """For each bin of a variable of choice (e.g. speed, position, distance to shelter) what is the mean enural activity
    remove_empty: if True remove bins with no behavioral data (and therefore no firing data)
    THIS FUNCTION USES THE MEAN"""
    angles_firing = np.full(nbins, np.nan)
    unique_groups, group_counts = np.unique(var, return_counts=True)
    # mean firing
    group_sums = np.bincount(var, weights=neural_activity)
    angles_firing[unique_groups] = group_sums[unique_groups] / group_counts
    if remove_empty:
        angles_firing = angles_firing[unique_groups]
    return angles_firing

def smoothed_firing_by_bin(var, neural_activity, nbins):
    """This function is an alternative for interpolation.
     It creates teeny tiny bins and computes the time in each bin as well as the activity in each bin and then divides the activity by the time in each bin"""
    # TODO: This function is not working yet

    # neural_activity = escape_matrix[0,:]
    # nbins = int(np.amax(esc_var+1))
    bin_occupancy = np.zeros(nbins)
    bin_sum_activity = np.zeros(nbins)
    unique_groups, group_counts = np.unique(var, return_counts=True)
    group_sums = np.bincount(var, weights=neural_activity)
    angles_firing[unique_groups] = group_sums[unique_groups] / group_counts
    return angles_firing

def check_not_list(var):
    if np.logical_or(isinstance(var[0], list),
                     isinstance(var[0], np.ndarray)):
        var = [x[0] for x in var]
    return var

def smooth_firing_by_bin_by_trial(matrix, fr_all_time, method, filtering = 'savgol'):
    """Take the matrix of firing by bin on each trial and/or across all time and smooth it
    INPUTS:
        fr_all_time: is a vector of firing x bins across all time
        matrix: is a matrix of trials x bins of firing
        method: which one do we want to smooth and use to create the vector of firing by bin
        """

    smooth_test = []

    if len(matrix) > 0:
        # remove trials that are all nan
        all_nan_rows = np.all(np.isnan(matrix), axis=1)
        matrix = matrix[~all_nan_rows,:]
        smooth_test = np.full_like(matrix, np.nan)
        for sidx, sxm in enumerate(matrix):
            this_line = np.full_like(sxm, np.nan)
            where_nan = np.isnan(sxm)
            if filtering == 'gaussian':
                smoothed = gaussian_filter1d(sxm[~where_nan], 2, mode = 'nearest')
                this_line[~where_nan] = smoothed * (np.mean(sxm[~where_nan]) / np.mean(smoothed))
            if filtering == 'savgol':
                this_line[~where_nan] = savgol_filter(sxm[~where_nan], window_length=11, polyorder=3)
            smooth_test[sidx,:] = this_line

    # extract firing by bin for this neuron in this condition
    if method == 'across_trials': # only works if len(mat_by_cond) > 0:
        # alternative: obtain firing rates by taking median across trial
        all_nan_cols = np.all(np.isnan(smooth_test), axis=0)
        smoothed_firing_rates = np.full(len(all_nan_cols), np.nan)
        smoothed_firing_rates[~all_nan_cols] = np.nanmedian(smooth_test[:,~all_nan_cols], axis = 0)
    elif method == 'across_all_time':
        # from scipy.ndimage import gaussian_filter1d
        sigma = 3.0  # Standard deviation of the Gaussian kernel
        smoothed_firing_rates = np.full_like(fr_all_time, np.nan)
        smoothed_firing_rates[~np.isnan(fr_all_time)] = gaussian_filter1d(fr_all_time[~np.isnan(fr_all_time)], sigma)

    # make firing rates positive
    if np.nanmin(smoothed_firing_rates) < 0:
        shift_constant = abs(np.nanmin(smoothed_firing_rates))+ 1e-6 # Add a small epsilon to avoid exact zero
        smoothed_firing_rates = smoothed_firing_rates + shift_constant
    else:
        shift_constant = 0
    distances = np.arange(len(smoothed_firing_rates))

    return distances, smoothed_firing_rates, shift_constant, smooth_test