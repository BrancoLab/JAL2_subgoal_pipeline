"""This is a file of functions to support computing escape tuning curves."""

import numpy as np
import re
from loguru import logger
import os

from behave_analysis.analyze.PlaceCells.place_cell_utils import create_centered_bins
from behave_analysis.utils.creating_directories import make_directory


def define_bin_edges(settings, tuning_var):
    """Define bin edges based on settings.tuning_var and settings.tuning_bins."""
    # if tuning_bins is an integer, create that many bins between min and max of the variable

    range_dict = {
        "bird_dist_shelter": (0, 900),
        "frac_route": (0, 1),
        "distance_shelter": (0, 1200),
        "speed": (0, 100),
        "y_pos": (0, 1024)
    }

    if isinstance(settings.ep_bins, int):
        if tuning_var == "2D_position":
            bin_edges = create_centered_bins(bin_size=settings.place_cell_bin_size_pix)
        else:
            if tuning_var in range_dict:
                var_range = range_dict[tuning_var]
                bin_edges = np.linspace(var_range[0], var_range[1], settings.ep_bins + 1)
                bin_edges[-1] = bin_edges[-1] + 1e-10  # add a tiny bit to the last edge to make sure the max value is included in the last bin
            else:
                raise NotImplementedError("Define bin edges for your tuning variable in define_bin_edges function in escape_pattern_utils.py")
    elif isinstance(settings.ep_bins, list):
        bin_edges = settings.ep_bins

    return bin_edges


def residual_neural_matrix(neural_matrix_t1, cond_t1, var2_t1, fr_var2_t2):
    """This function computes the residual neural matrix after removing activity predicted by var2 (e.g. distance to shelter)
    INPUTS:
        neural_matrix_t1: matrix of neural activity already restricted to <ctx1> [neurons x time]
        cond_t1: a vector [of length time] of the condition at each time point in <ctx1>
        var2_t1: vector of length time of binned <var2> (e.g. distance to shelter) np.unique(<var2>) = np.shape(fr_var2_t2)[2]
        fr_var2_t2: firing rates at each binned <var2> in <ctx2> [condition x neuron x bins]
    """

    # initialize variables
    v2_predicted_matrix = np.full_like(neural_matrix_t1, np.nan)
    n_neur = neural_matrix_t1.shape[0]
    n_cond = len(np.unique(cond_t1))

    # 1. make predicted neural matrix
    if var2_t1.ndim > 1:
        if var2_t1.shape[1] == 2: # 2D position!
            # var2_t1 should be zero indexed
            valid_idx = ~np.isnan(var2_t1).any(axis=1)  # only consider time points where we have non-nan values for both dimensions of position
            for n in range(n_neur):
                for c in range(n_cond):
                    u = var2_t1[(cond_t1 == c) & valid_idx, :].astype(int)  # binned <var2> in <ctx1> and condition c
                    v = fr_var2_t2[c, :, :, n]  # firing rates for neuron n at each binned <var2> in <ctx2> and condition c
                    pred = v[u[:, 0], u[:, 1]]
                    v2_predicted_matrix[n, (cond_t1 == c) & valid_idx] = pred
        elif var2_t1.shape[1] == 1: # 1D variable!
            var2_t1 = var2_t1.ravel()  # make sure it's a 1D vector
            # identify non-nan time points in var2_t1
            valid_idx = ~np.isnan(var2_t1)
            for n in range(n_neur):
                for c in range(n_cond):
                    u= var2_t1[(cond_t1 == c) & valid_idx].astype(int)  # binned <var2> in <ctx1> and condition c
                    v = fr_var2_t2[c, n, :]  # firing rates for neuron n at each binned <var2> in <ctx2> and condition c
                    pred = v[u]
                    v2_predicted_matrix[n, (cond_t1 == c) & valid_idx] = pred
        else:
            raise ValueError("Your discretized behavioral variable has too many columns")

    # 2. subtract predicted neural activity from actual neural activity
    v2_residual_matrix = neural_matrix_t1 - v2_predicted_matrix

    return v2_residual_matrix


def homing_escape_onsets(aefizz, escape_pattern_time, spatial_efficiency_threshold=[0.9, 1.05]):
    """This function creates two vectors of onset and offset times for homing and escape periods
    TODO: currently does not filter based on first/second leg, or minimum homing length
    RETURNS:
        ons: vector of onset times in frames for homing and escape periods
        offs: vector of offset times in frames for homing and escape periods
        esc_ons: vector of onset times in frames for escape periods
    """

    ons = []
    offs = []
    esc_ons = []
    trajectory_length = []
    condition = []

    # check if we want to include escapes
    if "escape" in escape_pattern_time:
        # pull out escape onsets and calculate offset estimate based on stimulus duration (assuming 40 fps) - mouse will likely lon gbe in shelter by then
        esc_ons = np.array(check_not_list(aefizz.escape_dict["onset_frames"]))
        not_nan = np.where(~np.isnan(esc_ons))[0]  # if the mouse didn't perform an escape after the stim!
        ons = np.append(ons, esc_ons[not_nan])
        offs = np.append(offs, np.array(check_not_list(aefizz.escape_dict["offset_frames"]))[not_nan])
        trajectory_length = np.append(trajectory_length, np.array(aefizz.escape_dict["trajectory_length"])[not_nan])
        condition = np.append(condition, [c for idx, c in enumerate(aefizz.escape_dict["condition"]) if idx in not_nan])

    if "homing" in escape_pattern_time:
        # pull out homing onsets and offsets
        ons = np.append(ons, check_not_list(aefizz.homing_dict["onset_frames"]))
        offs = np.append(offs, check_not_list(aefizz.homing_dict["offset_frames"]))
        trajectory_length = np.append(trajectory_length, aefizz.homing_dict["trajectory_length"])
        condition = np.append(condition, aefizz.homing_dict["condition"])

    # combine homing and escape onsets and offsets
    sort_idx = np.argsort(ons)
    ons = ons[sort_idx]
    offs = offs[sort_idx]
    trajectory_length = trajectory_length[sort_idx]
    condition = condition[sort_idx]
    cond_list = np.unique(condition)

    # compute spatial efficiency for each run
    optimal_distances = np.zeros(len(ons))
    for idx in range(len(ons)):
        dist_to_shelter = compute_dist_shelt(
            x_pos=np.array([aefizz.video_df["mouse_x_position"].to_numpy()[int(ons[idx])]]),
            y_pos=np.array([aefizz.video_df["mouse_y_position"].to_numpy()[int(ons[idx])]]),
            cond=np.array([condition[idx]]),
            shelter_location=[np.mean([aefizz.session.shelter_location[0][0], aefizz.session.shelter_location[1][0]]), aefizz.session.shelter_location[0][1]],
            barrier_location1=aefizz.session.barrier_location[0] if aefizz.session.barrier_location is not None else None,
            barrier_location2=aefizz.session.barrier_location[1] if aefizz.session.barrier_location is not None else None,
        )
        optimal_distances[idx] = dist_to_shelter[0]

    spatial_efficiency = optimal_distances / trajectory_length

    keepers = np.ones_like(ons, dtype=bool)

    assert not (("correct" in escape_pattern_time) and ("error" in escape_pattern_time)), "You cannot select for both correct and error homings/escapes at the same time"
    assert not (("full" in escape_pattern_time) and ("to_subgoal" in escape_pattern_time)), "You cannot select for both full and to_subgoal homings/escapes at the same time"

    if "correct" in escape_pattern_time:  # homings with spatial efficiency of 0.95-1.05 are considered correct
        keepers = keepers & ((spatial_efficiency > spatial_efficiency_threshold[0]) & (spatial_efficiency < spatial_efficiency_threshold[1]))
    if "error" in escape_pattern_time:
        keepers = keepers & ((spatial_efficiency < spatial_efficiency_threshold[0]) | (spatial_efficiency > spatial_efficiency_threshold[1]))
    if "full" in escape_pattern_time:  # homings that go from the threat zone-ish to the shelter-ish
        starts = np.array([aefizz.video_df["mouse_y_position"].to_numpy()[int(on)] for on in ons])
        ends = np.array([aefizz.video_df["mouse_y_position"].to_numpy()[int(off)] for off in offs])
        keepers = keepers & (starts < 300) & (ends > 800)
    if "to_subgoal" in escape_pattern_time:  # homings that go to the subgoal-ish
        ends = np.array([aefizz.video_df["mouse_y_position"].to_numpy()[int(off)] for off in offs])
        keepers = keepers & (ends > 450) & (ends < 530)
    
    if "not" in escape_pattern_time:
        keepers = ~keepers
    
    if np.sum(keepers) < 5:
        logger.warning(f"{np.sum(keepers)} homing/escape periods does not meet the criteria for {escape_pattern_time}")

    return {
        "ons": ons[keepers],
        "offs": offs[keepers],
        "esc_ons": np.array([e for e in esc_ons if e in ons[keepers]]),
        "condition": condition[keepers],
        "trajectory_length": trajectory_length[keepers],
        "spatial_efficiency": spatial_efficiency[keepers],
    }

def homing_escape_filtering_vector(nframes, onset_dict, xpos, ypos, shelter_location, interpolation_mult=1):
    """This function builds two boolean vectors of length time which are True when the mouse is in homing or escape periods
    It removes any time after shelter entry within each homing
    It uses the array of onsets and offsets created in homing_escape_onsets function
    (this could be only homings, homings+escapes, long homings, etc. depending on context in tuning passed to ComputeEscapeTuning)"""

    homing_vector = np.zeros(nframes, dtype=bool)
    escape_vector = np.zeros(nframes, dtype=bool)

    # iterate over homings
    for on, of in zip(onset_dict["ons"], onset_dict["offs"]):
        on = int(on)
        of = int(of)

        if on in onset_dict["esc_ons"]:
            esc = True
        else:
            esc = False

        # extract mouse position in the run
        this_y = ypos[on:of]
        this_x = xpos[on:of]

        # crop homings at shelter entry
        # find actual length of time until mouse is in shelter
        in_shelt = np.logical_and(
            this_y > shelter_location[0][1],
            np.logical_and(this_x > shelter_location[0][0], this_x < shelter_location[1][0]),
        )
        shelter_entry = np.where(np.diff(in_shelt) > 0)[0][0] + 1 if np.any(np.diff(in_shelt) > 0) else len(in_shelt)
        of = on + shelter_entry

        # do we want to crop homings into first and second leg?

        if interpolation_mult > 1:
            on = on * interpolation_mult
            of = of * interpolation_mult

        homing_vector[on:of] = True
        escape_vector[on:of] = True if esc else False

    return homing_vector, escape_vector

def get_homings_onsets_in_filtered_time(filtering_vector):
    """This function returns the homing onsets that are within the filtered time vector
    while filtered_vector gives you the onsets in recording time, this function returns them in filtered time (e.g. to index into the escape_matrix or escape_tuning discretized var)
    """
    ons = np.where(np.diff(filtering_vector.astype(int)) == 1)[0] + 1  # homing onsets
    offs = np.where(np.diff(filtering_vector.astype(int)) == -1)[0] + 1  # homing offsets
    h_start = np.cumsum(offs - ons)
    h_start = np.concatenate(([0], h_start[:-1]))  # add a leading zero for the onset of the first homing
    return h_start


def select_onset_offsets_in_shift_vector(shift_vector, ons, offs):
    """This function selects homing/escape onsets and offsets that are within the shift vector"""
    # which ones to keep: the start and end of the homing has to inside the chunks defined by shifted_vec
    mask = np.logical_and(shift_vector[ons], shift_vector[offs])
    ons = ons[mask]
    offs = offs[mask]
    # build new filtering vector
    filtering_vector = np.zeros_like(shift_vector)
    for on, off in zip(ons, offs):
        filtering_vector[on:off] = 1

    return filtering_vector.astype(bool)


def homing_escape_boolean_vectors(object, n_frames):
    """This function creates two boolean vectors for homing and escape periods"""
    runs_period = np.zeros(n_frames, dtype=bool)
    for onset, offset in zip(object.onset_frames, object.offset_frames):
        runs_period[int(onset) : int(offset) + 1] = True

    return runs_period


###------------------------COMPUTE BEHAVIORAL VARIABLES----------------------


def create_discretized_behave_var(aefizz, x, y, condition, tuning_var, time_mask_vector=[], interpolation=True, discretize = True):
    """This function returns the discretized behavioral variable of interest
    INPUTS:
        aefizz: AnalyzeEfizz object
        x: mouse x position vector - the variable of interest will be computed based on this and the y position vector
        y: mouse y position vector - the variable of interest will be computed based on this and the x position vector
        condition: vector of string of the name of the condition at each time point
        tuning_var: string defining which behavioral variable to compute
        time_mask_vector: boolean vector defining time periods of interest (e.g. homings)
        bin_edges: edges of bins to discretize variable into (if empty, will be defined based on settings)
    """
    # compute distance to shelter along the shortest path (i.e. around barrier if present)
    if tuning_var in ["distance_shelter"]:
        shelter = [np.mean([aefizz.session.shelter_location[0][0], aefizz.session.shelter_location[1][0]]), aefizz.session.shelter_location[0][1]]
        var = compute_dist_shelt(
            x, y, condition, shelter_location=shelter, 
            barrier_location1=aefizz.session.barrier_location[0] if len(aefizz.session.barrier_location) > 0 else None,
            barrier_location2=aefizz.session.barrier_location[1] if len(aefizz.session.barrier_location) > 1 else None
        )

    # compute distance to first goal (either shelter or subgoal)
    elif tuning_var == "distance_first_goal":
        raise NotImplementedError("distance to first goal not yet implemented")

    # compute bird's eye distance to shelter or first goal (i.e. through barrier if present)
    elif tuning_var in ["bird_dist_shelter"]:
        shelter = [np.mean([aefizz.session.shelter_location[0][0], aefizz.session.shelter_location[1][0]]), aefizz.session.shelter_location[0][1]]
        var = compute_dist_shelt(
            x, y, cond=np.full_like(x, "shelter_only"), shelter_location=shelter, 
            barrier_location1=aefizz.session.barrier_location[0] if len(aefizz.session.barrier_location) > 0 else None, 
            barrier_location2=aefizz.session.barrier_location[1] if len(aefizz.session.barrier_location) > 1 else None
        )
        # cond=np.zeros_like(x) is a hack which forces bird's eye distance, ignoring barrier

    # compute fraction of trajectory
    elif "frac_route" in tuning_var:
        # iterate over trials to compute distance travelled during each
        ons = np.where(np.diff(time_mask_vector.astype(int)) == 1)[0] + 1  # homing onsets
        offs = np.where(np.diff(time_mask_vector.astype(int)) == -1)[0] + 1  # homing offsets
        homie_starts = offs - ons
        first = 0
        var = np.zeros_like(x)
        for hs in homie_starts:
            dd = compute_total_trajectory(x[first : first + hs], y[first : first + hs])
            var[first : first + hs] = dd / np.amax(dd)
            first += hs

    # use speed or y position directly
    elif tuning_var == "speed" or tuning_var == "y_pos":
        if tuning_var == "speed":
            var = aefizz.video_df["speed"].to_numpy()
        elif tuning_var == "y_pos":
            var = y
        # interpolate!
        if interpolation == True:
            # if interpolation is true, time_mask_vector also needs to be interpolated!!
            current_time = np.arange(len(aefizz.video_df["speed"].to_numpy()))
            new_time = np.arange(0, len(aefizz.video_df["speed"].to_numpy()), 1 / aefizz.settings.ep_interpolation_mult)
            var = np.interp(new_time, current_time, var)
        var = var[time_mask_vector] if len(time_mask_vector) > 0 else var

    elif tuning_var == "2D_position":
        var = np.column_stack([x, y])

    elif tuning_var == "Delta_HDIR":
        raise NotImplementedError("Delta HDIR not yet implemented")

    # discretize variable into bins
    if discretize:
        bin_edges = define_bin_edges(aefizz.settings, tuning_var)
        var = discretize_nd(var, bin_edges)

    return var


def compute_total_trajectory(xpos, ypos, start=0, stop=-1):
    # compute cumulative distance travelled at every time point
    distance_travelled = np.zeros_like(xpos)
    all_time = np.arange(len(xpos) + 1)
    used_time = all_time[start:stop]
    for n, i in enumerate(used_time):
        if n > 0:
            dist = np.sqrt((xpos[i] - xpos[i - 1]) ** 2 + (ypos[i] - ypos[i - 1]) ** 2)
            distance_travelled[i] = dist + distance_travelled[i - 1]
    return distance_travelled


def compute_dist_shelt(x_pos, y_pos, cond, shelter_location, barrier_location1, barrier_location2):
    """This function creates a vector of the distance of the mouse to the shelter at any position.
    The distance is computed as the shortest path between mouse and shelter (around barrier, if necessary)
    INPUTS:
        x_pos, y_pos: vector of the x and y position of the mouse at any given time
        cond: vector of strings of the condition the mouse is in at any given time
        shelter_location: (x, y) coordinates of the shelter (X is the middle of the shelter, Y is the top edge of the shelter)
        barrier_location1: (x, y) coordinates of the edge of the barrier in the preflip condition
        barrier_location2: (x, y) coordinates of the edge of the barrier in the postflip condition

    RETURNS:
        dist: a vector of length x_pos of the fistance of the mouse to the shelter.
    """
    # initialize distance vector
    dist = np.zeros((len(x_pos)))

    # for times when the mouse is in the top half of the arena and the barrier is inserted,
    # compute path around barrier: mouse → barrier_edge → shelter
    top_barrier = np.logical_and(cond == "barrier_pre_flip", y_pos < 512)
    if np.sum(top_barrier.astype(int)) > 0:
        dist[top_barrier] = np.sqrt(((x_pos[top_barrier] - barrier_location1[0]) ** 2) + ((y_pos[top_barrier] - barrier_location1[1]) ** 2)) + np.sqrt(
            ((barrier_location1[0] - shelter_location[0]) ** 2) + ((barrier_location1[1] - shelter_location[1]) ** 2)
        )

    top_barrierflip = np.logical_and(cond == "barrier_post_flip", y_pos < 512)
    if np.sum(top_barrierflip.astype(int)) > 0:
        dist[top_barrierflip] = np.sqrt(((x_pos[top_barrierflip] - barrier_location2[0]) ** 2) + ((y_pos[top_barrierflip] - barrier_location2[1]) ** 2)) + np.sqrt(
            ((barrier_location2[0] - shelter_location[0]) ** 2) + ((barrier_location2[1] - shelter_location[1]) ** 2)
        )

    # for all other positions (bottom half or no barrier), use direct distance to shelter
    no_barrier_path = ~(top_barrier | top_barrierflip)
    dist[no_barrier_path] = np.sqrt(((x_pos[no_barrier_path] - shelter_location[0]) ** 2) + ((y_pos[no_barrier_path] - shelter_location[1]) ** 2))

    return dist


def compute_tuning_stat(stat: str, shifted_matrix: np.array, shift0: int, neural_matrix=None, condition=None):
    """
    INPUTS:
        shifted_matrix: a matrix of (shifts,conditions,n_neurons,n_bins) includes the zero shift!
            most commonly this will be data['y_fitted_shift'], but could also be data['fr_shift']
        shift0: which of the shifts of shifted_matrix is the zero shift
        stat: a string defining which statistic we want to compute
                'zscore_peak' - the peak of the zscored trace
                'peak' - the peak of the trace (can find a peak even for very flat curves)
                'peak_to_mean' - the ratio of the peak to the mean firing of the tuning curve (high values for very low firing cells!)
        if stat == 'zscore_peak' need to pass:
            neural_matrix: the original neural matrix used to compute shifted_matrix (neurons x time)
            condition: vector of length time of the condition at each time point used to compute shifted_matrix
    """
    if stat == "peak_to_mean":
        peak = np.nanmax(shifted_matrix, axis=3)
        mean = np.nanmean(shifted_matrix, axis=3)
        shift_stat = np.divide(peak, mean, out=np.zeros_like(peak, dtype=np.float64), where=mean != 0)
    elif stat == "peak":
        shift_stat = np.nanmax(shifted_matrix, axis=3)
    elif stat == "zscore_peak":
        assert (neural_matrix is not None) & (condition is not None), "Need to pass neural_matrix and condition to compute zscore_peak"
        mean_fr = np.zeros((shifted_matrix.shape[1], shifted_matrix.shape[2]))  # condition x neuron
        std_fr = np.zeros((shifted_matrix.shape[1], shifted_matrix.shape[2]))  # condition x neuron
        for c in np.unique(condition):
            mean_fr[int(c), :] = np.nanmean(neural_matrix[:, condition == int(c)], axis=1)
            std_fr[int(c), :] = np.nanstd(neural_matrix[:, condition == int(c)], axis=1)
        # transform shifted_matrix to z-scores using mean and std of original neural matrix, extended to all shifts and bins
        zscored = np.divide(
            shifted_matrix - mean_fr[np.newaxis, :, :, np.newaxis],
            std_fr[np.newaxis, :, :, np.newaxis],
            out=np.zeros_like(shifted_matrix, dtype=np.float64),
            where=std_fr[np.newaxis, :, :, np.newaxis] != 0,
        )
        shift_stat = np.nanmax(zscored, axis=3)
    real_stat = shift_stat[shift0, :, :]
    shift_stat = np.delete(shift_stat, shift0, axis=0)

    return real_stat, shift_stat


# ------------------------------------Linear Shift Stats------------------------------------

def build_shift_vector(ET, full_condition_vector, settings):

    """This function builds a list of shifts and a vector of where to sample the central third of each condition for linear shift statistics
    If settings.escape_pattern_time is 'homing&escape' it makes sure there are enough homings in each central third"""

    # --- 1. Compute condition boundaries (in interpolated frame space) ---
    mult = settings.ep_interpolation_mult
    ttime = len(full_condition_vector) # total number of (interpolated) frames in the session
    cond_start_end = [(np.where(full_condition_vector == i)[0][0], np.where(full_condition_vector == i)[0][-1]) for i in np.unique(full_condition_vector)]

    # --- 2. Define the central third of each condition ---
    # These are the "null" windows: data that won't be shifted past its own condition boundary
    shift_total_size_one_side = (
            (settings.linshift_step * mult * (settings.linshift_step_n / 2)) + (settings.linshift_min_step * mult) + 1
    )# total size of shifts on one side (e.g. 10s step * 3 steps = 30s)
    shift_range = [(a+shift_total_size_one_side, b-shift_total_size_one_side) for (a,b) in cond_start_end]

    # --- 3. Define the shift amounts (one-sided, in interpolated frames) ---
    # e.g. min_step=3s, step=10s → shifts at 3s, 13s, 23s, ... (multiplied by interpolation factor)
    shifts_one_sided = np.arange(
        settings.linshift_min_step * mult,
        settings.linshift_step * mult + ((settings.linshift_step_n / 2) * settings.linshift_step * mult),
        settings.linshift_step * mult,
    )

    # --- 4. If using homing data, adjust central thirds to ensure enough homings fall inside ---
    if ET.escape_pattern_time == "homing&escape":
        all_ons = np.where(np.diff(ET.homing_vector.astype(int)) == 1)[0] + 1  # all homing onset frames

        # -- Adjust central third of each condition if too few homings fall inside ---
        for i, (a, b) in enumerate(shift_range):
            if np.sum(np.logical_and(all_ons > a, all_ons < b)) < settings.ep_linshift_min_homings:
                # Redefine: start at the 1/3 mark of homings in this condition, span 1/3 of the condition length
                homings_this_cond = all_ons[np.logical_and(all_ons > cond_start_end[i][0], all_ons < cond_start_end[i][1])]
                a = homings_this_cond[int(np.round(len(homings_this_cond) / 3))] - 1
                b = a + int((cond_start_end[i][1] - cond_start_end[i][0]) / 3)

                # Clamp: don't let the window start too close to 0 (need room for left shifts)
                if a < (cond_start_end[i][0] + np.amax(shifts_one_sided)):
                    b = b + (np.amax(shifts_one_sided) - a)
                    a = a + (np.amax(shifts_one_sided) - a)

                # Clamp: don't let the window end too close to the barrier onset (need room for right shifts)
                if b > (cond_start_end[i][1] - np.amax(shifts_one_sided)):
                    a = a - (b - (cond_start_end[i][1] - np.amax(shifts_one_sided)))
                    b = b - (b - (cond_start_end[i][1] - np.amax(shifts_one_sided)))

                shift_range[i] = (a, b)
            print(f"Number of homings in {ET.all_conditions[i]} centre chunk: " + str(np.sum(np.logical_and(all_ons > a, all_ons < b))))

    # --- 5. Build the boolean shift_vector marking the central chunk of each condition ---
    shift_vector = np.zeros(ttime)
    for (a, b) in shift_range:
        shift_vector[int(a):int(b)] = 1
    shift_vector = shift_vector.astype(bool)

    # --- 6. Trim shifts that would go out of bounds ---
    shifts_left = shifts_one_sided[shifts_one_sided < shift_range[0][0]]  # left shifts that stay within session start
    shifts_right = shifts_one_sided[(shifts_one_sided + shift_range[-1][1]) < ttime]  # right shifts that stay within session end

    # --- 7. Combine into symmetric shifts (negative = past, positive = future) plus zero ---
    shifts = np.sort(np.hstack((0, shifts_right, -shifts_left)))

    return shifts, shift_vector

# ------------------------------------Helper functions------------------------------------

def discretize(var, bins):
    """Bin the var using bins,
    INPUTS:
        var: vector of continuous variable to be binned
        bins: vector of bin edges"""
    disc_var = np.digitize(var, bins)
    shifted_disc_var = (disc_var - 1).astype(float)
    shifted_disc_var[disc_var == 0] = np.nan  # Handle values below the first bin
    shifted_disc_var[disc_var >= len(bins)] = np.nan  # Handle values above the last bin
    return shifted_disc_var


def discretize_nd(var, bins):
    """Bin the var using bins for each dimension,
    INPUTS:
        var: array of shape (time, dimensions) of continuous variables to be binned
        bins: vector of bin edges"""
    if var.ndim == 1:
        var = var[:, np.newaxis]  # add a dimension if var is 1D
    disc_var = np.zeros_like(var, dtype=float)
    for dim in range(var.shape[1]):
        disc_var[:, dim] = discretize(var[:, dim], bins)
    # extend nan to whole row if any dimension is out of bounds
    disc_var[np.any(np.isnan(disc_var), axis=1)] = np.nan
    return disc_var


def check_not_list(var):
    if len(var) == 0:
        return var
    if np.logical_or(isinstance(var[0], list), isinstance(var[0], np.ndarray)):
        var = [x[0] for x in var]
    return var


def parse_residual_string(s):
    """Parse a residual string of the form
    'residual: <var1> in <context1> - <var2> in <context2>'
    and return var1, context1, var2, context2
    """
    # remove prefix ("residual:" or "TunED:")
    s = s.split(":", 1)[1].strip()

    # normalize dash-like unicode to ASCII hyphen
    s = re.sub(r"[\u2010-\u2015\u2212]", "-", s)

    # split around the first hyphen
    if "-" not in s:
        raise ValueError("Residual string must contain a '-'! The format is 'residual: <var1> in <context1> - <var2> in <context2>'")

    left, right = s.split("-", 1)
    left = left.strip()
    right = right.strip()

    left_var, left_ctx = parse_side(left)
    right_var, right_ctx = parse_side(right)

    return left_var, left_ctx, right_var, right_ctx


def parse_side(side):
    """Parse a string of the form '<var> in <context>' and return var, context"""
    if " in " not in side:
        raise ValueError("The tuning string must be of the form '<var> in <context>'")
    var, ctx = side.split(" in ", 1)
    return var.strip(), ctx.strip()

