"""This is a file of functions to support computing escape tuning curves."""

import numpy as np
from loguru import logger

from settings.settings_analyze_efizz import Settings_ae as settings

def define_bin_edges(settings):
    """Define bin edges based on settings.tuning_var and settings.tuning_bins."""
    # if tuning_bins is an integer, create that many bins between min and max of the variable
    if isinstance(settings.escape_tuning_bins, int):
        if settings.escape_tuning_var == "bird_dist_shelter":
            bin_edges = np.linspace(0, 925, settings.escape_tuning_bins + 1)
        elif settings.escape_tuning_var == "escape":
            bin_edges = np.append(np.arange(0, 1, 1 / settings.escape_tuning_bins), 1 + 1e-10)
        else: 
            bin_edges = []
    elif isinstance(settings.escape_tuning_bins, list):
        bin_edges = settings.escape_tuning_bins

    return bin_edges

def homing_escape_onsets(aefizz):
    """This function creates two vectors of onset and offset times for homing and escape periods
    RETURNS:
        ons: vector of onset times in frames for homing and escape periods
        offs: vector of offset times in frames for homing and escape periods
        esc_ons: vector of onset times in frames for escape periods
    """
    # pull out escape onsets and calculate offset estimate based on stimulus duration (assuming 40 fps) - mouse will likely lon gbe in shelter by then
    esc_ons = check_not_list(aefizz.session.audio.onset_frames)
    st = [x * 40 for x in check_not_list(aefizz.session.audio.stimulus_durations)]
    esc_offs = (np.add(esc_ons, st)).astype(int)

    # add escapes to homings onsets and offsets
    ons = np.sort(np.append(check_not_list(aefizz.homings_object.onset_frames), esc_ons))
    offs = np.sort(np.append(check_not_list(aefizz.homings_object.offset_frames), esc_offs))

    return ons, offs, esc_ons

def select_onset_offsets_in_shift_vector(ET, shift_vector):
    """This function selects homing/escape onsets and offsets that are within the shift vector"""

    logger.warning("Debug this function to make sure it is working correctly!")

    ons = np.where(np.diff(ET.homing_vector.astype(int)) == 1)[0] + 1  # homing onsets
    offs = np.where(np.diff(ET.homing_vector.astype(int)) == -1)[0] + 1  # homing offsets
    esc_ons = np.where(np.diff(ET.escape_vector.astype(int)) == 1)[0] + 1  # escape onsets
    # which ones to keep: the start and end of the homing has to inside the chunks defined by shifted_vec
    mask = np.logical_and(shift_vector[ons], shift_vector[offs])
    ons = ons[mask]
    offs = offs[mask]
    esc_ons = [x for x in esc_ons if x in ons]
    # build new filtering vector
    filtering_vector = np.zeros_like(ET.homing_vector)
    for on, off in zip(ons, offs):
        filtering_vector[on:off] = 1
    
    return filtering_vector.astype(bool)

def create_discretized_behave_var(aefizz, ET, x, y, condition, homing_vector):
    """This function returns the discretized behavioral variable of interest
    INPUTS:
        aefizz: AnalyzeEfizz object
        ET: EscapeTuning object
        x: mouse x position vector
        y: mouse y position vector
    """
    # compute distance to shelter along the shortest path (i.e. around barrier if present)
    if settings.escape_tuning_var in ["distance_shelter"]:
        var = compute_dist_shelt(x, y, condition, aefizz.session)
    
    # compute distance to first goal (either shelter or subgoal)
    elif settings.escape_tuning_var == "distance_first_goal":
        raise NotImplementedError("distance to first goal not yet implemented")

    # compute bird's eye distance to shelter or first goal (i.e. through barrier if present)
    elif settings.escape_tuning_var in ["bird_dist_shelter"]:
        var = compute_dist_shelt(x, y, 
                                 cond=np.zeros_like(x), # this is a hack which forces bird's eye distance, ignoring barrier
                                 session=aefizz.session)

    # use y position directly
    elif settings.escape_tuning_var == "y_pos":    
        var = y
        if (not ET.bin_edges) & isinstance(settings.tuning_bins, int):
            ET.bin_edges = np.arange(np.amin(var), np.amax(var), np.amax(var) / settings.tuning_bins)
    
    # compute fraction of escape trajectory
    elif "escape" in settings.escape_tuning_var:
        # iterate over escape trials to compute distance travelled during each escape
        ons = np.where(np.diff(homing_vector.astype(int)) == 1)[0] + 1  # homing onsets
        offs = np.where(np.diff(homing_vector.astype(int)) == -1)[0] + 1  # homing offsets
        homie_starts = offs-ons
        first = 0
        var = np.zeros_like(x)
        for hs in homie_starts:
            dd = compute_escape_trajectory(x[first:first+hs], y[first:first+hs])
            var[first:first+hs] = dd / np.amax(dd)
            first += hs
    
    # use speed directly
    elif settings.escape_tuning_var == "speed":
        current_time = np.arange(len(aefizz.video_df["speed"].to_numpy()))
        new_time = np.arange(0, len(aefizz.video_df["speed"].to_numpy()), 1/settings.escape_pattern_interpolation_mult)
        var = np.interp(new_time, current_time, aefizz.video_df["speed"].to_numpy())
        if (not ET.bin_edges) & isinstance(settings.tuning_bins, int):
            ET.bin_edges = np.arange(0, np.amax(var), 1) 

    # discretize variable into bins
    discretized_var = discretize(var, ET.bin_edges)

    return discretized_var

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
    # extract the location of behavioral variables
    shelter = [
        np.mean([session.shelter_location[0][0], session.shelter_location[1][0]]),
        session.shelter_location[0][1],
    ]
    bar1 = session.barrier_location[0]
    bar2 = session.barrier_location[1]
    
    # set mouse distance to zero
    dist = np.zeros((len(x_pos)))

    # for times when the mouse is in the top half of the arena and the barrier is inserted, add the distance to the barrier edge
    top_barrier = np.logical_and(cond == 1, y_pos < 512)
    dist[top_barrier] = np.sqrt(
        ((x_pos[top_barrier] - bar1[0]) ** 2) + ((y_pos[top_barrier] - bar1[1]) ** 2)
    )
    top_barrierflip = np.logical_and(cond == 2, y_pos < 512)
    dist[top_barrierflip] = np.sqrt(
        ((x_pos[top_barrierflip] - bar2[0]) ** 2)
        + ((y_pos[top_barrierflip] - bar2[1]) ** 2)
    )

    # add the distance of the mouse to shelter
    dist = dist + np.sqrt(((x_pos - shelter[0]) ** 2) + ((y_pos - shelter[1]) ** 2))
    return dist

# ------------------------------------Linear Shift Stats------------------------------------

def build_shift_vector(aefizz, ET):
    """This function builds a list of shifts and a vector of where to sample the central third of each condition for linear shift statistics
    If settings.escape_pattern_time is 'homing + escape' it makes sure there are enough homings in each central third"""

    mult = settings.escape_pattern_interpolation_mult
    ttime = len(aefizz.video_df) * mult  # total time after interpolation
    # the end of the shelter_only condition
    shelter = np.where(aefizz.video_df["barrier_present"].to_numpy() == True)[0][0] * mult
    # the end of the barrier condition
    bar_in = np.where(aefizz.video_df["barrier_flipped"].to_numpy() == True)[0][0] * mult
    # the central third of the shelter condition
    mid_shelter = [int(shelter / 3), int((shelter / 3) * 2)] 
    # the central third of the barrier condition
    mid_bar = [int(shelter + ((bar_in - shelter) / 3)), int(shelter + (((bar_in - shelter) / 3) * 2))]
    # the central third of the flipped barrier condition
    mid_flip = [int(bar_in + ((ttime - bar_in) / 3)), int(bar_in + (((ttime - bar_in) / 3) * 2))]

    # define shifts based on settings (in seconds, needs to be doubled to shift into both past and future)
    # NB: have a min step of 3 seconds, and then steps of 10s, not sure why
    shifts_one_sided = np.arange(settings.ep_linshift_min_step * mult, 
                                 settings.ep_linshift_step * mult + ((settings.ep_linshift_step_n / 2) * settings.ep_linshift_step * mult), 
                                 settings.ep_linshift_step * mult)

    if settings.escape_pattern_time == "homing + escape":
        # check that this gives us a minimum number of homings/escapes
        all_ons = np.where(np.diff(ET.homing_vector.astype(int)) == 1)[0] + 1  # homing onsets
        if np.sum(np.logical_and(all_ons > mid_shelter[0], all_ons < mid_shelter[1])) < settings.ep_linshift_min_homings:
            mid_shelter = [
                all_ons[int(np.round(len(all_ons[all_ons < shelter]) / 3))] - 1,  # starting a third of the way into the homings
                (all_ons[int(np.round(len(all_ons[all_ons < shelter]) / 3))] - 1) + int(shelter / 3),
            ]
            if mid_shelter[0] < np.amax(shifts_one_sided):
                mid_shelter = [x + (np.amax(shifts_one_sided) - mid_shelter[0]) for x in mid_shelter]
            if mid_shelter[1] > (shelter - np.amax(shifts_one_sided)):
                mid_shelter = [x - (mid_shelter[1] - (shelter - np.amax(shifts_one_sided))) for x in mid_shelter]
            print("Number of homings in shelter_only centre chunk: " + str(np.sum(np.logical_and(all_ons > mid_shelter[0], all_ons < mid_shelter[1]))))
        if np.sum(np.logical_and(all_ons > mid_bar[0], all_ons < mid_bar[1])) < settings.ep_linshift_min_homings:
            h_bar = all_ons[np.logical_and(all_ons > shelter, all_ons < bar_in)]
            mid_bar = [h_bar[int(np.round(len(h_bar) / 3))] - 1, int(shelter + h_bar[int(np.round(len(h_bar) / 3))] - 1)]
            if mid_bar[0] < (shelter + np.amax(shifts_one_sided)):
                mid_bar = [x + ((shelter + np.amax(shifts_one_sided)) - mid_bar[0]) for x in mid_bar]
            if mid_bar[1] > (bar_in - np.amax(shifts_one_sided)):
                mid_bar = [bar_in - np.amax(shifts_one_sided) - ((bar_in - shelter) / 3), bar_in - np.amax(shifts_one_sided)]
            print("Number of homings in barrier centre chunk: " + str(np.sum(np.logical_and(all_ons > mid_bar[0], all_ons < mid_bar[1]))))
        if np.sum(np.logical_and(all_ons > mid_flip[0], all_ons < mid_flip[1])) < settings.ep_linshift_min_homings:
            h_flip = all_ons[np.logical_and(all_ons > bar_in, all_ons < ttime)]
            mid_flip = [h_flip[int(np.round(len(h_flip) / 3))] - 1, int(bar_in + h_flip[int(np.round(len(h_flip) / 3))] - 1)]
            if mid_flip[0] < (bar_in + np.amax(shifts_one_sided)):
                mid_flip = [x + ((bar_in + np.amax(shifts_one_sided)) - mid_flip[0]) for x in mid_flip]
            if mid_flip[1] > (len(ttime) - np.amax(shifts_one_sided)):
                mid_flip = [ttime - np.amax(shifts_one_sided) - ((ttime - bar_in) / 3), ttime - np.amax(shifts_one_sided)]
            print("Number of homings in flipped barrier centre chunk: " + str(np.sum(np.logical_and(all_ons > mid_flip[0], all_ons < mid_flip[1]))))

    # build the vector that gives us the central chunk of each conditon
    shift_vector = np.zeros(ttime)
    shift_vector[int(mid_shelter[0]) : int(mid_shelter[1])] = 1
    shift_vector[int(mid_bar[0]) : int(mid_bar[1])] = 1
    shift_vector[int(mid_flip[0]) : int(mid_flip[1])] = 1
    shift_vector = shift_vector.astype(bool)

    # make sure we're not shifting out of range
    shifts_left = shifts_one_sided[shifts_one_sided < mid_shelter[0]]
    shifts_right = shifts_one_sided[(shifts_one_sided + mid_flip[1]) < ttime]

    # add shift of zero and double them so we go in both directions
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
    shifted_disc_var[disc_var >= len(bins)] = np.nan  # Handle values above the last bin
    return shifted_disc_var

def check_not_list(var):
    if np.logical_or(isinstance(var[0], list),
                     isinstance(var[0], np.ndarray)):
        var = [x[0] for x in var]
    return var