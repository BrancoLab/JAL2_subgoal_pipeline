import numpy as np
from scipy.stats import zscore

from JR_test_scripts.escape.functions.escape_utils import (
    check_not_list,
    compute_dist_shelt,
    compute_escape_trajectory,
    discretize_x_axis,
    compute_dist_first_goal,
)


def extract_homing_and_escape_periods(
    session, frame_by_cluster_matrix, behave, y_pos, x_pos, bar, barflip, compression_var, ons, offs, shifted_vec=[], interpolation=True, no_stationary=False, return_escape=False, zscore=True, bins=[]
):
    """For a given session, extract the time around escapes and homing periods in neural data and a behavioral variable of interest (compression_var)
    INPUTS:
        session: session object, used to get the escape onsets and offsets
        frame_by_cluster_matrix: is a matrix of neural data, time x neurons
        behave, y_pos, x_pos: the vectors of speed, y and x position of the mouse
        bar, barflip: the vectors of barrier and flipped barrier
        compression_var: variable to compress the data into bins (full_distance_shelter, y_pos, escape, speed, distance_shelter, distance_first_goal, escape_shelter, escape_first_goal)
        ons, offs: the vectors of onsets and offsets of homings
        interpolation: if True, interpolate over time to double the number of samples
        no_stationary: if True, exclude stationary periods from the analysis (i.e. when speed < 0.5 cm/s) # TODO might not work
        bins: could be an emty list, an integer (number of bins) or a list of bin edges
    RETURNS:
        esc_var: the behavioral variable of interest, discretized into bins
        escape_matrix: a matrix of neural data, neurons x time
        cond: a vector of length time indicating what experimental condition the homing/escape was in
        h_start: the start time of the homings or escapes, duration of homing/escape is cropped to when the mouse reaches shelter
        esc_start: the start time of the escapes only, duration of homing/escape is cropped to when the mouse reaches shelter
    """

    # extract the time around escapes
    ons, offs, esc_ons = homing_escape_onsets(session, ons, offs)
    if len(shifted_vec) > 0:
        # which ones to keep
        mask = np.logical_and(shifted_vec[ons], shifted_vec[offs])
        ons = ons[mask]
        offs = offs[mask]
        esc_ons = [x for x in esc_ons if x in ons]
        # find the new timepoints
        on_vec = np.zeros_like(shifted_vec)
        on_vec[ons] = 1
        on_vec = on_vec[shifted_vec]
        ons = np.where(on_vec == 1)[0]
        off_vec = np.zeros_like(shifted_vec)
        off_vec[offs] = 1
        off_vec = off_vec[shifted_vec]
        offs = np.where(off_vec == 1)[0]
        eon_vec = np.zeros_like(shifted_vec)
        eon_vec[esc_ons] = 1
        eon_vec = eon_vec[shifted_vec]
        esc_ons = np.where(eon_vec == 1)[0]

    # initialize variables:
    # start is the list of all start times, with homings and escapes of full duration
    # h_start is the list of all start times, with duration cropped to when the mouse reaches shelter
    # esc_start is the list of the start time of the escapes only
    start = [0]
    h_start = [0]
    esc_start = [0]

    # if interpolating, we are doubling time
    mult = 1
    if interpolation:
        mult = 2

    # initialize variables
    n_neur = frame_by_cluster_matrix.shape[1]
    escape_matrix = np.zeros((n_neur, np.sum(offs - ons) * mult))  # x 2 because of interpolation over time
    esc_var = np.zeros(np.sum(offs - ons) * mult)
    all_frames_to_keep = np.zeros(np.sum(offs - ons) * mult)
    cond = np.zeros(np.sum(offs - ons) * mult)

    for tr, (on, of) in enumerate(zip(ons, offs)):
        # extract variables
        neur = frame_by_cluster_matrix[on:of, :]  # time x neurons
        this_speed = behave[on:of]
        this_y = y_pos[on:of]
        this_x = x_pos[on:of]

        if interpolation:
            # interpolate over time, double the samples
            this_speed, this_y, this_x, neur = interpolate_time(
                this_x,
                this_y,
                this_speed,
                neur,
            )

        # find actual length of time until mouse is in shelter
        in_shelt_y = this_y > session.shelter_location[0][1]
        in_shelt_x = np.logical_and(
            this_x > session.shelter_location[0][0],
            this_x < session.shelter_location[1][0],
        )
        in_shelt = np.logical_and(in_shelt_x, in_shelt_y)

        if no_stationary:
            moving = this_speed > 0.5
            frames_to_keep = np.logical_and(in_shelt == 0, moving)
        else:
            frames_to_keep = in_shelt == 0

        # find times when the mouse is in the first or second leg of the escape, use this for cropping after
        # if it's a homing in a barrier or flipped barrier condition (barrier present is true), crop away the time when the mouse was in the threat zone
        if np.logical_and(
            compression_var in ["distance_shelter", "escape_shelter"],
            bar[of] == True,
        ):
            shelter_zone = this_y > 512
            frames_to_keep = np.logical_and(frames_to_keep, shelter_zone)
        # if it's a homing in a barrier or flipped barrier condition (barrier present is true), crop away the time when the mouse was in the shelter zone
        if np.logical_and(
            compression_var in ["distance_first_goal", "escape_first_goal"],
            bar[of] == True,
        ):
            threat_zone = this_y < 512
            frames_to_keep = np.logical_and(frames_to_keep, threat_zone)

        # condition vector
        c = np.zeros((len(this_y)))
        if bar[of] == True:
            c += 1
        if barflip[of] == True:
            c += 1

        disc_var = create_discretized_behave_var(
            session,
            compression_var,
            this_x,
            this_y,
            this_speed,
            c,
            bins=bins,
        )

        # add data from this trial to the escape matrix and behavioral variable
        escape_matrix[
            :,
            start[-1] : start[-1] + len(disc_var),
        ] = neur.T  # neurons x time
        esc_var[start[-1] : start[-1] + len(disc_var)] = disc_var
        all_frames_to_keep[start[-1] : start[-1] + len(disc_var)] = frames_to_keep
        cond[start[-1] : start[-1] + len(disc_var)] = c

        # add the start time of this trial to tell us later where the trials were in the escape matrix
        if on in esc_ons:
            esc_start.append(h_start[-1])
        start.append(start[-1] + len(disc_var))
        if len(np.where(frames_to_keep == 0)[0]) == 0:
            h_start.append(h_start[-1] + len(disc_var))  # never reaches shelter
        elif len(disc_var[frames_to_keep]) == 0:
            h_start = h_start
        else:
            h_start.append(h_start[-1] + len(disc_var[frames_to_keep]))  # only keep homing until mouse reaches shelter

    # if the first trial was not as escape, remove 0 from the list of escape start times
    if np.amin(ons) < np.amin(esc_ons):
        esc_start = esc_start[1:]

    # crop data to time before the mouse enters the shelter
    cond = cond[all_frames_to_keep == 1]
    esc_var = esc_var[all_frames_to_keep == 1]
    escape_matrix = escape_matrix[:, all_frames_to_keep == 1]
    # zscore the neural data
    if zscore:
        escape_matrix = zscore(escape_matrix, axis=1)

    if return_escape:
        return (
            esc_var,
            escape_matrix,
            cond,
            esc_start,
            h_start[:-1],
        )
    else:
        return (
            esc_var,
            escape_matrix,
            cond,
            h_start[:-1],
        )


def extract_homing_behave_var(
    session, y_pos, x_pos, bar, barflip, compression_var, ons, offs, shifted_vec=[], interpolation=True, bins=[]):
    """For a given session, extract the time around escapes and homing periods in neural data and a behavioral variable of interest (compression_var)
    INPUTS:
        session: session object, used to get the escape onsets and offsets
        frame_by_cluster_matrix: is a matrix of neural data, time x neurons
        behave, y_pos, x_pos: the vectors of speed, y and x position of the mouse
        bar, barflip: the vectors of barrier and flipped barrier
        compression_var: variable to compress the data into bins (full_distance_shelter, y_pos, escape, speed, distance_shelter, distance_first_goal, escape_shelter, escape_first_goal)
        ons, offs: the vectors of onsets and offsets of homings
        interpolation: if True, interpolate over time to double the number of samples
        no_stationary: if True, exclude stationary periods from the analysis (i.e. when speed < 0.5 cm/s) # TODO might not work
        bins: could be an emty list, an integer (number of bins) or a list of bin edges
    RETURNS:
        esc_var: the behavioral variable of interest, discretized into bins
        escape_matrix: a matrix of neural data, neurons x time
        cond: a vector of length time indicating what experimental condition the homing/escape was in
        h_start: the start time of the homings or escapes, duration of homing/escape is cropped to when the mouse reaches shelter
        esc_start: the start time of the escapes only, duration of homing/escape is cropped to when the mouse reaches shelter
    """

    # extract the time around escapes
    ons, offs, esc_ons = homing_escape_onsets(session, ons, offs)
    if len(shifted_vec) > 0:
        # which ones to keep: the start and end of the homing has to inside the chunks defined by shifted_vec
        mask = np.logical_and(shifted_vec[ons], shifted_vec[offs])
        ons = ons[mask]
        offs = offs[mask]
        esc_ons = [x for x in esc_ons if x in ons]
        # find the new timepoints
        on_vec = np.zeros_like(shifted_vec)
        on_vec[ons] = 1
        on_vec = on_vec[shifted_vec]
        ons = np.where(on_vec == 1)[0]
        off_vec = np.zeros_like(shifted_vec)
        off_vec[offs] = 1
        off_vec = off_vec[shifted_vec]
        offs = np.where(off_vec == 1)[0]
        eon_vec = np.zeros_like(shifted_vec)
        eon_vec[esc_ons] = 1
        eon_vec = eon_vec[shifted_vec]
        esc_ons = np.where(eon_vec == 1)[0]

    # initialize variables:
    # start is the list of all start times, with homings and escapes of full duration
    # h_start is the list of all start times, with duration cropped to when the mouse reaches shelter
    # esc_start is the list of the start time of the escapes only
    start = [0]
    h_start = [0]
    esc_start = [0]

    # if interpolating, we are doubling time
    mult = 1
    if interpolation:
        mult = 2

    # initialize variables
    esc_var = np.zeros(np.sum(offs - ons) * mult)
    all_frames_to_keep = np.zeros(np.sum(offs - ons) * mult)
    cond = np.zeros(np.sum(offs - ons) * mult)

    for tr, (on, of) in enumerate(zip(ons, offs)):
        # extract variables
        this_y = y_pos[on:of]
        this_x = x_pos[on:of]

        if interpolation:
            # interpolate over time, double the samples
            current_time = np.arange(len(this_x))
            new_time = np.arange(0, len(this_x), 0.5)
            this_y = np.interp(new_time, current_time, this_y)
            this_x = np.interp(new_time, current_time, this_x)

        # find actual length of time until mouse is in shelter
        in_shelt_y = this_y > session.shelter_location[0][1]
        in_shelt_x = np.logical_and(
            this_x > session.shelter_location[0][0],
            this_x < session.shelter_location[1][0],
        )
        in_shelt = np.logical_and(in_shelt_x, in_shelt_y)

        frames_to_keep = in_shelt == 0

        # find times when the mouse is in the first or second leg of the escape, use this for cropping after
        # if it's a homing in a barrier or flipped barrier condition (barrier present is true), crop away the time when the mouse was in the threat zone
        if np.logical_and(
            compression_var in ["distance_shelter", "escape_shelter"],
            bar[of] == True,
        ):
            shelter_zone = this_y > 512
            frames_to_keep = np.logical_and(frames_to_keep, shelter_zone)
        # if it's a homing in a barrier or flipped barrier condition (barrier present is true), crop away the time when the mouse was in the shelter zone
        if np.logical_and(
            compression_var in ["distance_first_goal", "escape_first_goal"],
            bar[of] == True,
        ):
            threat_zone = this_y < 512
            frames_to_keep = np.logical_and(frames_to_keep, threat_zone)

        # condition vector
        c = np.zeros((len(this_y)))
        if bar[of] == True:
            c += 1
        if barflip[of] == True:
            c += 1

        disc_var = create_discretized_behave_var(
            session,
            compression_var,
            this_x,
            this_y,
            [],
            c,
            bins=bins,
        )

        # add data from this trial to the escape matrix and behavioral variable
        esc_var[start[-1] : start[-1] + len(disc_var)] = disc_var
        all_frames_to_keep[start[-1] : start[-1] + len(disc_var)] = frames_to_keep
        cond[start[-1] : start[-1] + len(disc_var)] = c

        # add the start time of this trial to tell us later where the trials were in the escape matrix
        if on in esc_ons:
            esc_start.append(h_start[-1])
        start.append(start[-1] + len(disc_var))
        if len(np.where(frames_to_keep == 0)[0]) == 0:
            h_start.append(h_start[-1] + len(disc_var))  # never reaches shelter
        elif len(disc_var[frames_to_keep]) == 0:
            h_start = h_start
        else:
            h_start.append(h_start[-1] + len(disc_var[frames_to_keep]))  # only keep homing until mouse reaches shelter

    # if the first trial was not as escape, remove 0 from the list of escape start times
    if np.amin(ons) < np.amin(esc_ons):
        esc_start = esc_start[1:]

    # crop data to time before the mouse enters the shelter
    cond = cond[all_frames_to_keep == 1]
    esc_var = esc_var[all_frames_to_keep == 1]

    return esc_var, cond, h_start[:-1]

def extract_explore_periods(session,frame_by_cluster_matrix,behave,y_pos,x_pos,bar,barflip,compression_var,homie,escape,outofshelter,bins=[],interpolation=True,no_stationary=False,zscore=False):
    """A functon to extract the neural data and discretized behavioral variable for all exploration periods
    These are times when the mouse is out of the shelter, not in a homing or escape period.
    INPUTS:
        session: session object, used to get the escape onsets and offsets
        frame_by_cluster_matrix: is a matrix of neural data, time x neurons
        behave, y_pos, x_pos: the vectors of speed, y and x position of the mouse
        bar, barflip: the vectors of barrier and flipped barrier
        compression_var: variable to compress the data into bins (distance_shelter, y_pos, escape, speed)
        ons, offs: the vectors of onsets and offsets of homings
        interpolation: if True, interpolate over time to double the number of samples
        no_stationary: if True, exclude stationary periods from the analysis (i.e. when speed < 0.5 cm/s) # TODO might not work
        bins: could be an emty list, an integer (number of bins) or a list of bin edges
    RETURNS:
        esc_var: the behavioral variable of interest, discretized into bins
        escape_matrix: a matrix of neural data, neurons x time
        cond: a vector of length time indicating what experimental condition the homing/escape was in
    """

    if compression_var in ["escape", "escape_shelter", "escape_first_goal"]:
        print("You can't compute fraction of escape route during exploration period")
        return

    if interpolation:
        # interpolate over time, double the samples
        this_speed, this_y, this_x, neur = interpolate_time(x_pos, y_pos, behave, frame_by_cluster_matrix)

    # create vector of conditions
    if interpolation:
        current_time = np.arange(len(bar))
        new_time = np.arange(0, len(bar), 0.5)
        bar = np.interp(new_time, current_time, bar) > 0
        barflip = np.interp(new_time, current_time, barflip) > 0
    cond = np.zeros(len(bar))
    cond[bar == True] += 1
    cond[barflip == True] += 1

    # create discretized behavioral varable
    disc_var = create_discretized_behave_var(session, compression_var, this_x, this_y, this_speed, cond, bins = bins)

    # remove data when mouse is in shelter or in homing/escape
    frames_to_remove = np.logical_or(
        np.logical_or(homie, escape),
        outofshelter == False,
    )
    if interpolation:
        frames_to_remove = np.interp(new_time, current_time, frames_to_remove) > 0

    if no_stationary:
        stationary = this_speed < 0.5
        frames_to_remove = np.logical_or(frames_to_remove, stationary)

    cond = cond[frames_to_remove == False]
    escape_matrix = neur[frames_to_remove == False, :].T  # neurons x time
    esc_var = disc_var[frames_to_remove == False]

    # zscore the neural data
    if zscore:
        escape_matrix = zscore(escape_matrix, axis=1)

    return esc_var, escape_matrix, cond


def build_shift_vector(bar, barflip, session, ons, offs, shifts_one_sided):
    """This function builds a list of shifts and a vector of where to sample the central third of each condition for linear shift statistics"""

    shelter = np.where(bar == True)[0][0]  # the end of the shelter_only condition
    bar_in = np.where(barflip == True)[0][0]  # the end of the barrier condition
    mid_shelter = [int(shelter / 3), int((shelter / 3) * 2)]
    mid_bar = [int(shelter + ((bar_in - shelter) / 3)), int(shelter + (((bar_in - shelter) / 3) * 2))]
    mid_flip = [int(bar_in + ((len(bar) - bar_in) / 3)), int(bar_in + (((len(bar) - bar_in) / 3) * 2))]

    # check that this gives us at least 2(?) homings/escapes
    all_ons, _, _ = homing_escape_onsets(session, ons, offs)
    min_h = 5  # minimum number of homings per condition
    if np.sum(np.logical_and(all_ons > mid_shelter[0], all_ons < mid_shelter[1])) < min_h:
        mid_shelter = [
            all_ons[int(np.round(len(all_ons[all_ons < shelter]) / 3))] - 1,  # starting a third of the way into the homings
            (all_ons[int(np.round(len(all_ons[all_ons < shelter]) / 3))] - 1) + int(shelter / 3),
        ]
        if mid_shelter[0] < np.amax(shifts_one_sided):
            mid_shelter = [x + (np.amax(shifts_one_sided) - mid_shelter[0]) for x in mid_shelter]
        if mid_shelter[1] > (shelter - np.amax(shifts_one_sided)):
            mid_shelter = [x - (mid_shelter[1] - (shelter - np.amax(shifts_one_sided))) for x in mid_shelter]
        print("Number of homings in shelter_only centre chunk: " + str(np.sum(np.logical_and(all_ons > mid_shelter[0], all_ons < mid_shelter[1]))))
    if np.sum(np.logical_and(all_ons > mid_bar[0], all_ons < mid_bar[1])) < min_h:
        h_bar = all_ons[np.logical_and(all_ons > shelter, all_ons < bar_in)]
        mid_bar = [h_bar[int(np.round(len(h_bar) / 3))] - 1, int(shelter + h_bar[int(np.round(len(h_bar) / 3))] - 1)]
        if mid_bar[0] < (shelter + np.amax(shifts_one_sided)):
            mid_bar = [x + ((shelter + np.amax(shifts_one_sided)) - mid_bar[0]) for x in mid_bar]
        if mid_bar[1] > (bar_in - np.amax(shifts_one_sided)):
            mid_bar = [bar_in - np.amax(shifts_one_sided) - ((bar_in - shelter) / 3), bar_in - np.amax(shifts_one_sided)]
        print("Number of homings in barrier centre chunk: " + str(np.sum(np.logical_and(all_ons > mid_bar[0], all_ons < mid_bar[1]))))
    if np.sum(np.logical_and(all_ons > mid_flip[0], all_ons < mid_flip[1])) < min_h:
        h_flip = all_ons[np.logical_and(all_ons > bar_in, all_ons < len(bar))]
        mid_flip = [h_flip[int(np.round(len(h_flip) / 3))] - 1, int(bar_in + h_flip[int(np.round(len(h_flip) / 3))] - 1)]
        if mid_flip[0] < (bar_in + np.amax(shifts_one_sided)):
            mid_flip = [x + ((bar_in + np.amax(shifts_one_sided)) - mid_flip[0]) for x in mid_flip]
        if mid_flip[1] > (len(bar) - np.amax(shifts_one_sided)):
            mid_flip = [len(bar) - np.amax(shifts_one_sided) - ((len(bar) - bar_in) / 3), len(bar) - np.amax(shifts_one_sided)]
        print("Number of homings in flipped barrier centre chunk: " + str(np.sum(np.logical_and(all_ons > mid_flip[0], all_ons < mid_flip[1]))))

    # build the vector that gives us the central chunk of each conditon
    shift_vector = np.zeros(len(bar))
    shift_vector[int(mid_shelter[0]) : int(mid_shelter[1])] = 1
    shift_vector[int(mid_bar[0]) : int(mid_bar[1])] = 1
    shift_vector[int(mid_flip[0]) : int(mid_flip[1])] = 1
    shift_vector = shift_vector.astype(bool)

    # make sure we're not shifting our of range
    shifts_left = shifts_one_sided[shifts_one_sided < mid_shelter[0]]
    shifts_right = shifts_one_sided[(shifts_one_sided + mid_flip[1]) < len(bar)]

    # now double them so we go in both directions
    shifts = np.sort(np.hstack((shifts_right, -shifts_left)))

    return shifts, shift_vector


##-----------------UTILS----------------------


def homing_escape_onsets(session, ons, offs):
    """This function creates two vectors of onset and offset times for homing and escape periods
    INPUTS:
        session: session object, used to get the escape onsets and offsets
        ons: vector of onset times for homing periods
        offs: vector of offset times for homing periods
    RETURNS:
        ons: vector of onset times in frames for homing and escape periods
        offs: vector of offset times in frames for homing and escape periods
        esc_ons: vector of onset times in frames for escape periods
    """
    esc_ons = check_not_list(session.audio.onset_frames)
    st = [x * 40 for x in check_not_list(session.audio.stimulus_durations)]
    esc_offs = (np.add(esc_ons, st)).astype(int)

    ons = np.sort(np.append(check_not_list(ons), esc_ons))
    offs = np.sort(np.append(check_not_list(offs), esc_offs))
    return ons, offs, esc_ons


def create_discretized_behave_var(
    session,
    compression_var,
    this_x,
    this_y,
    this_speed,
    c,
    bins=[],  # set bin size for discritizing behavioral variables, this will be changed depending on the behavioral variable
):
    """This function returns the discretized behavioral variable of interest
    INPUTS:
        session: session object
        compression_var: behavioral variable we want to extract and discretize (distance_shelter, y_pos, escape, speed)
        this_x: x position of the mouse
        this_y: y position of the mouse
        this_speed: speed of the mouse
        c: experimental condition of this trial [0 for shelter_only, 1 for barrier, 2 for barrier_flip]
        bins: if vector it defnes the bin ranges, if number how many bins to use
    RETURNS:
        disc_var: the discretized behavioral variable of interest, in time
    """

    if compression_var in [
        "full_distance_shelter",
        "distance_shelter",
    ]:
        var = compute_dist_shelt(this_x, this_y, c, session)
    elif compression_var == "distance_first_goal":
        var = compute_dist_first_goal(this_x, this_y, c, session)
    elif compression_var in ["bird_dist_shelter", "bird_dist_first_goal"]:
        var = compute_dist_shelt(this_x, this_y, cond=np.zeros_like(this_x), session=session)  # straight distance computed by pretending there is never a barrier
    elif compression_var == "y_pos":
        var = this_y
    elif "escape" in compression_var:
        start, stop = [0, -1]
        if np.logical_and(c[0] > 0, compression_var == "escape_shelter"):  # don't crop for shelter only trials
            start = np.where(this_y > 512)[0]
            if len(start) > 0:
                start = start[0]
            else:
                return np.zeros_like(this_x)
        if np.logical_and(c[0] > 0, compression_var == "escape_first_goal"):  # don't crop for shelter only trials
            stop = np.where(this_y < 512)[0]
            if len(stop) > 0:
                stop = stop[-1]
            else:
                return np.zeros_like(this_x)
        dd = compute_escape_trajectory(this_x, this_y, start, stop)
        var = dd / np.amax(dd)
        if isinstance(bins, list):
            if (not bins):
                bins = np.arange(0, 1, 0.01)  # .01
    elif compression_var == "speed":
        var = this_speed
        if isinstance(bins, list):
            if (not bins) & ("dist" in compression_var):
                bins = np.arange(0, np.amax(var), 1)  # 1

    if isinstance(bins, list):
        if (not bins) & ("dist" in compression_var):
            bins = np.arange(0, np.amax(var), np.amax(var) / 10)

    if isinstance(bins, int):
        bins = np.arange(0, np.amax(var), np.amax(var) / bins)

    disc_var = discretize_x_axis(var, bins)
    return disc_var


def interpolate_time(x, y, speed, neural):
    """Interpolate over time, double the samples"""

    current_time = np.arange(len(speed))
    new_time = np.arange(0, len(speed), 0.5)
    this_speed = np.interp(new_time, current_time, speed)
    this_y = np.interp(new_time, current_time, y)
    this_x = np.interp(new_time, current_time, x)
    new_neur = np.zeros((len(this_speed), np.shape(neural)[1]))
    for i in np.arange(np.shape(neural)[1]):
        new_neur[:, i] = np.interp(new_time, current_time, neural[:, i])

    return this_speed, this_y, this_x, new_neur
