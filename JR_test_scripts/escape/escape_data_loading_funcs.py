import numpy as np
from scipy.stats import zscore

from JR_test_scripts.escape.escape_utils import check_not_list, compute_dist_shelt, compute_escape_trajectory, discretize_x_axis

def extract_homing_and_escape_periods(session, frame_by_cluster_matrix, behave, y_pos, x_pos, bar, barflip, compression_var, ons, offs, interpolation = True, no_stationary = False, return_escape = False):
    """For a given session, extract the time around escapes and homing periods in neural data and a behavioral variable of interest (compression_var)
    INPUTS:
        session: session object, used to get the escape onsets and offsets
        frame_by_cluster_matrix: is a matrix of neural data, time x neurons
        behave, y_pos, x_pos: the vectors of speed, y and x position of the mouse
        bar, barflip: the vectors of barrier and flipped barrier
        compression_var: variable to compress the data into bins (distance_shelter, y_pos, escape, speed)
        ons, offs: the vectors of onsets and offsets of homings
        interpolation: if True, interpolate over time to double the number of samples
        no_stationary: if True, exclude stationary periods from the analysis (i.e. when speed < 0.5 cm/s) # TODO might not work
    RETURNS:
        esc_var: the behavioral variable of interest, discretized into bins
        escape_matrix: a matrix of neural data, neurons x time
        cond: a vector of length time indicating what experimental condition the homing/escape was in
        h_start: the start time of the homings or escapes, duration of homing/escape is cropped to when the mouse reaches shelter
        esc_start: the start time of the escapes only, duration of homing/escape is cropped to when the mouse reaches shelter
        """

    # extract the time around escapes
    ons, offs, esc_ons = homing_escape_onsets(session, ons, offs)

    # initialize variables: 
    # start is the list of all start times, with homings and escapes of full duration
    # h_start is the list of all start times, with duration cropped to when the mouse reaches shelter
    # esc_start is the list of the start time of the escapes only
    start = [0]
    h_start = [0]
    esc_start = [0]

    # if interpolating, we are doubling time
    mult = 1
    if interpolation: mult = 2

    # initialize variables
    escape_matrix = np.zeros((np.shape(frame_by_cluster_matrix)[1],np.sum(offs - ons)*mult)) # x 2 because of interpolation over time
    esc_var = np.zeros(np.sum(offs - ons)*mult)
    all_frames_to_keep = np.zeros(np.sum(offs - ons)*mult)
    cond = np.zeros(np.sum(offs - ons)*mult)

    for tr, (on,of) in enumerate(zip(ons, offs)):
        # extract variables
        neur = frame_by_cluster_matrix[on:of,:] # time x neurons
        this_speed = behave[on:of]
        this_y = y_pos[on:of]
        this_x = x_pos[on:of]

        if interpolation:
            # interpolate over time, double the samples
            this_speed, this_y, this_x, neur = interpolate_time(this_x, this_y, this_speed, neur)

        # find actual length of time until mouse is in shelter
        in_shelt_y = this_y > session.shelter_location[0][1]
        in_shelt_x = np.logical_and(this_x > session.shelter_location[0][0], this_x < session.shelter_location[1][0])
        in_shelt = np.logical_and(in_shelt_x, in_shelt_y)

        if no_stationary:
            moving = this_speed > .5
            frames_to_keep = np.logical_and(in_shelt == 0, moving)
        else:
            frames_to_keep = in_shelt == 0

        # condition vector
        c = np.zeros((len(this_y)))
        if bar[of] == True: c += 1
        if barflip[of] == True: c += 1

        disc_var = create_discretized_behave_var(session, compression_var, this_x, this_y, this_speed, c)

        # add data from this trial to the escape matrix and behavioral variable
        escape_matrix[:,start[-1]:start[-1]+len(disc_var)] = neur.T # neurons x time
        esc_var[start[-1]:start[-1]+len(disc_var)] = disc_var
        all_frames_to_keep[start[-1]:start[-1]+len(disc_var)] = frames_to_keep
        cond[start[-1]:start[-1]+len(disc_var)] = c
        
        # add the start time of this trial to tell us later where the trials were in the escape matrix
        if on in esc_ons:
            esc_start.append(h_start[-1])
        start.append(start[-1]+len(disc_var))
        if len(np.where(frames_to_keep == 0)[0]) == 0:        
            h_start.append(h_start[-1]+len(disc_var)) # never reaches shelter
        else:
            h_start.append(h_start[-1]+len(disc_var[frames_to_keep])) # only keep homing until mouse reaches shelter
    
    # if the first trial was not as escape, remove 0 from the list of escape start times
    if np.amin(ons) < np.amin(esc_ons):
        esc_start = esc_start[1:]

    # crop data to time before the mouse enters the shelter
    cond = cond[all_frames_to_keep == 1]
    esc_var = esc_var[all_frames_to_keep == 1]
    escape_matrix = escape_matrix[:,all_frames_to_keep == 1]
    
    # zscore the neural data
    escape_matrix = zscore(escape_matrix, axis = 1)
    
    if return_escape:
        return esc_var, escape_matrix, cond, esc_start, h_start[:-1]
    else:
        return esc_var, escape_matrix, cond, h_start[:-1]
    
def extract_non_homing_escape_periods(session, frame_by_cluster_matrix, behave, y_pos, x_pos, bar, barflip, compression_var, homie, escape, outofshelter, interpolation = True, no_stationary = False):
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
    RETURNS:
        esc_var: the behavioral variable of interest, discretized into bins
        escape_matrix: a matrix of neural data, neurons x time
        cond: a vector of length time indicating what experimental condition the homing/escape was in
    """

    if interpolation:
        # interpolate over time, double the samples
        this_speed, this_y, this_x, neur = interpolate_time(x_pos, y_pos, behave, frame_by_cluster_matrix)

    # create vector of conditions
    if interpolation:
        current_time = np.arange(len(bar))
        new_time = np.arange(0,len(bar),.5)
        bar = np.interp(new_time, current_time, bar) > 0
        barflip = np.interp(new_time, current_time, barflip) > 0
    cond = np.zeros(len(bar))
    cond[bar == True] += 1
    cond[barflip == True] += 1

    # create discretized behavioral varable
    disc_var = create_discretized_behave_var(session, compression_var, this_x, this_y, this_speed, cond)

    # remove data when mouse is in shelter or in homing/escape
    frames_to_remove = np.logical_or(np.logical_or(homie, escape), outofshelter == False)
    if interpolation:
        frames_to_remove = np.interp(new_time, current_time, frames_to_remove) > 0

    if no_stationary:
        stationary = this_speed < .5
        frames_to_remove = np.logical_or(frames_to_remove, stationary)

    cond = cond[frames_to_remove == False]
    escape_matrix = neur[frames_to_remove == False,:].T # neurons x time
    esc_var = disc_var[frames_to_remove == False]

    # zscore the neural data
    escape_matrix = zscore(escape_matrix, axis = 1)

    return esc_var, escape_matrix, cond

def homing_escape_onsets(session, ons, offs):
    """This function creates two vectors of onset and offset times for homing and escape periods
    INPUTS:
        session: session object, used to get the escape onsets and offsets
        ons: vector of onset times for homing periods
        offs: vector of offset times for homing periods
    RETURNS:
        ons: vector of onset times in frames for homing and escape periods
        offs: vector of offset times in frames for homing and escape periods
    """
    esc_ons = check_not_list(session.audio.onset_frames)
    st = [x*40 for x in check_not_list(session.audio.stimulus_durations)]
    esc_offs = (np.add(esc_ons, st)).astype(int)

    ons = np.sort(np.append(check_not_list(ons), esc_ons))
    offs = np.sort(np.append(check_not_list(offs), esc_offs))
    return ons, offs, esc_ons

def create_discretized_behave_var(session, compression_var, this_x, this_y, this_speed, c):
    """This function returns the discretized behavioral variable of interest
    INPUTS:
        session: session object
        compression_var: behavioral variable we want to extract and discretize (distance_shelter, y_pos, escape, speed)
        this_x: x position of the mouse
        this_y: y position of the mouse
        this_speed: speed of the mouse
        c: experimental condition of this trial [0 for shelter_only, 1 for barrier, 2 for barrier_flip]
    RETURNS:
        disc_var: the discretized behavioral variable of interest, in time
    """
    
    # set bin size for discritizing behavioral variables
    bin_size = 10 # this will be changed depending on the behavioral variable

    if compression_var == 'distance_shelter':
        var = compute_dist_shelt(this_x, this_y, c, session)
    elif compression_var == 'y_pos':
        var = this_y
    elif compression_var == 'escape':
        dd = compute_escape_trajectory(this_x, this_y)
        var = (dd/np.amax(dd))
        bin_size = .01 # .01
    elif compression_var == 'speed':
        var = this_speed
        bin_size = 1 # 1

    disc_var = discretize_x_axis(var, bin_size)
    return disc_var

def interpolate_time(x, y, speed, neural):
    """Interpolate over time, double the samples"""

    current_time = np.arange(len(speed))
    new_time = np.arange(0,len(speed),.5)
    this_speed = np.interp(new_time, current_time, speed)
    this_y = np.interp(new_time, current_time, y)
    this_x = np.interp(new_time, current_time, x)
    new_neur = np.zeros((len(this_speed),np.shape(neural)[1]))
    for i in np.arange(np.shape(neural)[1]):
        new_neur[:,i] = np.interp(new_time, current_time, neural[:,i])

    return this_speed, this_y, this_x, new_neur