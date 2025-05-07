import numpy as np


def make_escape_bool(h_e_bool, full_e_bool, h_cond):
    """A function that takes in the homing and escape boolean and returns a list of start times for the homings and escapes."""
    # starts of homings & escapes
    starts = np.where(np.diff(h_e_bool.astype(int)) > 0)[0] + 1
    ends = np.where(np.diff(h_e_bool.astype(int)) < 0)[0] + 1
    homie_lengths = ends - starts
    counter, h_start, e_start = 0, np.full(len(homie_lengths), 0), np.full(len(homie_lengths) + 1, False)
    for i, h in enumerate(homie_lengths):
        if full_e_bool[starts[i]]:
            e_start[i] = True  # a bool that tells us which one of the h_starts are escapes
        h_start[i] = counter
        counter += h
    h_start = np.append(h_start, len(h_cond))  # a list of the start indices of homings + escapes in the homing/escape time period
    e_bool = full_e_bool[h_e_bool]  # a boolean that tells us which periods of the homing/escape are escapes

    return h_start, e_bool

def make_long_homing_bool(h_start, X, Y):
    """A functon that identifies which homings and escapes are long: defined as ones that go from threat zone all the way to shelter"""
    # where did the mouse start and end
    y_start = Y[h_start[:-1]]
    y_end = Y[h_start[1:] - 1]
    long_homie_bool = np.full(len(h_start[:-1]), False)
    homie_id = np.full(len(X), 0)  # a vector that increases with each homing
    for h, (s, e) in enumerate(zip(y_start, y_end)):
        homie_id[h_start[h] : h_start[h + 1]] = h
        if (s < 512) & (e > 700):
            long_homie_bool[h] = True

    return long_homie_bool


def pre_homing_bool(h_e_bool, time_s, long_homie_bool=[], e_start=[]):
    """A function that takes in the homing and escape boolean and returns a boolean for the time before the homing starts.
    It also returns a list of the homings that are escapes."""

    time = int(time_s * 40)  # convert seconds to frames
    # find the times before the homie starts
    homie = np.where(np.diff(h_e_bool.astype(int)) > 0)[0] + 1

    # if we're not given the long_homie_bool, we make a dummy one - this means we don't care about isolating the long homies
    if len(long_homie_bool) == 0:
        long_homie_bool = np.full(len(homie), True)
    # if we're not given the e_start, we make a dummy one - this means we don't care about isolating the escapes
    if len(e_start) == 0:
        e_start = np.full(len(homie), False)

    # make variables
    prebool = np.full(len(h_e_bool), False)
    e_long = []
    counter = 0
    for idx, i in enumerate(homie):
        if long_homie_bool[idx]:
            prebool[i: i+time] = True
            counter += 1
            if e_start[idx]:
                e_long.append(counter)

    return prebool, e_long
