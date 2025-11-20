"""This is a file of functions to support computing escape tuning curves."""

import numpy as np


def define_bin_edges(settings):
    """Define bin edges based on settings.tuning_var and settings.tuning_bins."""
    # if tuning_bins is an integer, create that many bins between min and max of the variable
    if settings.tuning_bins.type == int:
        if settings.tuning_var == "bird_dist_shelter":
            bin_edges = np.linspace(0, 925, settings.tuning_bins + 1)
        elif settings.tuning_var == "escape":
            bin_edges = np.append(np.arange(0, 1, 1 / settings.tuning_bins), 1 + 1e-10)
    elif settings.tuning_bins.type == list:
        bin_edges = settings.tuning_bins

    return bin_edges

def homing_escape_onsets(aefizz):
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
    # pull out escape onsets and calculate offset estimate based on stimulus duration (assuming 40 fps) - mouse will likely lon gbe in shelter by then
    esc_ons = check_not_list(aefizz.session.audio.onset_frames)
    st = [x * 40 for x in check_not_list(aefizz.session.audio.stimulus_durations)]
    esc_offs = (np.add(esc_ons, st)).astype(int)

    # add escapes to homings onsets and offsets
    ons = np.sort(np.append(check_not_list(aefizz.homings_object.onset_frames), esc_ons))
    offs = np.sort(np.append(check_not_list(aefizz.homings_object.offset_frames), esc_offs))

    return ons, offs, esc_ons

def check_not_list(var):
    if np.logical_or(isinstance(var[0], list),
                     isinstance(var[0], np.ndarray)):
        var = [x[0] for x in var]
    return var