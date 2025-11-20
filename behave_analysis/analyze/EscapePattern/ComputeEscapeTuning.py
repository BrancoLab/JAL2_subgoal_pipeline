import numpy as np
from loguru import logger

from settings.settings_analyze_efizz import Settings_ae as settings
from behave_analysis.analyze.EscapePattern.escape_pattern_utils import define_bin_edges
from behave_analysis.analyze.EscapePattern.EscapeTuning import EscapeTuning as ET
from behave_analysis.analyze.EscapePattern.escape_pattern_utils import homing_escape_onsets


class ComputeEscapeTuning:
    """A class for computing the tuning to escape-related variables and storing them in the EscapeTuning dataclass
    1. Extract the data: this either concatenates all homings+escapes or exploration periods
    2. Compute the firing-by-variable tuning curves: variables can be % escape, distance to shelter
    3. Compute the leave one out reliability: NB this can only be done on homings+escapes
    4. Compute the statistical significance of the tuning curves via linear shift
    """

    def __init__(self):
        ET.nbins = settings.escape_tuning_bins
        ET.bin_edges = define_bin_edges(settings)
        ET.tuning_var = settings.escape_tuning_var
        ET.settings = settings # TODO: maybe we only want to save the EscapeTuning settings, not all the aefizz ones too about other methods
        pass

    def homing_escape_filtering_vector(self, aefizz):
        """This function builds two boolean vectors of length time which are True when the mouse is in homing or escape periods
        TODO: currently does not filter based on correct/incorrect homings, first/second leg, or minimum homing length"""

        homing_vector = np.zeros(aefizz.frame_by_cluster_matrix.shape[0], dtype=bool)
        escape_vector = np.zeros(aefizz.frame_by_cluster_matrix.shape[0], dtype=bool)

        # do we want to add escapes too?
        ons, offs, esc_ons = homing_escape_onsets(aefizz)

        # iterate over homings
        for tr, (on, of) in enumerate(zip(ons, offs)):

            # extract mouse position in the run
            this_y = aefizz.video_df["mouse_y_position"].to_numpy()[on:of]
            this_x = aefizz.video_df["mouse_x_position"].to_numpy()[on:of]

            # crop homings at shelter entry
            # find actual length of time until mouse is in shelter
            in_shelt = np.logical_and(
                this_y > aefizz.session.shelter_location[0][1], 
                np.logical_and(this_x > aefizz.session.shelter_location[0][0], this_x < aefizz.session.shelter_location[1][0])
            )
            shelter_entry = np.where(np.diff(in_shelt) > 0)[0][0] + 1 if np.any(np.diff(in_shelt) > 0) else len(in_shelt)
            of = on + shelter_entry

            # do we only want long homings?

            # only 'correct' homings?

            # do we want to crop homings into first and second leg?

            homing_vector[on:of] = True
            escape_vector[on:of] = True if on in esc_ons else False

        return homing_vector, escape_vector

    def explore_filtering_vector(self, aefizz):
        """This function builds a boolean vector of length time which is True when the mouse is exploring
        i.e. not in homing or escape periods and is outside of the shelter"""

        logger.warning("Quality of homing Period boolean vector not checked yet")
        explore_vector = np.logical_or(
            np.logical_or(aefizz.video_df["homingPeriod"] == False, aefizz.video_df["EscapePeriod"] == False),
            aefizz.video_df["OutofshelterIdx"] == True,
        )
        return explore_vector

    def extract_data_matrix(self, aefizz, interpolation=True, no_stationary=False, shifted_vec=[]):
        """This is a function that builds a matrix of neurons x time of activity in escape+homings or exploration
        and a behavioral variable of interest (var) discretized into bins (determined in settings)
        INPUTS:
            AnalyzeEfizz object:
                session: session object, used to get the escape onsets and offsets
                frame_by_cluster_matrix: is a matrix of neural data, time x neurons
                behave, y_pos, x_pos: the vectors of speed, y and x position of the mouse
                bar, barflip: the vectors of barrier and flipped barrier
                ons, offs: the vectors of onsets and offsets of homings
            Settings object:
                tuning_var: variable to compress the data into bins (full_distance_shelter, y_pos, escape, speed, distance_shelter, distance_first_goal, escape_shelter, escape_first_goal)
                interpolation: if True, interpolate over time to double the number of samples
                no_stationary: if True, exclude stationary periods from the analysis (i.e. when speed < 0.5 cm/s) # TODO might not work
            tuning_bin_edges: bin edges for discretizing the variable of interest
        RETURNS:
            EscapeTuning object:
                var: the behavioral variable of interest, discretized into bins
                escape_matrix: a matrix of neural data, neurons x time
                cond: a vector of length time indicating what experimental condition the homing/escape was in
                h_start: the start time (with respect to escape matrix?) of the homings or escapes, duration of homing/escape is cropped to when the mouse reaches shelter
        """
        # interpolate time

        # stationary?

        return

    def compute_tuning_curves(self):
        pass

    def compute_leave_one_out_reliability(self):
        pass

    def compute_statistical_significance(self):
        """This function performs linear shift stats on the tuning curves
        1. It builds a boolean shift vector of length time which subselect the central 1/3 of each condition
        2. It applies the shift vector to the neural and behavioral data to compute the null (the homings or explore periods need to be subselected carefully)
        3. The shift vector is shifted and the shifted vector is applied to the neural data"""

        # build shift vector to subselect central 1/3 of each condition
        shifts, shift_vector = build_shift_vector(bar, barflip, session, ons, offs, shifts_one_sided)

        # select which onsets and offsets to keep based on shift vector

        # compute the tuning curve on the unshifted, subselected data

        # iterate over shifts, compute the tuning curves
        pass

    def save_escape_tuning(self):
        pass
