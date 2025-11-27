import gc
import os
import numpy as np
from loguru import logger
from scipy.ndimage import gaussian_filter1d
import dill as pickle

from settings.settings_analyze_efizz import Settings_ae as settings
from behave_analysis.analyze.EscapePattern.EscapeTuning import init_escape_tuning
from behave_analysis.analyze.EscapePattern.escape_pattern_utils import (select_onset_offsets_in_shift_vector, 
                                                                        homing_escape_onsets, 
                                                                        create_discretized_behave_var, 
                                                                        build_shift_vector)
from behave_analysis.analyze.EscapePattern.tuning_functions import compute_tuning_curves, tuning_method_no_trials_with_pool
from behave_analysis.utils.creating_directories import make_directory

class ComputeEscapeTuning:
    """A class for computing the tuning to escape-related variables and storing them in the EscapeTuning dataclass
    1. Extract the data: this either concatenates all homings+escapes or exploration periods
    2. Compute the firing-by-variable tuning curves: variables can be % escape, distance to shelter
    3. Compute the leave one out reliability: NB this can only be done on homings+escapes
    4. Compute the statistical significance of the tuning curves via linear shift
    """

    def __init__(self, aefizz):
        self.ET = init_escape_tuning(settings)
        self.fcm, self.x, self.y, self.condition = self.load_data(aefizz)
        # do we want to add escapes too?
        self.ons, self.offs, self.esc_ons = homing_escape_onsets(aefizz)
        self.ET.settings = settings # TODO: maybe we only want to save the EscapeTuning settings, not all the aefizz ones too about other methods
        # build save path
        self.ET.savepath = make_directory(os.path.join(aefizz.session.base_path, 
                                                       aefizz.session.processed_path, 
                                                       "escape_tuning",
                                                       settings.escape_pattern_time))
        # check that we're not trying to compute %escape tuning during explore periods
        if settings.escape_pattern_time == "explore" and settings.escape_tuning_var == "escape":
            raise ValueError("Cannot compute escape tuning during explore periods")
        pass

    def extract_data_and_tuning(self, aefizz):
        """This is a function that builds a matrix of neurons x time of activity in escape+homings or exploration
        and a behavioral variable of interest (var) discretized into bins (determined in settings)
        """
        # create filtering vector
        if settings.escape_pattern_time == "homing + escape":
            logger.info("Computing Escape Pattern Tuning during homing + escape periods")
            self.ET.homing_vector, self.ET.escape_vector = self.homing_escape_filtering_vector(aefizz)
            filtering_vector = self.ET.homing_vector
        elif settings.escape_pattern_time == "explore":
            logger.info("Computing Escape Pattern Tuning during exploration periods")
            self.ET.explore_vector = self.explore_filtering_vector(aefizz)
            filtering_vector = self.ET.explore_vector

        # filter data during homing+escape or explore periods
        x = self.x[filtering_vector]
        y = self.y[filtering_vector]
        self.ET.neural_matrix = self.fcm[filtering_vector, :].T # neurons x time
        self.ET.condition = self.condition[filtering_vector]

        # compute behavioral variable
        self.ET.discretized_var = create_discretized_behave_var(aefizz, self.ET, x, y, self.ET.condition, self.ET.homing_vector if settings.escape_pattern_time == "homing + escape" else None)

        # compute tuning curves for each neuron
        if settings.escape_pattern_time == "homing + escape":
            y_fit, R, fr, params, mat, loo = compute_tuning_curves(var = self.ET.discretized_var, 
                                                                escape_matrix = self.ET.neural_matrix, 
                                                                cond = self.ET.condition, 
                                                                bins = settings.escape_tuning_bins, 
                                                                filtering_vector = filtering_vector,
                                                                n_cond = len(np.unique(self.ET.condition)), 
                                                                n_neur = self.ET.neural_matrix.shape[0], 
                                                                avg = 'winsorized', 
                                                                fitting = settings.ep_gaussian_fitting, # whether to fit a gaussian to each response curve
                                                                loo = settings.ep_compute_loo_reliability) # whether to compute leave one out reliability
            self.ET.mat_num_cond = mat
            if settings.ep_compute_loo_reliability:
                self.ET.loo_reliability_full = loo
        
        elif settings.escape_pattern_time == "explore":
            y_fit, R, fr, params = tuning_method_no_trials_with_pool(var = self.ET.discretized_var, 
                                                                    escape_matrix = self.ET.neural_matrix, 
                                                                    cond = self.ET.condition, 
                                                                    bins = settings.escape_tuning_bins, 
                                                                    n_cond = len(np.unique(self.ET.condition)), 
                                                                    n_neur = self.ET.neural_matrix.shape[0], 
                                                                    fitting = settings.ep_gaussian_fitting) # whether to fit a gaussian to each response curve
        
        self.ET.fr_full, self.ET.params_full, self.ET.y_fitted_full = fr, params, y_fit
        if settings.ep_gaussian_fitting:
            self.ET.R_full = R

    def compute_statistical_significance(self, aefizz):
        """This function performs linear shift stats on the tuning curves
        1. It builds a boolean shift vector of length time which subselect the central 1/3 of each condition
        2. It applies the shift vector to the neural and behavioral data to compute the null (the homings or explore periods need to be subselected carefully)
        3. The shift vector is shifted and the shifted vector is applied to the neural data"""
        
        # build shift vector to subselect central 1/3 of each condition
        shifts, shift_vector = build_shift_vector(aefizz, self.ET)

        # select which onsets and offsets to keep based on shift vector
        if settings.escape_pattern_time == "homing + escape":
            filtering_vector = select_onset_offsets_in_shift_vector(self.ET, shift_vector)
        elif settings.escape_pattern_time == "explore":
            filtering_vector = self.ET.explore_vector[shift_vector]

        x = self.x[filtering_vector]
        y = self.y[filtering_vector]
        condition = self.condition[filtering_vector]
        
        # compute behavioral variable
        discretized_var = create_discretized_behave_var(aefizz, self.ET, x, y, condition, filtering_vector)

        # initialize variables for output
        step_n, n_cond, n_neur, Nbins = len(shifts), len(np.unique(condition)), self.ET.neural_matrix.shape[0], settings.escape_tuning_bins
        self.ET.y_fitted_shift = np.full((step_n, n_cond, n_neur, Nbins), np.nan) # conditions x neurons x n_bins
        if settings.ep_gaussian_fitting:
            self.ET.R_shift = np.zeros((step_n,n_neur, n_cond)) # neurons x conditions
        if settings.ep_compute_loo_reliability:
            self.ET.loo_shift = np.zeros((step_n, n_cond, n_neur)) # conditions x neurons
        self.ET.params_shifts = np.zeros((step_n,n_neur, n_cond, 6)) # neurons x conditions
        self.ET.fr_shift = np.full((step_n, n_cond, n_neur, Nbins), np.nan)
        # trial n per condition
        trial_start_cond = self.condition[np.where(np.diff(filtering_vector)>0)[0]]
        trial_n_cond = np.bincount(trial_start_cond.astype(int)) 
        self.ET.mat_shift_cond = np.full((step_n, n_cond, n_neur, max(trial_n_cond), Nbins), np.nan)
        
        # iterate over shifts, compute the tuning curves
        for s_idx, s in enumerate(shifts):

            shifted_vec = np.roll(filtering_vector,int(s))

            # filter data during homing+escape or explore periods and central third
            neural_matrix = self.fcm[shifted_vec, :].T

            # compute the tuning curve on the unshifted, subselected data
            if settings.escape_pattern_time == "homing + escape":
                y, gf, fr, p, mat, reli = compute_tuning_curves(var = discretized_var, 
                                                                    escape_matrix = neural_matrix, 
                                                                    cond = condition, 
                                                                    bins = Nbins, 
                                                                    filtering_vector = filtering_vector,
                                                                    n_cond = n_cond, 
                                                                    n_neur = n_neur, 
                                                                    avg = 'winsorized', 
                                                                    fitting = settings.ep_gaussian_fitting, # whether to fit a gaussian to each response curve
                                                                    loo = settings.ep_compute_loo_reliability) # whether to compute leave one out reliability
                
                self.ET.mat_shift_cond[s_idx,:,:,:] = mat 
                if settings.ep_compute_loo_reliability:
                    self.ET.loo_shift[s_idx,:,:] = reli

            elif settings.escape_pattern_time == "explore":
                y, gf, fr, p = tuning_method_no_trials_with_pool(var = discretized_var, 
                                                                escape_matrix = neural_matrix, 
                                                                cond = condition, 
                                                                bins = Nbins, 
                                                                n_cond = n_cond, 
                                                                n_neur = n_neur, 
                                                                fitting = settings.ep_gaussian_fitting) # whether to fit a gaussian to each response curve

            self.ET.y_fitted_shift[s_idx,:,:,:], self.ET.fr_shift[s_idx,:,:,:], self.ET.params_shifts[s_idx,:,:,:] = y, fr, p
            if settings.ep_gaussian_fitting:
                self.ET.R_shift[s_idx,:,:] = gf

    def save_escape_tuning(self):
        """Save EscapeTuning dataclass to file"""
        # savepath building
        filename = self.ET.savepath + os.sep + settings.escape_tuning_var + "_" + str(settings.escape_tuning_bins) + "bins.pkl"
        with open(filename, "wb") as f:
            pickle.dump(self.ET, f)

# ----------------------------Data loading and processing functions----------------------------

    def homing_escape_filtering_vector(self, aefizz):
        """This function builds two boolean vectors of length time which are True when the mouse is in homing or escape periods
        TODO: currently does not filter based on correct/incorrect homings, first/second leg, or minimum homing length"""

        homing_vector = np.zeros_like(self.condition, dtype=bool)
        escape_vector = np.zeros_like(self.condition, dtype=bool)

        # iterate over homings
        for tr, (on, of) in enumerate(zip(self.ons, self.offs)):



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

            if settings.escape_pattern_interpolation_mult > 1:
                on = on * settings.escape_pattern_interpolation_mult
                of = of * settings.escape_pattern_interpolation_mult

            homing_vector[on:of] = True
            escape_vector[on:of] = True if on in self.esc_ons else False

        return homing_vector, escape_vector

    def explore_filtering_vector(self, aefizz):
        """This function builds a boolean vector of length time which is True when the mouse is exploring
        i.e. not in homing or escape periods and is outside of the shelter
        TODO: double check the logic!"""

        logger.warning("Quality of homing Period boolean vector not checked yet")
        if settings.no_stationary:
            Explore_vector = np.logical_and(
                np.logical_or(aefizz.video_df["homingPeriod"] == False, aefizz.video_df["EscapePeriod"] == False),
                aefizz.video_df["OutofshelterIdx"] == True,
            ) & (aefizz.video_df["speed"] > 0.5)
        else:
            explore_vector = np.logical_or(
                np.logical_or(aefizz.video_df["homingPeriod"] == False, aefizz.video_df["EscapePeriod"] == False),
                aefizz.video_df["OutofshelterIdx"] == True,
            )
        
        return explore_vector.to_numpy(dtype=bool)

    def load_data(self, aefizz):
        """This function loads the data from the aefizz object and does any necessary preprocessing
        """
        # gaussian filter
        fcm = gaussian_filter1d(aefizz.frame_by_cluster_matrix, 2, axis = 0)

        # load behavioral data
        y = aefizz.video_df["mouse_y_position"].to_numpy()
        x = aefizz.video_df["mouse_x_position"].to_numpy()
        bar = aefizz.video_df["barrier_present"].to_numpy()
        barflip = aefizz.video_df["barrier_flipped"].to_numpy()

        # interpolate time
        if settings.escape_pattern_interpolation_mult > 1:
            current_time = np.arange(len(y))
            new_time = np.arange(0, len(y), 1/settings.escape_pattern_interpolation_mult)
            # mouse position
            y = np.interp(new_time, current_time, y)
            x = np.interp(new_time, current_time, x)
            # experimental condition
            bar = np.interp(new_time, current_time, bar)
            barflip = np.interp(new_time, current_time, barflip)
            # neural data
            new_neur = np.zeros((len(y), np.shape(fcm)[1]))
            for i in np.arange(np.shape(fcm)[1]):
                new_neur[:, i] = np.interp(new_time, current_time, fcm[:, i])
            fcm = new_neur
            del new_neur
            gc.collect()
        # experimental condition vector
        condition = np.zeros(len(bar))
        condition[bar == True] += 1
        condition[barflip == True] += 1

        return fcm, x, y, condition


