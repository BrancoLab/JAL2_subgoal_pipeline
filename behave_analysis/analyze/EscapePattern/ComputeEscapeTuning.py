import gc
import os
import numpy as np
from loguru import logger
from scipy.ndimage import gaussian_filter1d
import dill as pickle

from settings.settings_analyze_efizz import Settings_ae as settings
from behave_analysis.analyze.EscapePattern.EscapeTuning import init_escape_tuning
from behave_analysis.analyze.EscapePattern.escape_pattern_utils import (
    select_onset_offsets_in_shift_vector,
    homing_escape_onsets,
    create_discretized_behave_var,
    build_shift_vector,
    residual_neural_matrix,
    parse_residual_string,
    parse_side,
)
from behave_analysis.analyze.EscapePattern.tuning_functions import compute_tuning_curves, compute_tuning_curves_no_trials
from behave_analysis.utils.creating_directories import make_directory


class ComputeEscapeTuning:
    """A class for computing the tuning to escape-related variables and storing them in the EscapeTuning dataclass
    1. Extract the data: this either concatenates all homings+escapes or exploration periods
    2. Compute the firing-by-variable tuning curves: variables can be % escape, distance to shelter
    3. Compute the leave one out reliability: NB this can only be done on homings+escapes
    4. Compute the statistical significance of the tuning curves via linear shift
    """

    def __init__(self, tuning, aefizz):

        # metadata
        self.ET = init_escape_tuning(settings, tuning)
        logger.info(f"{'Computing Residual of ' if 'residual' in self.ET.name.lower() else 'Computing '}Escape Pattern Tuning on {self.ET.tuning_var} during {self.ET.escape_pattern_time} periods")
        
        # load raw data
        self.load_data(aefizz)
        if "residual" in self.ET.name:
            self.load_residual_data(aefizz)
        
        # are we using escape, homings or both?
        self.ons, self.offs, self.esc_ons = homing_escape_onsets(aefizz, self.ET.escape_pattern_time)
        
        # build save path
        self.ET.savepath = make_directory(os.path.join(aefizz.session.base_path, aefizz.session.processed_path, "escape_tuning", self.ET.escape_pattern_time))
        # check that we're not trying to compute %escape tuning during explore periods
        # self.ET.escape_pattern_time == "explore": looking at exploration period
        # "escape" in self.ET.tuning_var: trying to compute tuning to % escape which can't be done in exploration
        if self.ET.escape_pattern_time == "explore" and "escape" in self.ET.tuning_var:
            raise ValueError("Cannot compute escape tuning during explore periods")
        pass

    def extract_data_and_tuning(self, aefizz):
        """This is a function that builds a matrix of neurons x time of activity in escape+homings or exploration
        and a behavioral variable of interest (var) discretized into bins (determined in settings)
        """
        # filter data based on time period (homing+escape or explore)
        filtering_vector, x, y = self.filter_data(aefizz)

        # compute behavioral variable
        self.ET.discretized_var = create_discretized_behave_var(
            aefizz, self.ET, x, y, self.ET.condition, 
            self.ET.tuning_var,
            self.ET.homing_vector if self.ET.escape_pattern_time == "homing&escape" else None
        )

        # how many trials are in each condition?
        trial_start_cond = self.condition[np.where(np.diff(filtering_vector) > 0)[0]]
        trial_n_cond = np.bincount(trial_start_cond.astype(int))

        # compute tuning curves for each neuron
        if self.ET.escape_pattern_time == "homing&escape":
            y_fit, R, fr, params, mat, loo = compute_tuning_curves(var=self.ET.discretized_var,
                                                                    escape_matrix=self.ET.neural_matrix,
                                                                    cond=self.ET.condition,
                                                                    bins=settings.escape_tuning_bins,
                                                                    filtering_vector=filtering_vector,
                                                                    n_cond=len(np.unique(self.ET.condition)),
                                                                    n_neur=self.ET.neural_matrix.shape[0],
                                                                    n_trials=max(trial_n_cond),
                                                                    avg="winsorized",
                                                                    fitting=settings.ep_gaussian_fitting,  # whether to fit a gaussian to each response curve
                                                                    loo=settings.ep_compute_loo_reliability,
                                                                )  # whether to compute leave one out reliability
            self.ET.mat_num_cond = mat
            if settings.ep_compute_loo_reliability:
                self.ET.loo_reliability_full = loo

        elif self.ET.escape_pattern_time == "explore":
            # TODO: there is a parrallelized version of this function that could be used instead, but has BUGS
            y_fit, R, fr, params = compute_tuning_curves_no_trials(var=self.ET.discretized_var,
                                                                    escape_matrix=self.ET.neural_matrix,
                                                                    cond=self.ET.condition,
                                                                    bins=settings.escape_tuning_bins,
                                                                    n_cond=len(np.unique(self.ET.condition)),
                                                                    n_neur=self.ET.neural_matrix.shape[0],
                                                                    fitting=settings.ep_gaussian_fitting,
                                                                )  # whether to fit a gaussian to each response curve

        self.ET.fr_full, self.ET.params_full, self.ET.y_fitted_full = fr, params, y_fit
        if settings.ep_gaussian_fitting:
            self.ET.R_full = R

    def compute_statistical_significance(self, aefizz):
        # TODO: refactor for residuals
        """This function performs linear shift stats on the tuning curves
        1. It builds a boolean shift vector of length time which subselect the central 1/3 of each condition
        2. It applies the shift vector to the neural and behavioral data to compute the null (the homings or explore periods need to be subselected carefully)
        3. The shift vector is shifted and the shifted vector is applied to the neural data"""

        # build shift vector to subselect central 1/3 of each condition
        shifts, shift_vector = build_shift_vector(aefizz, self.ET)

        # select which onsets and offsets to keep based on shift vector
        if self.ET.escape_pattern_time == "homing&escape":
            filtering_vector = select_onset_offsets_in_shift_vector(self.ET, shift_vector)
        elif self.ET.escape_pattern_time == "explore":
            filtering_vector = np.logical_and(self.ET.explore_vector, shift_vector)

        x = self.x[filtering_vector]
        y = self.y[filtering_vector]
        condition = self.condition[filtering_vector]

        # how many trials per condition?
        trial_start_cond = self.condition[np.where(np.diff(filtering_vector) > 0)[0]]
        trial_n_cond = np.bincount(trial_start_cond.astype(int))

        # compute behavioral variable
        discretized_var = create_discretized_behave_var(aefizz, self.ET, x, y, condition, 
                                      self.ET.tuning_var, 
                                      homing_vector = filtering_vector)
        self.ET.discretized_var_shift = discretized_var
        
        # initialize variables for output
        step_n, n_cond, n_neur, Nbins = len(shifts), len(np.unique(condition)), self.ET.neural_matrix.shape[0], settings.escape_tuning_bins
        self.ET.y_fitted_shift = np.full((step_n, n_cond, n_neur, Nbins), np.nan)  # conditions x neurons x n_bins
        if settings.ep_gaussian_fitting:
            self.ET.R_shift = np.zeros((step_n, n_neur, n_cond))  # neurons x conditions
        if settings.ep_compute_loo_reliability:
            self.ET.loo_shift = np.zeros((step_n, n_cond, n_neur))  # conditions x neurons
        self.ET.params_shifts = np.zeros((step_n, n_neur, n_cond, 6))  # neurons x conditions
        self.ET.fr_shift = np.full((step_n, n_cond, n_neur, Nbins), np.nan)
        self.ET.mat_shift_cond = np.full((step_n, n_cond, n_neur, max(trial_n_cond), Nbins), np.nan)

        # iterate over shifts, compute the tuning curves
        for s_idx, s in enumerate(shifts):

            shifted_vec = np.roll(filtering_vector, int(s))

            # filter data during homing+escape or explore periods and central third
            if "residual" not in self.ET.name:
                neural_matrix = self.fcm[shifted_vec, :].T
            elif "residual" in self.ET.name:
                neural_matrix = residual_neural_matrix(
                    neural_matrix_t1=self.fcm[shifted_vec, :].T, cond_t1=condition, var2_t1=self.ET.residual_shift_var2_t1, fr_var_t2=self.ET.residual_fr_shift_var2_t2
                )

            # compute the tuning curve on the unshifted, subselected data
            if self.ET.escape_pattern_time == "homing&escape":
                y, gf, fr, p, mat, reli = compute_tuning_curves(var=discretized_var,
                                                                escape_matrix=neural_matrix,
                                                                cond=condition,
                                                                bins=Nbins,
                                                                filtering_vector=filtering_vector,
                                                                n_cond=n_cond,
                                                                n_neur=n_neur,
                                                                n_trials=max(trial_n_cond),
                                                                avg="winsorized",
                                                                fitting=settings.ep_gaussian_fitting,  # whether to fit a gaussian to each response curve
                                                                loo=settings.ep_compute_loo_reliability,
                                                            )  # whether to compute leave one out reliability

                self.ET.mat_shift_cond[s_idx, :, :, : np.shape(mat)[2], :] = mat
                if settings.ep_compute_loo_reliability:
                    self.ET.loo_shift[s_idx, :, :] = reli

            elif self.ET.escape_pattern_time == "explore":
                y, gf, fr, p = compute_tuning_curves_no_trials(
                    var=discretized_var, escape_matrix=neural_matrix, cond=condition, bins=Nbins, n_cond=n_cond, n_neur=n_neur, fitting=settings.ep_gaussian_fitting
                )  # whether to fit a gaussian to each response curve

            self.ET.y_fitted_shift[s_idx, :, :, :], self.ET.fr_shift[s_idx, :, :, :], self.ET.params_shifts[s_idx, :, :, :] = y, fr, p
            if settings.ep_gaussian_fitting:
                self.ET.R_shift[s_idx, :, :] = gf

    def save_escape_tuning(self):
        """Save EscapeTuning dataclass to file"""
        # savepath building
        res = ''
        if "residual" in self.ET.name:
            res = 'residual_'
        filename = self.ET.savepath + os.sep + res + self.ET.tuning_var + "_" + str(settings.escape_tuning_bins) + "bins.pkl"
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
                this_y > aefizz.session.shelter_location[0][1], np.logical_and(this_x > aefizz.session.shelter_location[0][0], this_x < aefizz.session.shelter_location[1][0])
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

        # check that homingPeriod column exists
        if "homingPeriod" not in aefizz.video_df.columns:
            homing_period =  np.zeros(len(aefizz.video_df), dtype=bool)
            for onset, offset in zip(aefizz.homings_object.onset_frames, aefizz.homings_object.offset_frames):
                homing_period[onset - 1 : offset - 1] = True
        else:
            homing_period = aefizz.video_df["homingPeriod"].to_numpy()

        # extract explore periods: out of shelter, not in homing, not in escape
        explore_vector = np.logical_and(
            np.logical_and(homing_period == False, aefizz.video_df["EscapePeriod"].to_numpy() == False),
            aefizz.video_df["OutofshelterIdx"].to_numpy() == True,
        )

        # do we include stationary periods or not?
        if settings.escape_pattern_no_stationary:
            explore_vector = np.logical_and(explore_vector, (aefizz.video_df["speed"] > 0.5))
        
        # interpolate if needed
        if settings.escape_pattern_interpolation_mult > 1:
            explore_vector = np.repeat(explore_vector, settings.escape_pattern_interpolation_mult)

        return explore_vector

    def filter_data(self, aefizz):
        """This function filters the data based on the selected time periods (homing+escape or explore)"""

        # create filtering vector
        if self.ET.escape_pattern_time == "homing&escape":
            self.ET.homing_vector, self.ET.escape_vector = self.homing_escape_filtering_vector(
                aefizz
            )  # TODO: add options for long homings, correct homings, first/second leg, etc.
            filtering_vector = self.ET.homing_vector
        elif self.ET.escape_pattern_time == "explore":
            self.ET.explore_vector = self.explore_filtering_vector(aefizz)
            filtering_vector = self.ET.explore_vector

        # filter data during homing+escape or explore periods
        x = self.x[filtering_vector]
        y = self.y[filtering_vector]
        self.ET.condition = self.condition[filtering_vector]

        if "residual" not in self.ET.name:
            self.ET.neural_matrix = self.fcm[filtering_vector, :].T  # neurons x time

        # create the residual neural matrix if that's what we need
        elif "residual" in self.ET.name:
            self.ET.neural_matrix = residual_neural_matrix(
                neural_matrix_t1=self.fcm[filtering_vector, :].T, 
                cond_t1=self.ET.condition, 
                var2_t1=self.ET.residual_var2_t1, 
                fr_var_t2=self.ET.residual_fr_var2_t2
            )

        return filtering_vector, x, y

    def load_data(self, aefizz):
        """This function loads the data from the aefizz object and does any necessary preprocessing"""
        # gaussian filter
        fcm = gaussian_filter1d(aefizz.frame_by_cluster_matrix, 2, axis=0)

        # load behavioral data
        self.y = aefizz.video_df["mouse_y_position"].to_numpy()
        self.x = aefizz.video_df["mouse_x_position"].to_numpy()
        bar = aefizz.video_df["barrier_present"].to_numpy()
        barflip = aefizz.video_df["barrier_flipped"].to_numpy()

        # interpolate time
        if settings.escape_pattern_interpolation_mult > 1:
            current_time = np.arange(len(self.y))
            new_time = np.arange(0, len(self.y), 1 / settings.escape_pattern_interpolation_mult)
            # mouse position
            self.y = np.interp(new_time, current_time, self.y)
            self.x = np.interp(new_time, current_time, self.x)
            # experimental condition
            bar = np.repeat(bar, settings.escape_pattern_interpolation_mult)
            barflip = np.repeat(barflip, settings.escape_pattern_interpolation_mult)
            # neural data
            new_neur = np.zeros((len(self.y), np.shape(fcm)[1]))
            for i in np.arange(np.shape(fcm)[1]):
                new_neur[:, i] = np.interp(new_time, current_time, fcm[:, i])
            self.fcm = new_neur
            del new_neur
            gc.collect()
        # experimental condition vector
        self.condition = np.zeros(len(bar))
        self.condition[bar == True] += 1
        self.condition[barflip == True] += 1

    def load_residual_data(self, aefizz):
        """This function loads the data necessary to compute tuning in residual neural activity"""

        _, time_period1, tuning_var2, time_period2 = parse_residual_string(self.ET.name)

        # load behavioral data for var2 from ComputeTuning object
        # this is the discretized behavioral variable for tuning_var2 in time_period1
        path = os.path.join(aefizz.session.base_path, aefizz.session.processed_path, "escape_tuning", time_period1)
        filename = path + os.sep + tuning_var2 + "_" + str(self.ET.nbins) + "bins.pkl"
        # check file exists
        if not os.path.exists(filename):
            logger.warning(f"Escape tuning to {tuning_var2} in {time_period1} file not found, computing now...")
            computeET = ComputeEscapeTuning(tuning_var2 + " in " + time_period1, aefizz=aefizz)
            computeET.extract_data_and_tuning(aefizz=aefizz)
            computeET.compute_statistical_significance(aefizz=aefizz)
            computeET.save_escape_tuning()

        with open(filename, "rb") as f:
            CT_var2 = pickle.load(f)
        self.ET.residual_var2_t1 = CT_var2.discretized_var
        self.TT.residual_shift_var2_t1 = CT_var2.discretized_var_shift

        # load tuning data for var2 in exploration from ComputeTuning object
        # this is the firing rate in the tuning curve to var2 in time_period2
        path = os.path.join(aefizz.session.base_path, aefizz.session.processed_path, "escape_tuning", time_period2)
        filename = path + os.sep + tuning_var2 + "_" + str(self.ET.nbins) + "bins.pkl"
        # check file exists
        if not os.path.exists(filename):
            logger.warning(f"Escape tuning to {tuning_var2} in {time_period2} file not found, computing now...")
            computeET = ComputeEscapeTuning(tuning_var2 + " in " + time_period2, aefizz=aefizz)
            computeET.extract_data_and_tuning(aefizz=aefizz)
            computeET.compute_statistical_significance(aefizz=aefizz)
            computeET.save_escape_tuning()

        with open(filename, "rb") as f:
            CT_var2 = pickle.load(f)
        self.ET.residual_fr_var2_t2 = CT_var2.fr_full
        mid = int(np.shape(CT_var2.y_fitted_shift)[0]/2)
        self.ET.residual_fr_shift_var2_t2 = CT_var2.fr_shift[mid, :,:,:]
