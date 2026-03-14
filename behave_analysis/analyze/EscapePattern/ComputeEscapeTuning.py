import gc
import os
import numpy as np
from loguru import logger
from scipy.ndimage import gaussian_filter1d
import dill as pickle
import pandas as pd
from dataclasses import asdict

from settings.settings_analyze_efizz import Settings_ae as settings
from behave_analysis.analyze.EscapePattern.EscapeTuning import init_escape_tuning
from behave_analysis.analyze.EscapePattern.escape_pattern_utils import (
    select_onset_offsets_in_shift_vector,
    homing_escape_onsets,
    create_discretized_behave_var,
    build_shift_vector,
    residual_neural_matrix,
    parse_residual_string,
    saving_path_and_file,
    homing_escape_boolean_vectors,
)
from behave_analysis.analyze.EscapePattern.tuning_functions import compute_tuning_curves, compute_tuning_curves_no_trials
from behave_analysis.utils.creating_directories import make_directory
from behave_analysis.analyze.results_database_utils import check_database_for_same_run, add_run_to_database, settings_to_check


class ComputeEscapeTuning:
    """A class for computing the tuning to escape-related variables and storing them in the EscapeTuning dataclass
    1. Extract the data: this either concatenates all homings+escapes or exploration periods
    1b. EscapeTuning can also be computed on residual neural activity after removing variance explained by another variable.
        In this case, we will check that Tuning has been computed for the variables of interest and a matrix of residual neural activity will be created
    2. Compute the firing-by-variable tuning curves: variables can be % escape, distance to shelter TODO: other variables like speed can be added
    3. Compute the leave-one-out-reliability of the tuning: NB this can only be done on homings+escapes
    4. Compute the statistical significance of the tuning curves via linear shift
    """

    def __init__(self, tuning, aefizz):

        # metadata
        self.ET = init_escape_tuning(settings, tuning)
        self.aefizz = aefizz

        # check that we're not trying to compute %escape tuning during explore periods
        # self.ET.escape_pattern_time == "explore": looking at exploration period
        # "escape" in self.ET.tuning_var: trying to compute tuning to % escape which can't be done in exploration
        if (self.ET.escape_pattern_time == "explore") and ("escape" in self.ET.tuning_var):
            raise ValueError("Cannot compute escape tuning during explore periods")
        pass

        # build save path to dump data in
        self.ET.savepath = make_directory(
            os.path.join(
                self.aefizz.session.base_path,
                self.aefizz.session.processed_path,
                "models",
                "escape_tuning",
            )
        )

        self.database, self.do_analysis, self.hexaname = check_database_for_same_run(
            db_settings={"variable": tuning, **settings_to_check(self.aefizz.settings, ["ep_", "linshift"])},
            results_csv_name=self.ET.savepath + os.sep + "EscapePattern_results.csv",
            settings=self.aefizz.settings,
        )

    def prepare_data(self):
        # get raw neural and behavioral data from aefizz
        self.preprocess_data()
        if "residual" in self.ET.name:
            # loads discretized behavioral variables and firing rate tuning curves for residual computation
            self.load_residual_data()

        if "homing" in self.ET.escape_pattern_time or "escape" in self.ET.escape_pattern_time:
            # find onsets of runs based on escape pattern time ('homings' or 'homing&escape')
            onset_dict = homing_escape_onsets(self.aefizz, self.ET.escape_pattern_time)
            self.ons, self.offs, self.esc_ons = onset_dict["ons"], onset_dict["offs"], onset_dict["esc_ons"]
            if len(self.ons) == 0:
                self.insufficient_data = True
                return
            self.insufficient_data = False

    def filter_data_and_compute_tuning(self):
        """This is a function that builds a matrix of neurons x time of activity in escape+homings or exploration
        and a behavioral variable of interest (var) discretized into bins (determined in settings)
        """

        h_and_e = False
        if "homing" in self.ET.escape_pattern_time or "escape" in self.ET.escape_pattern_time:
            h_and_e = True

        # create a filtering vector based on time period (homing+escape or explore)
        filtering_vector, x, y = self.filter_data()

        # extract behavioral variable that we compute the tuning to
        self.ET.discretized_var = create_discretized_behave_var(
            self.aefizz, x, y, self.ET.condition, tuning_var=self.ET.tuning_var, time_mask_vector=filtering_vector, bin_edges=self.ET.bin_edges
        )

        # compute tuning curves for each neuron
        if h_and_e:

            # how many trials are in each condition?
            trial_start_cond = self.condition[np.where(np.diff(filtering_vector) > 0)[0]]
            trial_n_cond = np.bincount(trial_start_cond.astype(int))

            y_fit, R, fr, params, mat, loo = compute_tuning_curves(
                var=self.ET.discretized_var,
                escape_matrix=self.ET.neural_matrix,
                cond=self.ET.condition,
                bins=settings.ep_bins,
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
            y_fit, R, fr, params = compute_tuning_curves_no_trials(
                var=self.ET.discretized_var,
                escape_matrix=self.ET.neural_matrix,
                cond=self.ET.condition,
                bins=settings.ep_bins,
                n_cond=len(np.unique(self.ET.condition)),
                n_neur=self.ET.neural_matrix.shape[0],
                fitting=settings.ep_gaussian_fitting,
            )  # whether to fit a gaussian to each response curve

        self.ET.fr_full, self.ET.params_full, self.ET.y_fitted_full = fr, params, y_fit
        if settings.ep_gaussian_fitting:
            self.ET.R_full = R

    def compute_statistical_significance(self):
        """This function performs linear shift stats on the tuning curves
        1. It builds a boolean shift vector of length time which subselect the central 1/3 of each condition
        2. It applies the shift vector to the neural and behavioral data to compute the null (the homings or explore periods need to be subselected carefully)
        3. The shift vector is shifted and the shifted vector is applied to the neural data"""

        # build shift vector to subselect central 1/3 of each condition
        self.ET.shifts, shift_vector = build_shift_vector(self.aefizz, self.ET)

        # select which onsets and offsets to keep based on shift vector
        if "homing" in self.ET.escape_pattern_time or "escape" in self.ET.escape_pattern_time:
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
        self.ET.discretized_var_shift = create_discretized_behave_var(self.aefizz, x, y, condition, self.ET.tuning_var, time_mask_vector=filtering_vector, bin_edges=self.ET.bin_edges)

        # initialize variables for output
        step_n, n_cond, n_neur, Nbins = len(self.ET.shifts), len(np.unique(condition)), self.ET.neural_matrix.shape[0], settings.ep_bins
        self.ET.y_fitted_shift = np.full((step_n, n_cond, n_neur, Nbins), np.nan)  # conditions x neurons x n_bins
        if settings.ep_gaussian_fitting:
            self.ET.R_shift = np.zeros((step_n, n_neur, n_cond))  # neurons x conditions
        if settings.ep_compute_loo_reliability:
            self.ET.loo_shift = np.zeros((step_n, n_cond, n_neur))  # conditions x neurons
        self.ET.params_shifts = np.zeros((step_n, n_neur, n_cond, 6))  # neurons x conditions
        self.ET.fr_shift = np.full((step_n, n_cond, n_neur, Nbins), np.nan)
        self.ET.mat_shift_cond = np.full((step_n, n_cond, n_neur, max(trial_n_cond), Nbins), np.nan)

        # iterate over shifts, compute the tuning curves
        for s_idx, s in enumerate(self.ET.shifts):

            shifted_vec = np.roll(filtering_vector, int(s))

            # filter data during homing+escape or explore periods and central third
            if "residual" not in self.ET.name:
                neural_matrix = self.fcm[shifted_vec, :].T
            elif "residual" in self.ET.name:
                neural_matrix = residual_neural_matrix(
                    neural_matrix_t1=self.fcm[shifted_vec, :].T, cond_t1=condition, var2_t1=self.ET.residual_shift0_var2_t1, fr_var2_t2=self.ET.residual_fr_shift0_var2_t2
                )

            # compute the tuning curve on the unshifted, subselected data
            if "homing" in self.ET.escape_pattern_time or "escape" in self.ET.escape_pattern_time:
                y, gf, fr, p, mat, reli = compute_tuning_curves(
                    var=self.ET.discretized_var_shift,
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
                    var=self.ET.discretized_var_shift, escape_matrix=neural_matrix, cond=condition, bins=Nbins, n_cond=n_cond, n_neur=n_neur, fitting=settings.ep_gaussian_fitting
                )  # whether to fit a gaussian to each response curve

            self.ET.y_fitted_shift[s_idx, :, :, :], self.ET.fr_shift[s_idx, :, :, :], self.ET.params_shifts[s_idx, :, :, :] = y, fr, p
            if settings.ep_gaussian_fitting:
                self.ET.R_shift[s_idx, :, :] = gf

    def save_escape_tuning(self, variable, return_dict=False):
        """Save EscapeTuning dataclass to file"""
        filename = os.path.join(self.ET.savepath, "EPtuning_" + self.hexaname)
        # build results dict and save
        results_dict = asdict(self.ET)
        np.savez(os.path.join(filename + "_results.npz"), **results_dict, allow_pickle=True)
        settings = asdict(self.aefizz.settings)
        np.savez(filename + "_settings.npz", **settings, allow_pickle=True)
        # add results to database
        db_settings = {"variable": variable, 
                       "insufficient_data": self.insufficient_data,
                       **settings_to_check(self.aefizz.settings, ["ep_", "linshift"])}
        add_run_to_database(self.database, 
                            db_settings, 
                            self.ET.savepath + os.sep + "EscapePattern_results.csv", 
                            self.hexaname)

        if return_dict:
            return results_dict

    # ----------------------------Data loading and processing functions----------------------------

    def homing_escape_filtering_vector(self):
        """This function builds two boolean vectors of length time which are True when the mouse is in homing or escape periods
        It removes any time after shelter entry within each homing
        It uses the array of onsets and offsets created in homing_escape_onsets function
        (this could be only homings, homings+escapes, long homings, etc. depending on context in tuning passed to ComputeEscapeTuning)"""

        homing_vector = np.zeros_like(self.condition, dtype=bool)
        escape_vector = np.zeros_like(self.condition, dtype=bool)

        # iterate over homings
        for on, of in zip(self.ons, self.offs):
            on = int(on)
            of = int(of)

            if on in self.esc_ons:
                esc = True
            else:
                esc = False

            # extract mouse position in the run
            this_y = self.aefizz.video_df["mouse_y_position"].to_numpy()[on:of]
            this_x = self.aefizz.video_df["mouse_x_position"].to_numpy()[on:of]

            # crop homings at shelter entry
            # find actual length of time until mouse is in shelter
            in_shelt = np.logical_and(
                this_y > self.aefizz.session.shelter_location[0][1],
                np.logical_and(this_x > self.aefizz.session.shelter_location[0][0], this_x < self.aefizz.session.shelter_location[1][0]),
            )
            shelter_entry = np.where(np.diff(in_shelt) > 0)[0][0] + 1 if np.any(np.diff(in_shelt) > 0) else len(in_shelt)
            of = on + shelter_entry

            # do we want to crop homings into first and second leg?

            if self.aefizz.settings.ep_interpolation_mult > 1:
                on = on * self.aefizz.settings.ep_interpolation_mult
                of = of * self.aefizz.settings.ep_interpolation_mult

            homing_vector[on:of] = True
            escape_vector[on:of] = True if esc else False

        return homing_vector, escape_vector

    def filtering_vector_exploration(self):
        """This function builds a boolean vector of length time which is True when the mouse is exploring
        i.e. not in homing or escape periods and is outside of the shelter
        TODO: double check the logic!"""

        # check that homingPeriod column exists
        if "homingPeriod" not in self.aefizz.video_df.columns:
            # NB: as soon as postprocess is rerun, this logic should be fixed and applied there as well
            homing_period, escape_period = homing_escape_boolean_vectors(self.aefizz)
        else:
            homing_period = self.aefizz.video_df["homingPeriod"].to_numpy()
            escape_period = self.aefizz.video_df["EscapePeriod"].to_numpy()

        # extract explore periods: out of shelter, not in homing, not in escape
        explore_vector = np.logical_and(
            np.logical_and(homing_period == False, escape_period == False),
            self.aefizz.video_df["OutofshelterIdx"].to_numpy() == True,
        )

        # do we include stationary periods or not?
        if settings.ep_no_stationary:
            explore_vector = np.logical_and(explore_vector, (self.aefizz.video_df["speed"] > 0.5))

        # interpolate if needed
        if settings.ep_interpolation_mult > 1:
            explore_vector = np.repeat(explore_vector, settings.ep_interpolation_mult)

        return explore_vector

    def filter_data(self):
        """This function filters the data based on the selected time periods (homing+escape or explore)"""

        # create time filtering vector
        if "homing" in self.ET.escape_pattern_time or "escape" in self.ET.escape_pattern_time:
            self.ET.homing_vector, self.ET.escape_vector = self.homing_escape_filtering_vector()
            # TODO: add options for long homings, correct homings, first/second leg, etc.
            filtering_vector = self.ET.homing_vector
        elif self.ET.escape_pattern_time == "explore":
            self.ET.explore_vector = self.filtering_vector_exploration()
            filtering_vector = self.ET.explore_vector

        # filter behavioral data during selected time periods
        x = self.x[filtering_vector]
        y = self.y[filtering_vector]
        self.ET.condition = self.condition[filtering_vector]

        # filter neural data during selected time periods
        if "residual" not in self.ET.name:
            self.ET.neural_matrix = self.fcm[filtering_vector, :].T  # neurons x time

        # create the residual neural matrix if that's what we need
        elif "residual" in self.ET.name:
            self.ET.neural_matrix = residual_neural_matrix(
                neural_matrix_t1=self.fcm[filtering_vector, :].T, cond_t1=self.ET.condition, var2_t1=self.ET.residual_var2_t1, fr_var2_t2=self.ET.residual_fr_var2_t2
            )

        return filtering_vector, x, y

    def preprocess_data(self):
        """This function organizes the data (loaded into the analyze efizz object) and does any necessary preprocessing"""
        # gaussian filter
        if not hasattr(self.aefizz, "frame_by_cluster_matrix"):
            self.aefizz.frame_by_cluster_matrix = np.load(
                os.path.join(self.aefizz.session.base_path, self.aefizz.session.processed_path) + "\\" + "frame_by_" + self.aefizz.cluster_type + "_cluster_matrix.npy"
            )
        fcm = gaussian_filter1d(self.aefizz.frame_by_cluster_matrix, 2, axis=0)

        # load behavioral data
        self.y = self.aefizz.video_df["mouse_y_position"].to_numpy()
        self.x = self.aefizz.video_df["mouse_x_position"].to_numpy()
        bar = self.aefizz.video_df["barrier_present"].to_numpy()
        barflip = self.aefizz.video_df["barrier_flipped"].to_numpy()

        # interpolate time
        if self.aefizz.settings.ep_interpolation_mult > 1:
            current_time = np.arange(len(self.y))
            new_time = np.arange(0, len(self.y), 1 / self.aefizz.settings.ep_interpolation_mult)
            # mouse position
            self.y = np.interp(new_time, current_time, self.y)
            self.x = np.interp(new_time, current_time, self.x)
            # experimental condition
            bar = np.repeat(bar, self.aefizz.settings.ep_interpolation_mult)
            barflip = np.repeat(barflip, self.aefizz.settings.ep_interpolation_mult)
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

    def load_residual_data(self):
        """This function loads the data necessary to compute tuning in residual neural activity.
        It checks whether tuning to <var2> in <context1> and <context2> has been computed, and if not, computes it.
        It extracts:
            1. the discretized behavioral variable for var2 in time_period1 (both the full time window and the central third used for linear shift stats)
            2. the firing rate tuning curve to var2 in time_period2 (both the full tuning curve and the one aligned to shift 0 used for linear shift stats)"""

        _, time_period1, tuning_var2, time_period2 = parse_residual_string(self.ET.name)

        # load behavioral data for var2 from ComputeTuning object
        # this is the discretized behavioral variable for tuning_var2 in time_period1
        CT_var2_t1 = load_or_compute_escape_tuning(self.aefizz, tuning_var2 + " in " + time_period1)

        self.ET.residual_var2_t1 = CT_var2_t1['discretized_var']
        self.ET.residual_shift0_var2_t1 = CT_var2_t1['discretized_var_shift']

        # load tuning data for var2 in exploration from ComputeTuning object
        # this is the firing rate in the tuning curve to var2 in time_period2
        CT_var2_t2 = load_or_compute_escape_tuning(self.aefizz, tuning_var2 + " in " + time_period2)

        self.ET.residual_fr_var2_t2 = CT_var2_t2['fr_full']
        if 'shifts' in CT_var2_t2.keys():
            mid = np.where(CT_var2_t2['shifts'] == 0)[0][0]
        else:
            mid = int(np.shape(CT_var2_t2['y_fitted_shift'])[0] / 2)
        self.ET.residual_fr_shift0_var2_t2 = CT_var2_t2['fr_shift'][mid, :, :, :]

# -----------------------------Helper functions for loading or running compitation ----------------------------

def load_or_compute_escape_tuning(aefizz, variable):
    """
    This function loads in or computes the escape tuning curves for a given variable
    INPUTS:
        aefizz: AnalyzeEfizz object
        tuning_var: string of the variable to compute the tuning curve for
    """
    savepath = make_directory(
        os.path.join(
            aefizz.session.base_path,
            aefizz.session.processed_path,
            "models",
            "escape_tuning",
        )
    )
    logger.info(f"checking for existing results to {variable} in database...")
    _, do_analysis, hexaname = check_database_for_same_run(
        db_settings={"variable": variable, **settings_to_check(aefizz.settings, ["ep_", "linshift"])},
        results_csv_name=savepath + os.sep + "EscapePattern_results.csv",
        settings=aefizz.settings,
    )
    # check file exists
    if do_analysis == False:
        EP_dict = np.load(savepath + os.sep + "EPtuning_" + hexaname + "_results.npz", allow_pickle=True)
    else:
        logger.warning(f"Escape tuning to {variable} file not found, computing now...")
        check_aefizz_completeness(aefizz)
        computeET = ComputeEscapeTuning(variable, aefizz)
        computeET.prepare_data()
        if computeET.insufficient_data:
            logger.warning(f"Insufficient data for {variable}, saving empty results")
            computeET.save_escape_tuning(variable)
            return {}
        computeET.filter_data_and_compute_tuning()
        computeET.compute_statistical_significance()
        EP_dict = computeET.save_escape_tuning(variable, return_dict=True)

    return EP_dict


def check_aefizz_completeness(aefizz):
    """This function checks that the aefizz object has all the necessary data and preprocessing to compute escape tuning curves.
    If not, it raises an error and specifies what is missing."""

    if not hasattr(aefizz, "frame_by_cluster_matrix"):
        aefizz.frame_by_cluster_matrix = np.load(
            os.path.join(aefizz.session.base_path, aefizz.session.processed_path) + "\\" + "frame_by_" + aefizz.cluster_type + "_cluster_matrix.npy"
        )
    if not hasattr(aefizz, "video_df"):
        import polars as pl

        aefizz.video_df = pl.read_csv(os.path.join(aefizz.session.base_path, aefizz.session.processed_path) + "\\" + "full_video_dataframe.csv")

    if not hasattr(aefizz, "cluster_Ids"):
        aefizz.cluster_Ids = np.load(str(os.path.join(aefizz.session.base_path, aefizz.session.processed_path) + "/" + aefizz.cluster_type + "_cluster_Ids.npy"))

    if not hasattr(aefizz, "homings_object"):
        homing_path = os.path.join(aefizz.session.base_path, aefizz.session.processed_path, "homings", "homings_obj.pkl")
        with open(homing_path, "rb") as f:
            aefizz.homings_object = pickle.load(f)

    if not hasattr(aefizz, "escape_object"):
        escape_path = os.path.join(aefizz.session.base_path, aefizz.session.processed_path, "escapes", "escapes_obj.pkl")
        with open(escape_path, "rb") as f:
            aefizz.escape_object = pickle.load(f)
