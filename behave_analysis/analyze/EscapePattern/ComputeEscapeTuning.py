import gc
import os
import numpy as np
from loguru import logger
from scipy.ndimage import gaussian_filter1d
import dill as pickle
import pandas as pd
import polars as pl
from dataclasses import asdict

from behave_analysis.analyze.CCA.find_shelter_exit_and_runs import find_shelter_exit_runs
# from settings.settings_analyze_efizz import Settings_ae as settings
from behave_analysis.analyze.EscapePattern.EscapeTuning import init_escape_tuning
from behave_analysis.analyze.EscapePattern.escape_pattern_utils import (
    select_onset_offsets_in_shift_vector,
    homing_escape_onsets,
    create_discretized_behave_var,
    build_shift_vector,
    residual_neural_matrix,
    parse_residual_string,
    homing_escape_boolean_vectors,
    homing_escape_filtering_vector,
)
from behave_analysis.analyze.EscapePattern.tuning_functions import compute_tuning_curves, compute_tuning_curves_no_trials
from behave_analysis.utils.creating_directories import make_directory
from behave_analysis.analyze.results_database_utils import check_database_for_same_run, add_run_to_database, generate_run_id, settings_to_check
from behave_analysis.analyze.PlaceCells.PlaceCells import PlaceCells, COLUMNS_TO_KEEP

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
        self.ET = init_escape_tuning(aefizz.settings, tuning)
        self.aefizz = aefizz
        self.insufficient_data = False
        self.settings = aefizz.settings

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
        logger.info(f"checking for existing results to {tuning} in EP tuning database...")
        self.database, self.do_analysis, self.hexaname = check_database_for_same_run(
            db_settings={"variable": tuning, **settings_to_check(self.settings, ["ep_", "linshift"])},
            results_csv_name=self.ET.savepath + os.sep + "EscapePattern_results.csv",
            settings=self.settings,
        )

    def prepare_data(self):
        # get raw neural and behavioral data from aefizz
        self.preprocess_data()
        if "residual" in self.ET.name:
            # loads discretized behavioral variables and firing rate tuning curves for residual computation
            self.load_data_for_residual()

        if "homing" in self.ET.escape_pattern_time or "escape" in self.ET.escape_pattern_time:
            # find onsets of runs based on escape pattern time ('homings' or 'homing&escape')
            self.onset_dict = homing_escape_onsets(self.aefizz, self.ET.escape_pattern_time)
            if len(self.onset_dict["ons"]) == 0:
                self.insufficient_data = True
                return
            self.insufficient_data = False
        
        if self.ET.escape_pattern_time == "shelter_outing":
            shelter_outing_vector = find_shelter_exit_runs(self.aefizz.video_df, min_speed_cm_s=3.0, min_distance_cm=20.0)
            self.onset_dict = {"ons": np.where(np.diff(shelter_outing_vector.astype(int)) > 0)[0] + 1,
                               "offs": np.where(np.diff(shelter_outing_vector.astype(int)) < 0)[0] + 1}

    def filter_data_and_compute_tuning(self):
        """This is a function that builds a matrix of neurons x time of activity in escape+homings or exploration
        and a behavioral variable of interest (var) discretized into bins (determined in settings)
        """

        trial_based = False
        if "homing" in self.ET.escape_pattern_time or "escape" in self.ET.escape_pattern_time or self.ET.escape_pattern_time == "shelter_outing":
            trial_based = True

        # create a filtering vector based on time period (homing+escape or explore)
        filtering_vector, x, y = self.filter_data()

        # extract behavioral variable that we compute the tuning to
        self.ET.discretized_var = create_discretized_behave_var(
            self.aefizz, x, y, self.ET.condition, tuning_var=self.ET.tuning_var, time_mask_vector=filtering_vector, interpolation=True if self.ET.tuning_var == "speed" else False
        )

        # compute tuning curves for each neuron
        if trial_based:

            # how many trials are in each condition?
            trial_start_cond = self.condition[np.where(np.diff(filtering_vector) > 0)[0]]
            trial_n_cond = np.bincount(trial_start_cond.astype(int))

            y_fit, R, fr, params, mat, loo = compute_tuning_curves(
                var=self.ET.discretized_var,
                escape_matrix=self.ET.neural_matrix,
                cond=self.ET.condition,
                bins=self.settings.ep_bins,
                filtering_vector=filtering_vector,
                n_cond=len(np.unique(self.ET.condition)),
                n_neur=self.ET.neural_matrix.shape[0],
                n_trials=max(trial_n_cond),
                avg="winsorized",
                fitting=self.settings.ep_gaussian_fitting,  # whether to fit a gaussian to each response curve
                loo=self.settings.ep_compute_loo_reliability,
            )  # whether to compute leave one out reliability
            self.ET.mat_num_cond = mat
            if self.settings.ep_compute_loo_reliability:
                self.ET.loo_reliability_full = loo

        elif self.ET.escape_pattern_time == "explore":
            # TODO: there is a parrallelized version of this function that could be used instead, but has BUGS
            y_fit, R, fr, params = compute_tuning_curves_no_trials(
                var=self.ET.discretized_var,
                escape_matrix=self.ET.neural_matrix,
                cond=self.ET.condition,
                bins=self.settings.ep_bins,
                n_cond=len(np.unique(self.ET.condition)),
                n_neur=self.ET.neural_matrix.shape[0],
                fitting=self.settings.ep_gaussian_fitting,
            )  # whether to fit a gaussian to each response curve

        self.ET.fr_full, self.ET.params_full, self.ET.y_fitted_full = fr, params, y_fit
        if self.settings.ep_gaussian_fitting:
            self.ET.R_full = R

    def compute_statistical_significance(self):
        """This function performs linear shift stats on the tuning curves
        1. It builds a boolean shift vector of length time which subselect the central 1/3 of each condition
        2. It applies the shift vector to the neural and behavioral data to compute the null (the homings or explore periods need to be subselected carefully)
        3. The shift vector is shifted and the shifted vector is applied to the neural data"""

        # build shift vector to subselect central 1/3 of each condition
        self.ET.shifts, shift_vector = build_shift_vector(self.aefizz, self.ET)

        # select which onsets and offsets to keep based on shift vector
        if "homing" in self.ET.escape_pattern_time or "escape" in self.ET.escape_pattern_time or self.ET.escape_pattern_time == "shelter_outing":
            filtering_vector = select_onset_offsets_in_shift_vector(shift_vector,
                                                ons = (self.onset_dict["ons"] * self.settings.ep_interpolation_mult).astype(int),  # homing onsets
                                                offs = (self.onset_dict["offs"] * self.settings.ep_interpolation_mult).astype(int))  # homing offsets)
        elif self.ET.escape_pattern_time == "explore":
            filtering_vector = np.logical_and(self.ET.explore_vector, shift_vector)

        x = self.x[filtering_vector]
        y = self.y[filtering_vector]
        condition = self.condition[filtering_vector]

        # how many trials per condition?
        trial_start_cond = self.condition[np.where(np.diff(filtering_vector) > 0)[0]]
        trial_n_cond = np.bincount(trial_start_cond.astype(int))

        # compute behavioral variable
        self.discretized_var_shift = create_discretized_behave_var(self.aefizz, x, y, condition, self.ET.tuning_var, time_mask_vector=filtering_vector, interpolation=True if self.ET.tuning_var == "speed" else False)

        # initialize variables for output
        step_n, n_cond, n_neur, Nbins = len(self.ET.shifts), len(np.unique(condition)), self.ET.neural_matrix.shape[0], self.settings.ep_bins
        self.ET.y_fitted_shift = np.full((step_n, n_cond, n_neur, Nbins), np.nan)  # conditions x neurons x n_bins
        if self.settings.ep_gaussian_fitting:
            self.ET.R_shift = np.zeros((step_n, n_neur, n_cond))  # neurons x conditions
        if self.settings.ep_compute_loo_reliability:
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
                neural_matrix = residual_neural_matrix(neural_matrix_t1=self.fcm[shifted_vec, :].T, 
                                                       cond_t1=condition, 
                                                       var2_t1= self.ET.residual_var2_all_time[shifted_vec], # self.ET.residual_shift0_var2_t1, 
                                                       fr_var2_t2=self.ET.residual_fr_shift0_var2_t2)
                

            # compute the tuning curve on the unshifted, subselected data
            if "homing" in self.ET.escape_pattern_time or "escape" in self.ET.escape_pattern_time or self.ET.escape_pattern_time == "shelter_outing":
                y, gf, fr, p, mat, reli = compute_tuning_curves(
                    var=self.discretized_var_shift,
                    escape_matrix=neural_matrix,
                    cond=condition,
                    bins=Nbins,
                    filtering_vector=filtering_vector,
                    n_cond=n_cond,
                    n_neur=n_neur,
                    n_trials=max(trial_n_cond),
                    avg="winsorized",
                    fitting=self.settings.ep_gaussian_fitting,  # whether to fit a gaussian to each response curve
                    loo=self.settings.ep_compute_loo_reliability,
                )  # whether to compute leave one out reliability

                self.ET.mat_shift_cond[s_idx, :, :, : np.shape(mat)[2], :] = mat
                if self.settings.ep_compute_loo_reliability:
                    self.ET.loo_shift[s_idx, :, :] = reli

            elif self.ET.escape_pattern_time == "explore":
                y, gf, fr, p = compute_tuning_curves_no_trials(
                    var=self.discretized_var_shift, escape_matrix=neural_matrix, cond=condition, bins=Nbins, n_cond=n_cond, n_neur=n_neur, fitting=self.settings.ep_gaussian_fitting
                )  # whether to fit a gaussian to each response curve

            self.ET.y_fitted_shift[s_idx, :, :, :], self.ET.fr_shift[s_idx, :, :, :], self.ET.params_shifts[s_idx, :, :, :] = y, fr, p
            if self.settings.ep_gaussian_fitting:
                self.ET.R_shift[s_idx, :, :] = gf

    def save_escape_tuning(self, variable, return_dict=False):
        """Save EscapeTuning dataclass to file"""
        filename = os.path.join(self.ET.savepath, "EPtuning_" + self.hexaname)
        # build results dict and save
        results_dict = asdict(self.ET)
        np.savez(os.path.join(filename + "_results.npz"), **results_dict, allow_pickle=True)
        settings = asdict(self.settings)
        np.savez(filename + "_settings.npz", **settings, allow_pickle=True)
        # add results to database
        db_settings = {"variable": variable, 
                       "insufficient_data": self.insufficient_data,
                       **settings_to_check(self.settings, ["ep_", "linshift"])}
        add_run_to_database(self.database, 
                            db_settings, 
                            self.ET.savepath + os.sep + "EscapePattern_results.csv", 
                            self.hexaname)

        if return_dict:
            return results_dict

    # ----------------------------Data loading and processing functions----------------------------

    def filtering_vector_exploration(self):
        """This function builds a boolean vector of length time which is True when the mouse is exploring
        i.e. not in homing or escape periods and is outside of the shelter"""

        # check that homingPeriod column exists
        if "homingPeriod" not in self.aefizz.video_df.columns:
            # NB: as soon as postprocess is rerun, this logic should be fixed and applied there as well
            homing_period = homing_escape_boolean_vectors(self.aefizz.homings_object, len(self.aefizz.video_df))
            escape_period = homing_escape_boolean_vectors(self.aefizz.escape_object, len(self.aefizz.video_df))
        else:
            homing_period = self.aefizz.video_df["homingPeriod"].to_numpy()
            escape_period = self.aefizz.video_df["EscapePeriod"].to_numpy()

        # extract explore periods: out of shelter, not in homing, not in escape
        explore_vector = np.logical_and(
            np.logical_and(homing_period == False, escape_period == False),
            self.aefizz.video_df["OutofshelterIdx"].to_numpy() == True,
        )

        # do we include stationary periods or not?
        if self.settings.ep_no_stationary:
            explore_vector = np.logical_and(explore_vector, (self.aefizz.video_df["speed"] > 0.5))

        # interpolate if needed
        if self.settings.ep_interpolation_mult > 1:
            explore_vector = np.repeat(explore_vector, self.settings.ep_interpolation_mult)

        return explore_vector

    def filter_data(self):
        """This function filters the data based on the selected time periods (homing+escape or explore)"""

        # create time filtering vector
        if "homing" in self.ET.escape_pattern_time or "escape" in self.ET.escape_pattern_time:
            self.ET.homing_vector, self.ET.escape_vector = homing_escape_filtering_vector(nframes=len(self.condition), onset_dict=self.onset_dict, xpos=self.x, ypos=self.y, shelter_location=self.aefizz.session.shelter_location, interpolation_mult=self.aefizz.settings.ep_interpolation_mult)
            filtering_vector = self.ET.homing_vector
        if self.ET.escape_pattern_time == "shelter_outing":
            filtering_vector = np.zeros_like(self.condition, dtype=bool)
            for on, of in zip(self.onset_dict["ons"], self.onset_dict["offs"]):
                if self.settings.ep_interpolation_mult > 1:
                    on *= self.settings.ep_interpolation_mult
                    of *= self.settings.ep_interpolation_mult
                filtering_vector[on:of] = True
            self.ET.shelter_outing_vector = filtering_vector
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
            self.ET.neural_matrix = residual_neural_matrix(neural_matrix_t1=self.fcm[filtering_vector, :].T, 
                                                           cond_t1=self.ET.condition, 
                                                           var2_t1=self.ET.residual_var2_all_time[filtering_vector], 
                                                           fr_var2_t2=self.ET.residual_fr_var2_t2)

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
        if self.settings.ep_interpolation_mult > 1:
            current_time = np.arange(len(self.y))
            new_time = np.arange(0, len(self.y), 1 / self.settings.ep_interpolation_mult)
            # mouse position
            self.y = np.interp(new_time, current_time, self.y)
            self.x = np.interp(new_time, current_time, self.x)
            # experimental condition
            bar = np.repeat(bar, self.settings.ep_interpolation_mult)
            barflip = np.repeat(barflip, self.settings.ep_interpolation_mult)
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

    def load_data_for_residual(self):
        """This function loads the data necessary to compute tuning in residual neural activity.
        It checks whether tuning to <var2> in <context1> and <context2> has been computed, and if not, computes it.
        It extracts:
            1. the discretized behavioral variable for var2 in time_period1 (both the full time window and the central third used for linear shift stats)
            2. the firing rate tuning curve to var2 in time_period2 (both the full tuning curve and the one aligned to shift 0 used for linear shift stats)"""

        _, time_period1, tuning_var2, time_period2 = parse_residual_string(self.ET.name)

                # this is the discretized behavioral variable for tuning_var2 in time_period1
        assert tuning_var2 != 'escape', "Residual tuning cannot subtract activity explained by escape in periods outside homing/escape"
        self.ET.residual_var2_all_time = create_discretized_behave_var(self.aefizz,
                                                                        self.x, 
                                                                        self.y, 
                                                                        self.condition, 
                                                                        interpolation=True if tuning_var2 == "speed" else False,
                                                                        tuning_var=tuning_var2)
        print("the residual var2 all time vector has length " + str(len(self.ET.residual_var2_all_time)))
        
        if tuning_var2 == "2D_position":
            # in this case, run and/or load data from PlaceCells pipeline instead of ComputeTuning pipeline
            PC_dict = load_or_compute_2d_position_tuning(self.aefizz, time_period2)
            if isinstance(PC_dict["shelter_only"], dict):
                self.ET.residual_fr_var2_t2 = np.array([PC_dict[c]["rate_map"] for c in ["shelter_only", "barrier_pre_flip", "barrier_post_flip"]])
                logger.warning("Using full rate map for residual tuning in linear shift as well!")
                # self.ET.residual_fr_shift0_var2_t2 = np.array([PC_dict[c]["rate_map_null"] for c in ["shelter_only", "barrier_pre_flip", "barrier_post_flip"]])
                self.ET.residual_fr_shift0_var2_t2 = np.array([PC_dict[c]["rate_map"] for c in ["shelter_only", "barrier_pre_flip", "barrier_post_flip"]])
            elif isinstance(PC_dict["shelter_only"], object):
                self.ET.residual_fr_var2_t2 = np.array([PC_dict[c].item()["rate_map"] for c in ["shelter_only", "barrier_pre_flip", "barrier_post_flip"]])
                logger.warning("Using full rate map for residual tuning in linear shift as well!")
                self.ET.residual_fr_shift0_var2_t2 = np.array([PC_dict[c].item()["rate_map"] for c in ["shelter_only", "barrier_pre_flip", "barrier_post_flip"]])
            check_bin_match(self.ET.residual_var2_all_time, self.ET.residual_fr_var2_t2)
        else:
            # load tuning data for var2 in exploration from ComputeTuning object
            # this is the firing rate in the tuning curve to var2 in time_period2
            CT_var2_t2 = load_or_compute_escape_tuning(self.aefizz, tuning_var2 + " in " + time_period2)

            self.ET.residual_fr_var2_t2 = CT_var2_t2['fr_full']
            if 'shifts' in CT_var2_t2.keys():
                mid = np.where(CT_var2_t2['shifts'] == 0)[0][0]
            else:
                mid = int(np.shape(CT_var2_t2['y_fitted_shift'])[0] / 2)
            self.ET.residual_fr_shift0_var2_t2 = CT_var2_t2['fr_shift'][mid, :, :, :]

# -----------------------------Helper functions for loading or running computation ----------------------------

def check_bin_match(residual_var2_all_time, residual_fr_var2_t2):
    """Validate that visited 2D bins are within the loaded place-field map bounds.
    This is robust to partial spatial coverage (mouse may not visit all bins)."""
    if residual_var2_all_time.ndim != 2 or residual_var2_all_time.shape[1] != 2:
        raise ValueError(
            f"Expected residual_var2_all_time to have shape (time, 2), got {residual_var2_all_time.shape}"
        )

    xy = residual_var2_all_time
    valid = ~np.isnan(xy).any(axis=1)

    if np.any(valid):
        xy_int = xy[valid].astype(int)
        x_idx = xy_int[:, 0]
        y_idx = xy_int[:, 1]

        n_x_bins = residual_fr_var2_t2.shape[1]
        n_y_bins = residual_fr_var2_t2.shape[2]

        out_of_bounds = (x_idx < 0) | (x_idx >= n_x_bins) | (y_idx < 0) | (y_idx >= n_y_bins)
        if np.any(out_of_bounds):
            bad_pairs = np.unique(xy_int[out_of_bounds], axis=0)
            preview = bad_pairs[:10].tolist()
            raise AssertionError(
                "Visited 2D position bins exceed loaded place-field map bounds. "
                f"Map shape: ({n_x_bins}, {n_y_bins}); "
                f"x range visited: [{x_idx.min()}, {x_idx.max()}], "
                f"y range visited: [{y_idx.min()}, {y_idx.max()}]. "
                f"Example out-of-bounds bins (up to 10): {preview}"
            )
    else:
        logger.warning("No valid 2D bins found in residual_var2_all_time (all NaN).")

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
    logger.info(f"Checking for existing results to {variable} in EP tuning database...")
    
    _, do_analysis, hexaname = check_database_for_same_run(
        db_settings={"variable": variable, **settings_to_check(aefizz.settings, ["ep_", "linshift"])},
        results_csv_name=savepath + os.sep + "EscapePattern_results.csv",
        settings=aefizz.settings,
    )
    
    # check file exists
    if do_analysis == False:
        EP_dict = np.load(savepath + os.sep + "EPtuning_" + hexaname + "_results.npz", allow_pickle=True)
    else:
        logger.warning(f"Tuning to {variable} file not found, computing now...")
        check_aefizz_completeness(aefizz, attrlist = ["frame_by_cluster_matrix", "video_df", "cluster_Ids", "homings_object", "escape_object"])
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

def load_or_compute_2d_position_tuning(aefizz, time_period):
    savepath = make_directory(
        os.path.join(
            aefizz.session.base_path,
            aefizz.session.processed_path,
            "models",
            "place_cells",
        )
    )
    logger.info(f"Checking for existing place cell results in {time_period} in place cell database...")
    _, do_analysis, hexaname = check_database_for_same_run(db_settings = {'time_period': time_period, **settings_to_check(aefizz.settings, ["linshift", "place_cell"])}, 
                                    results_csv_name = savepath + os.sep + "place_cell_results.csv", 
                                    settings = aefizz.settings) 
    if do_analysis == False:
        PC_dict = np.load(savepath + os.sep + "PC_" + hexaname + "_results.npz", allow_pickle=True)
    else:
        logger.warning(f"PlaceCell info for {time_period} not found, computing now!")
        check_aefizz_completeness(aefizz, attrlist = ["video_and_spike_data", "Cluster_Ids"])
        PC = PlaceCells(aefizz = aefizz, time_period = time_period)
        PC.preprocess_data()
        PC.compute_place_fields_conditions()
        PC.plot_place_fields_conditions()
        PC_dict = PC.save(return_dict = True)

    return PC_dict


def check_aefizz_completeness(aefizz, attrlist):
    """This function checks that the aefizz object has all the necessary data and preprocessing to compute escape tuning curves.
    If not, it raises an error and specifies what is missing."""

    if (not hasattr(aefizz, "frame_by_cluster_matrix")) & ("frame_by_cluster_matrix" in attrlist):
        aefizz.frame_by_cluster_matrix = np.load(
            os.path.join(aefizz.session.base_path, aefizz.session.processed_path) + "\\" + "frame_by_" + aefizz.cluster_type + "_cluster_matrix.npy"
        )
    if (not hasattr(aefizz, "video_df")) & ("video_df" in attrlist):
        aefizz.video_df = pl.read_csv(os.path.join(aefizz.session.base_path, aefizz.session.processed_path) + "\\" + "full_video_dataframe.csv")

    if (not hasattr(aefizz, "cluster_Ids")) & ("cluster_Ids" in attrlist):
        aefizz.cluster_Ids = np.load(str(os.path.join(aefizz.session.base_path, aefizz.session.processed_path) + "/" + aefizz.cluster_type + "_cluster_Ids.npy"))

    if (not hasattr(aefizz, "homings_object")) & ("homings_object" in attrlist):
        homing_path = os.path.join(aefizz.session.base_path, aefizz.session.processed_path, "homings", "homings_obj.pkl")
        with open(homing_path, "rb") as f:
            aefizz.homings_object = pickle.load(f)

    if (not hasattr(aefizz, "escape_object")) & ("escape_object" in attrlist):
        escape_path = os.path.join(aefizz.session.base_path, aefizz.session.processed_path, "escapes", "escapes_obj.pkl")
        with open(escape_path, "rb") as f:
            aefizz.escape_object = pickle.load(f)

    if (not hasattr(aefizz, "video_and_spike_data")) & ("video_and_spike_data" in attrlist):
        video_and_spike_path = os.path.join(aefizz.session.base_path, aefizz.session.processed_path, "good_video_spike_count_df.parquet")
        aefizz.video_and_spike_data = pl.read_parquet(video_and_spike_path)
        aefizz.video_and_spike_data = aefizz.video_and_spike_data.select([x for x in COLUMNS_TO_KEEP if x in aefizz.video_and_spike_data.columns])
        if "speed" not in aefizz.video_and_spike_data.columns:
            if hasattr(pl.col("frames"), "apply"):
                video_df = aefizz.video_df.select(
                    [pl.col("frames").apply(float), pl.exclude("frames")]
                )  # Cast frames to float to permit join and remove old frames column with wrong type
            else:
                video_df = aefizz.video_df.select([aefizz.video_df["frames"].cast(pl.Float64), pl.exclude("frames")])
            # map speed at each frome to the video and spike data df so we can exclude low speed frames in the place cell analysis
            aefizz.video_and_spike_data = aefizz.video_and_spike_data.join(video_df.select(["frames", "speed"]), on='frames', how='left')


