"""A script for computing CCA"""

import os
from loguru import logger
import numpy as np
from dataclasses import asdict
from sklearn.cross_decomposition import CCA

from behave_analysis.analyze.CCA.cca_utils import compute_distance_object, safe_corrcoef, get_correlation_loadings, select_xval_frames
from behave_analysis.analyze.CCA.find_shelter_exit_and_runs import find_shelter_exit_runs, find_bout_runs
from behave_analysis.analyze.results_database_utils import check_database_for_same_run, add_run_to_database, settings_to_check, check_database_for_matched_results, generate_run_id
from behave_analysis.analyze.filtering_data.filtering_functions import filter_video_dataframe
from behave_analysis.utils.creating_directories import make_directory
from behave_analysis.analyze.EscapePattern.escape_pattern_utils import homing_escape_onsets, homing_escape_filtering_vector

class CCAmodel:
    def __init__(self, aefizz):
        self.aefizz = aefizz
        self.settings = aefizz.settings
        self.savepath = make_directory(os.path.join(self.aefizz.session.base_path, self.aefizz.session.processed_path, "models", "CCA"))
        self.database, self.do_analysis, self.hexaname = check_database_for_same_run(settings_to_check(self.settings, ["cca"]), 
                                                                                    self.savepath + os.sep + "CCA_results.csv", 
                                                                                    self.settings)  
        self.n_components = min(len(self.settings.cca_behavioral_vars), self.settings.cca_n_components) # number of CCA components to keep, can't be more than number of behavioral variables
    
    def preprocess_neural_data(self):
        """This function z-scores the neural data"""
        fcm = self.aefizz.frame_by_cluster_matrix
        mean_fr = np.nanmean(fcm, axis=0)
        std_fr = np.nanstd(fcm, axis=0)
        self.fcm_z = (fcm - mean_fr) / std_fr
    
    def preprocess_session_vars(self):
        self.session_start = self.aefizz.session.valid_time[0]*60*40

    def preprocess_behavioral_data(self):
        """Build behavioral matrix (time x variables) using the variables listed in settings.
        Angles are given as sine and cosine to avoid issues with circular variables. 
        Speed is log-transformed to reduce skew."""
        self.Y = np.empty((self.aefizz.video_df.shape[0], 1))
        dt = 1 / self.aefizz.session.video.fps
        self.name = np.empty((1,))
        for col in self.settings.cca_behavioral_vars:
            if "distance" in col:
                dist = compute_distance_object(self.aefizz.video_df, col, self.aefizz.session)
                self.Y = np.append(self.Y, np.array(dist)[:, np.newaxis], axis=1)
                self.name = np.append(self.name, col)
            elif col in ["mouse_x_position", "mouse_y_position", "distance_to_shelter", "distance_to_barrier1", "distance_to_barrier2"]:
                self.Y = np.append(self.Y, self.aefizz.video_df[col].to_numpy()[:, np.newaxis], axis=1)
                self.name = np.append(self.name, col)
            elif col == "acceleration":
                # acc = savgol_filter(self.aefizz.video_df["speed"].to_numpy(), window_length=15, polyorder=2, deriv=1, delta=dt)
                acc = np.gradient(self.aefizz.video_df["speed"].to_numpy(), dt)
                self.Y = np.append(self.Y, acc[:, np.newaxis], axis=1)
                self.name = np.append(self.name, col)
            elif col == "hdir_velocity":
                # hv = savgol_filter(np.unwrap(self.aefizz.video_df["hdir"].to_numpy()), window_length=5, polyorder=2, deriv=1, delta=dt)
                hv = np.gradient(np.unwrap(self.aefizz.video_df["hdir"].to_numpy()), dt)
                self.Y = np.append(self.Y, hv[:, np.newaxis], axis=1)
                self.name = np.append(self.name, col)
            elif col == "speed":
                self.Y = np.append(self.Y, np.log1p(self.aefizz.video_df[col].to_numpy())[:, np.newaxis], axis=1)
                self.name = np.append(self.name, col)
            elif col in ["hdir", "hsa", "h_preflipbar_a", "h_postflipbar_a"]:
                self.Y = np.append(self.Y, np.cos(np.deg2rad(self.aefizz.video_df[col].to_numpy()))[:, np.newaxis], axis=1)
                self.Y = np.append(self.Y, np.sin(np.deg2rad(self.aefizz.video_df[col].to_numpy()))[:, np.newaxis], axis=1)
                self.name = np.append(self.name, col + "_cos")
                self.name = np.append(self.name, col + "_sin")
            else:
                raise ValueError(f"Variable {col} not recognized as a valid behavioral variable")
        self.Y = self.Y[:, 1:]  # remove the initial empty column
        self.name = self.name[1:]  # remove the initial empty column

    def run_cca(self,train_data, test_data, condition):
        """Assuming X_train is (time, neurons) and Y_train is (time, behaviors)
        INPUTS:
        - train_data: a dictionary with keys "X" and "Y" for the training data
        - test_data: a dictionary with keys for each test set, each containing a dictionary with keys "X" and "Y" for the test data
        - condition: the condition index for which CCA is being run (used for storing results)
        """
        # 1. Initialize CCA (n_components is the number of correlated pairs to find)
        cca = CCA(n_components=self.n_components)

        # 2. Fit the model (find the loading vectors)
        cca.fit(train_data["X"], train_data["Y"])

        # 3. Project the TRAINING data onto the loading vectors
        # X_c and Y_c are the "Canonical Scores"
        X_train_c, Y_train_c = cca.transform(train_data["X"], train_data["Y"])
        # get the canonical correlation
        for i in range(self.n_components):
            self.results["train_canonical_corr"][condition,i] = safe_corrcoef(X_train_c[:,i], Y_train_c[:,i])
        # get the loadings for the neurons
        self.results["train_loadings"][condition,:,:] = cca.x_loadings_

        # 4. Project the TEST data (using the same weights found during training)
        for k, key in enumerate(test_data.keys()):
            if len(test_data[key]["X"]) == 0:
                continue
            X_test, Y_test = test_data[key]["X"], test_data[key]["Y"]
            X_test_c, Y_test_c = cca.transform(X_test, Y_test)
            # get the loadings for the neurons
            self.results["test_loadings"][k,condition,:,:] = get_correlation_loadings(X_test, X_test_c)
            # get the canonical correlation
            for i in range(self.n_components):
                self.results["test_canonical_corr"][k,condition,i] = safe_corrcoef(X_test_c[:,i], Y_test_c[:,i])

    def set_up_results_dict(self):
        self.results = {"n_components": self.n_components,
                        "test_sets": [],
                        "behavioral_vars": self.name,
                        "train_loadings": np.full((len(self.aefizz.all_conditions), self.aefizz.frame_by_cluster_matrix.shape[1], self.n_components), np.nan),
                        "test_loadings": np.full((len(self.settings.cca_test_sets)+1, len(self.aefizz.all_conditions), self.aefizz.frame_by_cluster_matrix.shape[1], self.n_components), np.nan),
                        "train_canonical_corr": np.full((len(self.aefizz.all_conditions), self.n_components), np.nan),
                        "test_canonical_corr": np.full((len(self.settings.cca_test_sets)+1, len(self.aefizz.all_conditions), self.n_components), np.nan)}

    def get_train_test_indices(self, condition):
        """Get indices of frames (0-indexed!) to use for training and testing CCA.
        It will create a xval test set from the training set based on the method specified in settings.
        RETuRNS: train_idx, test_idx_dict where test_idx_dict has keys for each test set specified in settings and values are the corresponding indices"""
        test_idx_dict = {}
        for test in self.settings.cca_test_sets:
            if "xval" in test:
                continue
            elif "homing&escape" in test:
                # test_idx_dict[test] = filter_video_dataframe(self.aefizz.video_df, condition=condition, outofshelter=True, exclude_escape=False, select_homings=True, select_escape=True)["frames"].to_numpy() - 1
                onset_dict = homing_escape_onsets(self.aefizz, test)
                h_e_vec, _ = homing_escape_filtering_vector(
                    nframes=len(self.aefizz.video_df),
                    onset_dict=onset_dict,
                    xpos=self.aefizz.video_df["mouse_x_position"].to_numpy(),
                    ypos=self.aefizz.video_df["mouse_y_position"].to_numpy(),
                    shelter_location=self.aefizz.session.shelter_location,
                )
                test_idx_dict[test] = self.aefizz.video_df["frames"].to_numpy()[h_e_vec] - 1
                self.results["test_sets"].append(test)
            elif test == "shelter_outing":
                condition_df = filter_video_dataframe(self.aefizz.video_df, condition=condition, outofshelter=None, exclude_escape=True, exclude_homings=True)
                outside_runs = find_shelter_exit_runs(condition_df, min_distance_cm = 20.0)
                test_idx_dict[test] = condition_df["frames"].to_numpy()[outside_runs] - 1
                self.results["test_sets"].append(test)
            elif test == "bout_runs":
                condition_df = filter_video_dataframe(self.aefizz.video_df, condition=condition, outofshelter=None, exclude_escape=True, exclude_homings=True)
                bout_runs = find_bout_runs(condition_df, min_distance_cm=40, remove_shelter_outings=True)
                test_idx_dict[test] = condition_df["frames"].to_numpy()[bout_runs] - 1
                self.results["test_sets"].append(test)
            elif test == "explore": # this needs to be the last condition because it uses the remaining indices that aren't in the other test sets
                exp_idx = filter_video_dataframe(self.aefizz.video_df, condition=condition, outofshelter=True, exclude_escape=True, exclude_homings=True)["frames"].to_numpy() - 1
                # confirm that explore test set doesn't include indices from any other test sets
                for other_test in test_idx_dict.keys():
                    exp_idx = np.array([idx for idx in exp_idx if idx not in test_idx_dict[other_test]])
                test_idx_dict[test] = exp_idx
                self.results["test_sets"].append(test)
            else:
                raise ValueError(f"Test set {test} not recognized as a valid test set")
        
        if self.settings.cca_train_set in self.settings.cca_test_sets:
            raise ValueError("Train set cannot be the same as any of the test sets")
        
        if self.settings.cca_train_set == "explore":
            train_idx = filter_video_dataframe(self.aefizz.video_df, condition=condition, outofshelter=True, exclude_escape=True, exclude_homings=True)["frames"].to_numpy() - 1
            # make sure no train indices are in the test datasets
            for test in test_idx_dict.keys():
                train_idx = np.array([idx for idx in train_idx if idx not in test_idx_dict[test]])
        elif "homing&escape" in self.settings.cca_train_set:
            onset_dict = homing_escape_onsets(self.aefizz, self.settings.cca_train_set)
            h_e_vec, _ = homing_escape_filtering_vector(
                nframes=len(self.aefizz.video_df),
                onset_dict=onset_dict,
                xpos=self.aefizz.video_df["mouse_x_position"].to_numpy(),
                ypos=self.aefizz.video_df["mouse_y_position"].to_numpy(),
                shelter_location=self.aefizz.session.shelter_location,
            )
            train_idx = self.aefizz.video_df["frames"].to_numpy()[h_e_vec] - 1
        elif self.settings.cca_train_set == "shelter_outing":
            condition_df = filter_video_dataframe(self.aefizz.video_df, condition=condition, outofshelter=None, exclude_escape=True, exclude_homings=True)
            outside_runs = find_shelter_exit_runs(condition_df, min_distance_cm = 20.0)
            train_idx = condition_df["frames"].to_numpy()[outside_runs] - 1
        elif self.settings.cca_train_set == "bout_runs":
            condition_df = filter_video_dataframe(self.aefizz.video_df, condition=condition, outofshelter=None, exclude_escape=True, exclude_homings=True)
            bout_runs = find_bout_runs(condition_df, min_distance_cm=40, remove_shelter_outings=True)
            train_idx = condition_df["frames"].to_numpy()[bout_runs] - 1
        else:
            raise ValueError(f"Train set {self.settings.cca_train_set} not recognized as a valid train set")
        
        # if explore in test, make sure there is no overlap between train and explore
        if "explore" in test_idx_dict.keys():
            test_idx_dict["explore"] = np.array([idx for idx in test_idx_dict["explore"] if idx not in train_idx])

        # check that there is no overlap between train and test indices
        for test in test_idx_dict.keys():
            overlap = np.intersect1d(train_idx, test_idx_dict[test])
            if len(overlap) > 0:
                raise ValueError(f"Overlap between train and test indices for test set {test}: {overlap}")
        
        # split the train indices into train and validation sets using method specified in settings
        if (self.settings.cca_train_set == "homing&escape") | (self.settings.cca_train_set == "shelter_outing"):
            assert (self.settings.cca_xval_method == "random_split") | (self.settings.cca_xval_method == "half"), "Currently only random split and half methods are implemented for homing&escape and shelter_outing train set xval"
        comparison_idx = []
        if "match" in self.settings.cca_xval_method:
            if "homings" in self.settings.cca_xval_method:
                test_name = [name for name in test_idx_dict.keys() if "homing&escape" in name][0]
                comparison_idx = test_idx_dict[test_name]
            elif "shelter_outing" in self.settings.cca_xval_method:
                comparison_idx = test_idx_dict["shelter_outing"]
        train_idx, xval_idx = select_xval_frames(self.aefizz.video_df, train_idx, self.settings.cca_xval_method, comparison_indices=comparison_idx)
        test_idx_dict[self.settings.cca_train_set + "_xval"] = xval_idx
        self.results["test_sets"].append(self.settings.cca_train_set + "_xval")

        return train_idx, test_idx_dict

    def cca_across_conditions(self):
        # preprocess data
        self.preprocess_session_vars()
        self.preprocess_neural_data()
        self.preprocess_behavioral_data()

        # set up data structures to hold results
        self.set_up_results_dict()

        # run CCA for each condition
        logger.info(f"Running CCA with the following settings: n_components={self.n_components}, train set={self.settings.cca_train_set}, xval method={self.settings.cca_xval_method}, test sets={self.settings.cca_test_sets}")
        for c, cond in enumerate(self.aefizz.all_conditions):
            train_idx, test_idx_dict = self.get_train_test_indices(condition=cond)
            if len(train_idx) == 0:
                logger.warning(f"No training data for condition {cond}, skipping CCA for this condition")
                continue
            train_data = {"X": self.fcm_z[train_idx, :], "Y": self.Y[train_idx, :]}
            test_data = {}
            for key in self.results["test_sets"]:
                test_data[key] = {"X": self.fcm_z[test_idx_dict[key], :], "Y": self.Y[test_idx_dict[key], :]}
            self.run_cca(train_data, test_data, condition=c)
        
        self.save()

    def save(self, return_dict = False):
        """This function saves the results of the CCA analysis to a file."""
        logger.info("Saving CCA results to file and database")
        filename = os.path.join(self.savepath, "CCA_" + self.hexaname)
        np.savez(os.path.join(filename + "_results.npz"), 
                             **self.results,
                             allow_pickle=True)
        settings=asdict(self.settings)
        np.savez(filename + "_settings.npz", **settings, allow_pickle=True)
        # add results to database
        add_run_to_database(self.database, 
                            settings_to_check(self.settings, ["cca"]),  
                            self.savepath + os.sep + "CCA_results.csv", 
                            self.hexaname)
        if return_dict:
            return self.results_dict