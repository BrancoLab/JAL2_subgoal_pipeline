from pathlib import Path

import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import dill as pickle
import polars as pl
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from scipy import stats
import statsmodels.api as sm
from sklearn.metrics import r2_score
from scipy.stats import ttest_rel
from tqdm import tqdm

from behave_analysis.utils.creating_directories import make_directory


class UnitTests:
    """Unit tests for the SingleTrialRegression class"""

    @staticmethod
    def check_attributes_of_homing_dic(homings_obj):
        """Checking the attributes of the homing dictionary to make sure it is in the correct format"""
        try:
            onset_frames = homings_obj.onset_frames
            offset_frames = homings_obj.offset_frames
        except AttributeError:
            raise AttributeError("The homings object does not have the required attributes - Something upstream is wrong with the homings object")
        assert len(onset_frames) == len(
            offset_frames
        ), "The onset and offset frames are not the same length - Something is wrong with the homings object"

    @staticmethod
    def check_frame_indexes_are_incremental(arr: np.ndarray):
        """A test to check that frames increment by 1 and as such is continuous ensuring no frames are skipped

        Args:
            arr (np.ndarray): The array of frames to check"""
        # Check each element to see if it increments by 1
        for i in range(len(arr) - 1):
            if arr[i + 1] - arr[i] != 1:
                return False
        return True

    @staticmethod
    def check_index_is_valid(compute_index_func):
        """Check if the index is valid"""
        test_hsa = np.array([0, np.pi, np.pi, np.pi / 12, (5 * np.pi) / 6, (11 * np.pi) / 12, np.pi / 12, (23 * np.pi) / 12])
        test_angle = np.array([np.pi, 0, np.pi, (5 * np.pi) / 6, np.pi / 12, (11 * np.pi) / 12, (13 * np.pi) / 12, (7 * np.pi) / 6])

        test_result = compute_index_func(test_hsa, test_angle)

        assert test_result[0] == -1, "Test 1 failed, if mouse face shelter then expected -1 but got {}".format(test_result[0])
        assert test_result[1] == 1, "Test 2 failed, if mouse face the test goal expected 1 but got {}".format(test_result[1])
        assert test_result[2] == 0, "Test 3 failed, expected 0 but got {}".format(test_result[2])
        assert (
            np.around(test_result[3], 1) == -0.8
        ), "Test 4 failed, expected -0.8 but got {}. Should be negative as mouse facing closer to shelter".format(test_result[3])
        assert (
            np.around(test_result[4], 1) == 0.8
        ), "Test 4 failed, expected 0.8 but got {}. Should be positive as mouse facing closer to goal".format(test_result[3])
        assert test_result[5] == 0, "Test 5 failed, expected 0 as angles are the same but got {}".format(test_result[5])
        assert (
            np.around(test_result[6], 1) == -0.8
        ), "Test 6 failed, expected -0.8 but got {}. Answer should be closer to -0.9 as mouse is facing towards shelter ".format(test_result[6])
        assert (
            np.around(test_result[7], 1) == -0.8
        ), "Test 7 failed, expected -0.8 but got {}. Answer should be closer to -0.8 as mouse is facing towards shelter ".format(test_result[7])

        logger.success("All tests passed for the compute_predictor function")


class PreprocessSingleTrialRegression:
    """A class to preprocess the data for single trial regression analysis"""

    def __init__(
        self, video_df, homings_obj, video_and_spike_data, frame_by_cluster_matrix, save_path, similar_homings=False, orthogonalise_index=False
    ):
        self.homings_obj = homings_obj
        self.video_and_spike_data = video_and_spike_data
        self.frame_by_cluster_matrix = frame_by_cluster_matrix
        self.save_path = save_path
        self.video_df = self.remove_columns_from_video_df(video_df)
        self.should_we_orthogonalise_index = orthogonalise_index

        # Settings
        self.similar_homings = similar_homings

        # Unit tests
        UnitTests.check_attributes_of_homing_dic(self.homings_obj)
        UnitTests.check_index_is_valid(self.compute_index)

        # Preprocessing homing data
        self.homing_data_single_dataframe = self.preprocess_homing_data(select_similar_homings=self.similar_homings)

        # Add the dependent variable to the data
        self.homing_data_single_dataframe = self.add_dependent_index_variable_to_data(orthogonalise_index=self.should_we_orthogonalise_index)

        # Create the design matrix
        self.design_matrix, self.targets_df = self.create_the_design_matrix_and_targets(
            self.homing_data_single_dataframe, self.frame_by_cluster_matrix
        )

        # Descriptive plots
        self.plot_homing_durations()
        self.plot_y_coords_distribution()
        self.plot_the_index_distribution()
        self.plot_the_index_per_homing()
        self.plot_the_distribution_of_the_dependent_variables()

    # ------- Descriptive exploratory plots based on preprocessing -------------------------

    def plot_homing_durations(self) -> None:
        """Plotting and saving the homing durations"""
        durations = (self.homings_obj.offset_frames - self.homings_obj.onset_frames) / 40  # Hard coded 40 Hz for fps
        plt.hist(durations)
        plt.xlabel("Duration (s)")
        plt.ylabel("Number of homings")
        plt.title("Homing durations")
        plt.savefig(self.save_path / "homing_durations.png")
        plt.close()

    def plot_y_coords_distribution(self):
        """Plotting and saving the y axis bins to see the distribution of the homings

        Not used for anything, just for exploratory purposes showing non-uniform distribution of y coordinates"""
        ycoords = self.video_df["mouse_y_position"]
        ycoords = ycoords.filter(ycoords < 800)
        plt.title("Distribution of y coordinates in bins")
        bins = np.linspace(0, 800, 32)  # Remove near shelter as there are a lot of frames there
        plt.hist(ycoords, bins=bins, color="green")  # These are the defined bins
        plt.xlabel("Y coordinate")
        plt.ylabel("Number of frames")
        plt.savefig(self.save_path / "y_coordinates_distribution.png")
        plt.close()

    def plot_the_index_distribution(self):
        """Plotting and save the index distribution"""
        index = self.homing_data_single_dataframe["index"].to_numpy()
        plt.hist(index, bins=20)
        plt.xlabel("Index")
        plt.ylabel("Number of frames")
        plt.title("Distribution of the index")
        plt.savefig(self.save_path / "index_distribution.png")
        plt.close()

    def plot_the_index_per_homing(self):
        """Plotting and save the index per homing"""

        # Get the unique homing ids
        homing_ids = np.unique(self.homing_data_single_dataframe["homing_id"].to_numpy())

        # Plot params
        length = 20
        save_path = self.save_path / "index_per_homing"
        make_directory(save_path)

        # Loop through each homing id and plot the index
        for homing_id in homing_ids:
            homing = self.homing_data_single_dataframe.filter(self.homing_data_single_dataframe["homing_id"] == homing_id)

            # Get the x and y positions of the head direction
            dx = length * np.cos(homing["hdir"])
            dy = length * -np.sin(homing["hdir"])
            xs = homing["mouse_x_position"]
            ys = homing["mouse_y_position"]

            plt.quiver(xs, ys, dx, dy, angles="xy", scale_units="xy", scale=2, color="blue")
            plt.title(f"Index for homing {homing_id}. Arrow is hdir, text is index")

            # convert to pandas for the text
            homing_pd = homing.to_pandas()

            # Add the index every 3rd frame to reduce clutter
            for i, row in homing_pd.iloc[::3].iterrows():
                plt.text(x=row["mouse_x_position"] + 20, y=row["mouse_y_position"] + 10, s=str(np.around(row["index"], 1)), color="red", fontsize=6)

            plt.xlim(100, 900)
            plt.ylim(100, 900)
            plt.grid(True)
            plt.xlabel("X Coordinate")
            plt.ylabel("Y Coordinate")
            plt.savefig(save_path / f"index_for_homing_{homing_id}.png")
            plt.close()

    def create_descriptive_plots(self):
        """Creating the descriptive plots"""
        raise NotImplementedError

    def plot_the_distribution_of_the_dependent_variables(self):
        """In order to check the scale of the dependent variables, we plot the distribution of the dependent variables.
        If the dependent variables are not normally distributed then we may need to transform them to make them normally distributed
        or at least on the same scale. This is important for the regression analysis as the dependent variables should be on the same scale"""

        for dependent_variable in self.here_are_all_the_columns:
            plt.hist(self.targets_df[dependent_variable].to_numpy())
            plt.xlabel(dependent_variable)
            plt.ylabel("Number of frames")
            plt.title(f"Distribution of {dependent_variable}")
            plt.savefig(self.save_path / f"{dependent_variable}_distribution.png")
            plt.close()

    # ----------- Preprocessing -------------------------------------

    def remove_columns_from_video_df(self, video_df) -> pl.DataFrame:
        """Removing uncessary columns for memory purposes"""
        keep = [
            "frames",
            "mouse_x_position",
            "mouse_y_position",
            "OutofshelterIdx",
            "EscapePeriod",
            "shelter",
            "hdir",
            "barrier_present",
            "barrier_flipped",
            "hsa",
            "h_bar_north_a",
            "h_bar_south_a",
        ]
        return video_df.select(keep)

    def extract_data_from_homings(self) -> dict:
        """Extracting the data from the homings dictionary"""
        onset_frames = self.homings_obj.onset_frames
        offset_frames = self.homings_obj.offset_frames
        homing_info = []
        for onset, offset in zip(onset_frames, offset_frames):
            homing = self.video_df[onset:offset]
            homing = homing.select(
                [
                    "frames",
                    "mouse_x_position",
                    "mouse_y_position",
                    "hdir",
                    "hsa",
                    "h_bar_north_a",
                    "h_bar_south_a",
                ]
            )
            homing_info.append(homing)

        # Check the frame column of each homing information increments uniformly by 1 such that no frames are missed
        for homing in homing_info:
            assert UnitTests.check_frame_indexes_are_incremental(homing["frames"].to_numpy()), "Frames are missing in the homing information"

        return homing_info

    def select_similar_homings(self, extracted_homing_info) -> dict:
        """

        -- Similar time periods
        -- Similar mouse positions
        -- Similar targets

        Raises:
            NotImplementedError: _description_

        TODO - Refactor this to choose subgoals based escapes
        """
        # HARDCORE MODE - we like it rough and tough
        xcoordinate_min = 300
        xcoordinate_max = 700
        ycoordinate_min = 200
        ycoordinate_end_min = 750
        x_middle_chunk_min = 400
        x_middle_chunk_max = 600
        extracted_info = []

        for _, homing in enumerate(extracted_homing_info):

            # check if mouse x position is within the range - STARTS FARTS ONLY
            start_x = homing["mouse_x_position"][0]
            start_y = homing["mouse_y_position"][0]

            # Starts in a similar space
            if start_x > xcoordinate_min and start_x < xcoordinate_max and start_y < ycoordinate_min:

                # Ends in a similar space
                if homing["mouse_y_position"][-1] > ycoordinate_end_min:

                    # middle of frames is in a similar space
                    middle_x = homing["mouse_x_position"][int(len(homing) / 2)]
                    if middle_x < x_middle_chunk_min or middle_x > x_middle_chunk_max:
                        # plt.scatter(homing["mouse_x_position"], homing["mouse_y_position"])
                        # plt.hlines(y=512, xmin=150, xmax=900, color="k")

                        # CHOOSE ONE SIDE
                        if middle_x > 700:
                            plt.scatter(homing["mouse_x_position"], homing["mouse_y_position"])
                            plt.hlines(y=512, xmin=150, xmax=900, color="k")
                            plt.vlines(x=700, ymin=50, ymax=900, color="k")

                            extracted_info.append(homing)

            # Print all the homings
            # plt.scatter(homing["mouse_x_position"], homing["mouse_y_position"])

        return extracted_info

    def add_homing_id_to_homing_data(self, extracted_homing_info):
        """Adding the homing id to the homing data. This is needed for the group cross validation object"""

        for idx, homing in enumerate(extracted_homing_info):
            updated_homing = homing.with_columns(pl.lit(idx).alias("homing_id"))
            extracted_homing_info[idx] = updated_homing

        return extracted_homing_info

    def concatenate_the_homing_data(self, homing_info) -> pl.DataFrame:
        """Concatenating the homing data"""
        for idx, homing in enumerate(homing_info):
            if idx == 0:
                homing_data = homing
            else:
                homing_data = homing_data.vstack(homing)
        return homing_data

    def preprocess_homing_data(self, select_similar_homings) -> pl.DataFrame:
        """Preprocessing the data into a single dataframe for regression analysis"""
        extracted_homing_info = self.extract_data_from_homings()
        if select_similar_homings:
            self.homing_info = self.select_similar_homings(extracted_homing_info)
            extracted_homing_info = self.homing_info
        homing_info = self.add_homing_id_to_homing_data(extracted_homing_info)
        self.homing_data_single_dataframe = self.concatenate_the_homing_data(homing_info)
        return self.homing_data_single_dataframe

    def circular_difference(self, angle1: np.ndarray, angle2: np.ndarray) -> np.ndarray:
        """Calculates the shortest difference between two angles in radians from the origin"""

        # If the angle is greater than pi then subtract 2pi to get the smallest difference from the origin
        angle1 = np.where(angle1 > np.pi, (2 * np.pi) - angle1, angle1)
        angle2 = np.where(angle2 > np.pi, (2 * np.pi) - angle2, angle2)
        diff = np.arctan2(np.sin(angle1 - angle2), np.cos(angle1 - angle2))

        return diff

    def circular_sum(self, angle1, angle2):
        """Takes in radian values between 0 and 2pi as scalars or vectors and returns the circular sum."""
        # if the angle is greater than pi then subtract 2pi to get the smallest difference
        angle1 = np.where(angle1 > np.pi, (2 * np.pi) - angle1, angle1)
        angle2 = np.where(angle2 > np.pi, (2 * np.pi) - angle2, angle2)

        sum_angle = np.arctan2(np.sin(angle1 + angle2), np.cos(angle1 + angle2))
        # Adjust the result to be between 0 and 2pi because the arctan2 function returns values between -pi and pi
        asjusted_result = np.where(sum_angle < 0, sum_angle + 2 * np.pi, sum_angle)
        return asjusted_result

    def compute_index(self, angle1: np.ndarray, angle2: np.ndarray) -> pl.DataFrame:
        """Compute a normalised index between -1 and 1 between two angles:

        Args:
            angle1: The first angle in radians
            angle2: The second angle in radians

        Metric is computed as:
        -1: The angle to angle1 is close to zero and the angle2 is close to pi
            0: The angle to angle1 is close to the angle to angle2
            1: The angle to angle1 is close to pi and the angle to angle2 is close to zero
        """
        numerator = self.circular_difference(angle1, angle2)
        denominator = self.circular_sum(angle1, angle2)
        return numerator / denominator

    def add_dependent_index_variable_to_data(self, orthogonalise_index: bool = False):
        """Adding the dependent variable to the data. Where current -1 is facing shelter, 1 is facing goal and 0 is facing neither"""

        hsa = self.homing_data_single_dataframe["hsa"].to_numpy().copy()
        goal = self.homing_data_single_dataframe["h_bar_south_a"].to_numpy().copy()

        # Orthogonalise the angles to see if the correlations between the angles alone can explain the index
        if orthogonalise_index:

            # rotate the goal and hsa by pi/2 to make the angles orthogonal
            goal = goal + np.pi / 2
            hsa = hsa + np.pi / 2

            # Now make sure the angles are between -pi and pi
            hsa = np.where(hsa > np.pi, hsa - 2 * np.pi, hsa)
            goal = np.where(goal > np.pi, goal - 2 * np.pi, goal)

        # Quality check
        # Check arrays are not above or below -pi and pi
        assert np.all(hsa >= -np.pi) and np.all(hsa <= np.pi), "hsa values are not within the range of -pi and pi"
        assert np.all(goal >= -np.pi) and np.all(goal <= np.pi), "h_bar_north_a values are not within the range of -pi and pi"

        # if values negative radians then add 2pi to make them positive
        # turn negative radians into positive radians to make them easier to work with
        hsa = np.where(hsa < 0, hsa + 2 * np.pi, hsa)
        goal = np.where(goal < 0, goal + 2 * np.pi, goal)

        index = self.compute_index(hsa, goal)
        assert np.all(index >= -1) and np.all(index <= 1), "Predictor values are not within the range of -1 and 1"

        result = self.homing_data_single_dataframe.with_columns(pl.Series("index", index))
        return result

    def create_the_design_matrix_and_targets(self, data: pl.DataFrame, frame_by_cluster_matrix: np.ndarray) -> np.ndarray:
        """Creating the design matrix"""

        # Initialising the design matrix
        total_frames = len(data)
        total_features = frame_by_cluster_matrix.shape[1]
        design_matrix = np.zeros((total_frames, total_features))

        counter = 0
        for idx in range(len(np.unique(data["homing_id"]))):

            # Get the frames for the homing id for slicing
            frames = data.filter(data["homing_id"] == idx)["frames"].to_numpy()

            # Get the corresponding frame by cluster matrix
            spike_data = frame_by_cluster_matrix[frames[0] : frames[-1] + 1]

            # Add the spike data to the design matrix
            design_matrix[counter : counter + len(spike_data)] = spike_data
            counter += len(spike_data)

        # Normalise the design matrix using a simple scale by the standard deviation
        np_std = np.std(design_matrix, axis=0)
        design_matrix = np.divide(design_matrix, np_std)
        design_matrix = pd.DataFrame(design_matrix)
        design_matrix["homing_id"] = data["homing_id"].to_numpy()

        # Add all columns as a dependent variable
        all_columns_names = data.columns

        dependent_arrays = []

        # Add a bunch of dependent variables to the design matrix
        for idx, column_name in enumerate(all_columns_names):
            dependent_arrays.append(data[column_name].to_numpy())

        # Convert the list into pandas
        dependent_arrays = np.array(dependent_arrays).T
        dependent_df = pd.DataFrame(dependent_arrays, columns=all_columns_names)

        self.here_are_all_the_columns = all_columns_names

        return design_matrix, dependent_df


class SingleTrialRegression:
    """A class that performs single trial regression analysis on the data"""

    def __init__(self, design_matrix: pd.DataFrame, save_path: Path, all_dependent_names: list, targets_df: pd.DataFrame):
        self.design_matrix = design_matrix
        self.save_path = save_path
        self.dependent_names = all_dependent_names
        self.targets_df = targets_df
        self.number_of_neurons = self.design_matrix.shape[1] - 1  # Subtract 1 for the homing id column

        self.run(run_all_dependent_variables=True, shift_neural_data=False, explore_coeffs_with_other_predictors=True)

    # --------------------- Functions for running the regression ---------------------

    def run(self, run_all_dependent_variables: bool, shift_neural_data: bool, explore_coeffs_with_other_predictors: bool) -> None:
        """This function runs the regression with different modes depending on the analysis you want to conduct:

        Modes:
        - run_all_dependent_variables: Run the model for all dependent variables
        - shift_neural_data: Shift the neural data and run the model to see how the R2 score changes with chance
        - explore_coeffs_with_other_predictors: Explore the coefficients with other predictors to see how they change"""

        if run_all_dependent_variables:
            self.r2_score_dic, _, _ = self.run_all_dependent_variables()
            self.plot_the_r2_scores_for_all_dependents()

        if shift_neural_data:
            shifts, og_r2, shifted_r2_ols = self.run_all_shifts()
            self.plot_shift_results(shifts, og_r2, shifted_r2_ols)

        if explore_coeffs_with_other_predictors:

            # Where og is the original gangster and comp is the comparison model
            og_r2_score, og_coefficients, og_p_values = self.run_just_one_dependent_variable("index", self.design_matrix)
            comparison_design_matrix = self.add_other_predictors_to_design_matrix(self.design_matrix)
            comp_r2_score, comp_predictors_coeffs, comp_predictors_p_value = self.run_just_one_dependent_variable("index", comparison_design_matrix)

            # Select only the neural data coefficients and p values (Excluding intercept and non neural coeffs)
            og_coefficients = og_coefficients[: self.number_of_neurons]
            og_p_values = og_p_values[: self.number_of_neurons]
            comp_predictors_coeffs = comp_predictors_coeffs[: self.number_of_neurons]
            comp_predictors_p_value = comp_predictors_p_value[: self.number_of_neurons]
            
            sig_index_coefficients_indices = np.where(og_p_values < 0.05) # Which coefficients are significant in the original model
            self.plot_proportion_of_coeffs_that_remain_significant(og_p_values, comp_predictors_p_value, og_r2_score, comp_r2_score)
            # Check whether the significant coefficients change significantly between the two models
            x1 = np.ones(len(og_coefficients))
            x2 = 2 * np.ones(len(comp_predictors_coeffs))
            self.plot_coefficients_between_models(x1, x2, og_coefficients, comp_predictors_coeffs, sig_index_coefficients_indices)

        return None

    def run_just_one_dependent_variable(self, dependent_var_name: str, design_matrix: pd.DataFrame) -> tuple:
        """Run the model for just one dependent variable"""

        # Create a save location for the dependent variable
        make_directory(self.save_path / "ols_regression" / dependent_var_name)

        ols_fold_results = self.run_ols_model_with_cross_val(
            design_matrix=design_matrix, dependent_variable=self.targets_df[dependent_var_name].to_numpy(), dependent_var_name=dependent_var_name
        )
        r2_scores, coefficients, p_values = self.unpack_fold_results_and_average(ols_fold_results)
        logger.info("Mean test R2 score for {}: {}", dependent_var_name, r2_scores)

        return r2_scores, coefficients, p_values

    def run_all_dependent_variables(self):
        """Run the model for different dependent variables and store the mean R2 scores
        to see which dependent variable is best predicted by the neural data alone"""

        # Init some storage vars
        list_of_dependent_vars = []
        r2_scores = {}
        p_values = {}
        coefficients = {}

        # Create a directory to store the results for each dependent variable
        for dependent_var_name in self.dependent_names:
            list_of_dependent_vars.append(self.targets_df[dependent_var_name].to_numpy())
            make_directory(self.save_path / "ols_regression" / dependent_var_name)

        # Run the model for each dependent variable
        for var_idx, dependent_var in enumerate(list_of_dependent_vars):
            logger.info("Running the model for dependent variable: {}", self.dependent_names[var_idx])
            ols_fold_results = self.run_ols_model_with_cross_val(
                design_matrix=self.design_matrix, dependent_variable=dependent_var, dependent_var_name=self.dependent_names[var_idx]
            )

            # Unpack the results and average them
            r2_scores[self.dependent_names[var_idx]], p_values[self.dependent_names[var_idx]], coefficients[self.dependent_names[var_idx]] = (
                self.unpack_fold_results_and_average(ols_fold_results)
            )

        return r2_scores, p_values, coefficients

    # ------------------------ Changing the design matrix ------------------
    
    def add_other_predictors_to_design_matrix(self, design_matrix: pd.DataFrame) -> pd.DataFrame:
        """Add other predictors to the design matrix to see how they affect the coefficients and p values compared to the neural data alone
        Currently hard coded to add the following predictors:
            - h_bar_north_a
            - hdir
            - hsa
            - h_bar_south_a
            - mouse_y_position"""
        design_matrix = design_matrix.copy()
        dependents_to_add = ["h_bar_north_a", "hdir", "hsa", "h_bar_south_a", "mouse_y_position"]
        for dependent in dependents_to_add:
            design_matrix[dependent] = self.targets_df[dependent].to_numpy()

        return design_matrix

    # --------------------- Cross validation functions ---------------------

    def unpack_fold_results_and_average(self, fold_results: dict):
        """For each cross validation fold, average the returned statistics to obtain a single value for each metric"""
        r2_scores = []
        coefficients = []
        p_values = []

        for _, results in fold_results.items():
            r2_scores.append(results["test_r2"])
            coefficients.append(results["train_coefficients"])
            p_values.append(results["train_p_values"])

        return np.mean(r2_scores), np.mean(coefficients, axis=0), np.mean(p_values, axis=0)

    def run_ols_model_with_cross_val(self, design_matrix: pd.DataFrame, dependent_variable: np.ndarray, dependent_var_name: str):
        """Run an OLS statsmodel regression on the data, conducting a 4-fold cross validation"""

        # drop the homing id column from the design matrix to leave only the predictors
        X = design_matrix.drop(columns=["homing_id"])
        y = dependent_variable
        groups = design_matrix["homing_id"].to_numpy()  # Grouping according to homing_id

        # Set up cross validation by homing id
        group_kfold = GroupKFold(n_splits=4)
        assert len(np.unique(groups)) == len(np.unique(design_matrix["homing_id"])), "Number of groups is not equal to the number of homing ids"

        # Saveing info
        ols_save_path = self.save_path / "ols_regression" / dependent_var_name
        ols_fold_results = {}

        # Split using GroupKFold, ensuring groups do not overlap between folds
        for fold, (train_index, test_index) in enumerate(group_kfold.split(X, y, groups)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            ols_fold_results[fold] = self.ols_regression_statsmodel(X_train, y_train, fold, ols_save_path, X_test, y_test)

        return ols_fold_results

    # ------------------- Statistical functions -------------------

    def repeat_observation_ttest(
        self, indices_of_sig_coeffs: np.ndarray, original_model_coeffs: np.ndarray, comparison_model_coeffs: np.ndarray
    ) -> tuple:
        """Perform a paired t test to see if the coefficients between the original model and the comparison model are significantly different

        The null hypothesis is that the coefficients are the same between the two models.
            If the p value is less than 0.05 then the change is significant which is bad.
            If the p value is greater than 0.05 then the change is not significant which is good

        Returns:
            tuple: t_stat, p_value both floats"""
        t_stat, p_value = ttest_rel(original_model_coeffs[indices_of_sig_coeffs], comparison_model_coeffs[indices_of_sig_coeffs])
        return t_stat, p_value

    # ------------------- Regression functions -------------------

    def ols_regression_sklearn(self, X_train, y_train, fold, save_path: Path, X_test, y_test) -> dict:
        """Performing the OLS regression

        NOTES:
        - first coefficient is the intercept
        - first p-value is for the intercept
        - second p-value is for the first coefficient

        Returns:
            dict: A dictionary containing the training and test R2 scores, coefficients and p-values
        """

        # ------------------- Training data -------------------
        ols_model = LinearRegression(fit_intercept=True)
        ols_model.fit(X_train, y_train)
        train_coefficients = ols_model.coef_
        train_r2 = ols_model.score(X_train, y_train)

        # Predict data for training data
        train_pred = ols_model.predict(X_train)

        # Plot training predictions
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.plot(np.arange(len(y_train)), y_train, label="True")
        plt.plot(np.arange(len(train_pred)), train_pred, label="Predicted")
        plt.legend()
        plt.title(f"Fold {fold + 1} Train Data")

        # ------------------- Test data -------------------
        # Refit the model on the test data
        test_pred = ols_model.predict(X_test)
        test_r2 = r2_score(y_test, test_pred)
        test_coefficients = ols_model.coef_

        # Plot test predictions
        plt.subplot(122)
        plt.plot(np.arange(len(y_test)), y_test, label="True")
        plt.plot(np.arange(len(test_pred)), test_pred, label="Predicted")
        plt.legend()
        plt.title(f"Fold {fold + 1} Test Data")
        plt.savefig(save_path / f"fold_{fold + 1}_train_vs_test.png")
        plt.close()

        # NOTE - p-values are not available in sklearn - matching the statsmodel output so that the dictionary is consistent

        return {
            "train_coefficients": train_coefficients,
            "train_p_values": np.zeros(len(train_coefficients)),
            "train_r2": train_r2,
            "test_coefficients": test_coefficients,
            "test_p_values": np.zeros(len(train_coefficients)),
            "test_r2": test_r2,
            "test_predictions": test_pred,
        }

    def ols_regression_statsmodel(self, X_train, y_train, fold, save_path: Path, X_test, y_test) -> dict:
        """Performing the OLS regression

        NOTES:
         - first coefficient is the intercept
         - first p-value is for the intercept
         - second p-value is for the first coefficient

         Returns:
            dict: A dictionary containing the training and test R2 scores, coefficients and p-values
        """

        # ------------------- Training data -------------------
        X_train = sm.add_constant(X_train)
        train_mod = sm.OLS(y_train, X_train)
        train_results = train_mod.fit()
        train_coefficients = train_results.params[1:]  # Exclude the intercept which is the first coefficient
        train_p_values = train_results.pvalues[1:]  # Exclude the intercept which is the first p-value
        train_r2 = train_results.rsquared
        # print(train_results.summary())

        # Predict data for training data
        train_pred = train_results.predict(X_train)

        # Plot training predictions
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.plot(np.arange(len(y_train)), y_train, label="True")
        plt.plot(np.arange(len(train_pred)), train_pred, label="Predicted")
        plt.legend()
        plt.title(f"Fold {fold + 1} Train Data")

        # ------------------- Test data -------------------
        X_test = sm.add_constant(X_test)
        test_pred = train_results.predict(X_test)
        test_r2 = r2_score(y_test, test_pred)

        # Plot test predictions
        plt.subplot(122)
        plt.plot(np.arange(len(y_test)), y_test, label="True")
        plt.plot(np.arange(len(test_pred)), test_pred, label="Predicted")
        plt.legend()
        plt.title(f"Fold {fold + 1} Test Data")
        plt.savefig(save_path / f"fold_{fold + 1}_train_vs_test.png")
        plt.close()

        return {
            "train_coefficients": train_coefficients,
            "train_p_values": train_p_values,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "test_predictions": test_pred,
        }

    def svr_regression(self, X_train, y_train, fold, save_path: Path, X_test, y_test):
        """Performing the SVR regression"""

        svr_model = SVR(kernel="rbf", C=0.8, epsilon=0.05)  # You can tune these parameters
        svr_model.fit(X_train, y_train)

        # Predict and calculate R2 for training data
        train_pred = svr_model.predict(X_train)
        train_r2 = svr_model.score(X_train, y_train)

        # test data
        test_pred = svr_model.predict(X_test)
        test_r2 = svr_model.score(X_test, y_test)

        # Plot train and test predictions into one plot, different subplots
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.plot(np.arange(len(y_train)), y_train, label="True")
        plt.plot(np.arange(len(train_pred)), train_pred, label="Predicted")
        plt.legend()
        plt.title(f"svr Fold {fold + 1} Train Data")

        plt.subplot(122)
        plt.plot(np.arange(len(y_test)), y_test, label="True")
        plt.plot(np.arange(len(test_pred)), test_pred, label="Predicted")
        plt.legend()
        plt.title(f"Svr Fold {fold + 1} Test Data")
        plt.savefig(save_path / f"fold_{fold + 1}_train_vs_test.png")

        return test_r2

    # ------------------- Plotting functions -------------------

    def plot_shift_results(self, shifts, og_r2, shifted_r2_ols):
        """Plot the mean R2 scores for different shifts"""

        shifts = np.array(shifts)

        plt.figure(figsize=(12, 6))
        plt.plot(shifts, shifted_r2_ols, label="OLS R2", marker="o")
        plt.axhline(y=og_r2, color="b", linestyle="--", label="OLS Original R2")
        plt.xlabel("Shift Amount by frame")
        plt.ylabel("Mean R2 Score")
        plt.title("Mean R2 Score vs. Shift Amount")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.save_path / "shifted_mean_r2_scores.png")
        plt.close()

    def plot_the_r2_scores_for_all_dependents(self):
        """Plot the R2 scores for the different dependent variables"""

        dependent_vars = list(self.r2_score_dic.keys())
        ols_r2_scores = [self.r2_score_dic[dependent_var] for dependent_var in dependent_vars]
        plt.figure(figsize=(10, 6))
        x = np.arange(len(dependent_vars))
        bar_width = 0.35
        plt.bar(x + bar_width / 2, ols_r2_scores, bar_width, label="OLS")
        # Plot the r2 score as text above the bar
        for i, r2 in enumerate(ols_r2_scores):
            plt.text(x[i], r2 + 0.01, str(np.around(r2, 2)), color="black", ha="center")
        plt.xlabel("Dependent Variable")
        plt.ylabel("Mean R2 Score")
        plt.title("Mean R2 Score for Different Dependent Variables")
        plt.xticks(x + bar_width / 2, dependent_vars)
        plt.xticks(rotation=25)
        plt.xticks(fontsize=6)
        plt.legend()
        plt.grid(True)
        plt.savefig(self.save_path / "dependent_variable_r2_scores.png")
        plt.close()

    def plot_lines_between_points_on_plot(
        self, ax, x1: np.ndarray, x2: np.ndarray, original_model_coeffs: np.ndarray, comparison_model_coeffs: np.ndarray, iterator: np.ndarray
    ) -> None:
        """Connect the corresponding points with lines on a 2d plot to see changes across different models

        Args:
            ax: The axis to plot on
            x1: The x values for the first model
            x2: The x values for the second model
            original_model_coeffs: The coefficients for the first model
            comparison_model_coeffs: The coefficients for the second model
            iterator: The indices to iterate over to connect the points"""

        assert len(x1) == len(x2) == len(original_model_coeffs) == len(comparison_model_coeffs), "Lengths of arrays are not the same"
        for i in iterator:
            ax.plot([x1[i], x2[i]], [original_model_coeffs[i], comparison_model_coeffs[i]], color="gray", linestyle="--", linewidth=0.5)

    def plot_coefficients_between_models(self, x1, x2, original_model_coeffs, comparison_model_coeffs, sig_og_model_coeffs_indices):
        """Plot all and only the significant coefficients between the two models in separate plots"""

        # Create two axes
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot all coefficients on the first plot
        ax1.scatter(x1, original_model_coeffs, label="Neural data only")
        ax1.scatter(x2, comparison_model_coeffs, label="Neural data + other predictors correlated with the index")
        ax1.legend()
        ax1.set_xticks([1, 2])
        ax1.set_xticklabels(["Neural data only", "Neural data + \n other predictors"])
        ax1.set_ylabel("Coefficients")
        ax1.set_title("Coefficients for neural data only vs neural data + other predictors correlated with the index")
        self.plot_lines_between_points_on_plot(ax1, x1, x2, original_model_coeffs, comparison_model_coeffs, range(len(original_model_coeffs)))

        # Plot only the significant coefficients on the second plot
        ax2.scatter(x1[sig_og_model_coeffs_indices], original_model_coeffs[sig_og_model_coeffs_indices], label="Neural data only")
        ax2.scatter(
            x2[sig_og_model_coeffs_indices],
            comparison_model_coeffs[sig_og_model_coeffs_indices],
            label="Neural data + other predictors that are correlated with the index",
        )
        ax2.legend()
        ax2.set_xticks([1, 2])
        ax2.set_xticklabels(["Neural data only", "Neural data + \n other predictors"])
        ax2.set_ylabel("Coefficients")
        ax2.set_title("Significant coefficients for neural data only vs neural data + other predictors correlated with the index")
        self.plot_lines_between_points_on_plot(ax2, x1, x2, original_model_coeffs, comparison_model_coeffs, sig_og_model_coeffs_indices)

        # Put sig test of difference in title
        t_stat, p_value = self.repeat_observation_ttest(sig_og_model_coeffs_indices, original_model_coeffs, comparison_model_coeffs)
        are_results_significant = "significant" if p_value < 0.05 else "not significant"
        fig.suptitle(
            f"Significant coefficients for neural data only vs neural data + other predictors correlated with the index. \n T-stat: {t_stat}, p-value: {p_value}. \n Difference in coefficients is {are_results_significant}"
        )

        plt.savefig(self.save_path / "coefficients_between_models.png")

    def plot_proportion_of_coeffs_that_remain_significant(
        self, original_model_pvalues: np.ndarray, comparison_model_pvalues: np.ndarray, og_r2_score, comp_r2_score, alpha=0.05
    ) -> None:
        """Plots a bar chart showing the proportion of coefficients that remain significant between the two models, neural coefficients only"""

        # Check the number of coefficients that are significant in each model
        og_moel_significant_coeffs = np.sum(original_model_pvalues < alpha)
        all_predictors_significant = np.sum(comparison_model_pvalues < alpha)

        # Check the numebr of cells that remain sig after adding all predictors
        num_sig = 0
        num_sig = np.sum(
            [num_sig + 1 for i in range(len(original_model_pvalues)) if original_model_pvalues[i] < 0.05 and comparison_model_pvalues[i] < 0.05]
        )

        # Plot the number of cells that remain significant
        _, ax = plt.subplots()
        ax.bar(
            ["Neural data only", "Neural data + other predictors", "consistently sig"],
            [og_moel_significant_coeffs, all_predictors_significant, num_sig],
        )
        ax.set_ylabel("Number of significant coefficients")
        ax.set_title(
            f"Out of the {len(original_model_pvalues)} total coefficients, {num_sig} are significant under both models after adding additional predictors. Alpha = 0.05 \n Original R2: {og_r2_score}, Comparison R2: {comp_r2_score}"
        )
        plt.savefig(self.save_path / "proportion_of_significant_coeffs.png")

    # -------------------- Shifted data analysis --------------------

    def run_all_shifts(self):
        """As a quick pass, shift the neural data backwards and forwards and see how the R2 score changes with each shift"""
        
        shifted_r2_ols = []
        og_r2 = None
        shifts = np.arange(-400, 400, 10) # Shift forwards and backwards by 400 frames (40 frames is 1 second)        
        for shift_amount in tqdm(shifts, desc="Processing shifts"):
            if shift_amount == 0:
                shifted_design_matrix = self.design_matrix
            else:
                shifted_design_matrix = self.shift_spikes(self.design_matrix, shift_amount)
            r2, _, _ = self.run_just_one_dependent_variable("index", shifted_design_matrix)
            if shift_amount == 0: 
                og_r2 = r2
            shifted_r2_ols.append(r2)
        
        logger.success("Shifts have been completed")
        
        return shifts, og_r2, shifted_r2_ols

    def shift_spikes(self, design_matrix: pd.DataFrame, shift_amount: int) -> pd.DataFrame:
        """Shifts the spikes in the design matrix by a certain amount, wrapping around the end of the matrix"""
        new_index = (design_matrix.index + shift_amount) % len(design_matrix)
        return design_matrix.reindex(new_index).reset_index(drop=True)


# ------------------------------------------------- Hard code some data to test the classes -----------------------------------------------------------------------

if __name__ == "__main__":

    # Load data to test the SingleTrialRegression class
    video_df = pl.read_csv(r"E:\efizz\JAL006\JAL006_shelter_barrier_flip_6_2024_03_28T10_54_20\processed_data\full_video_dataframe.csv")
    homie_path = r"E:\efizz\JAL006\JAL006_shelter_barrier_flip_6_2024_03_28T10_54_20\processed_data\homings\homings_obj.pkl"
    with open(homie_path, "rb") as dill_file:
        homings = pickle.load(dill_file)
    frame_by_cluster_matrix = np.load(
        r"E:\efizz\JAL006\JAL006_shelter_barrier_flip_6_2024_03_28T10_54_20\processed_data\frame_by_good_cluster_matrix.npy"
    )
    video_and_spike_data = pl.read_parquet(
        r"E:\efizz\JAL006\JAL006_shelter_barrier_flip_6_2024_03_28T10_54_20\processed_data\good_video_spike_count_df.parquet"
    )

    save_path = Path(r"E:\efizz\JAL006\JAL006_shelter_barrier_flip_6_2024_03_28T10_54_20\processed_data\single_trial")

    # Run the pipeline
    pp = PreprocessSingleTrialRegression(
        video_df=video_df,
        homings_obj=homings,
        video_and_spike_data=video_and_spike_data,
        frame_by_cluster_matrix=frame_by_cluster_matrix,
        save_path=save_path,
        similar_homings=False,
        orthogonalise_index=False,
    )
    SingleTrialRegression(
        design_matrix=pp.design_matrix,
        save_path=save_path,
        all_dependent_names=pp.here_are_all_the_columns,
        targets_df=pp.targets_df,
    )
