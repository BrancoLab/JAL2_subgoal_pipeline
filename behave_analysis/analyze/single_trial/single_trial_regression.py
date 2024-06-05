"""
TODO:
-- Remove shelter times?
"""

from pathlib import Path

import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import dill as pickle
import polars as pl
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from scipy import stats
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from scipy.stats import ttest_rel
from tqdm import tqdm

from behave_analysis.analyze.filtering_data.filtering_functions import filter_video_dataframe
from behave_analysis.utils.creating_directories import make_directory
from behave_analysis.analyze.single_trial.tests import UnitTests


class PreprocessSingleTrialRegression:
    """Preprocess the data ready for single trial regression analysis. All homings are
    included in the analysis.

    Args:
         video_df (pl.DataFrame): The video dataframe
         homings_obj (object): The homings object
         video_and_spike_data (pl.DataFrame): The video with the neural data joined
         frame_by_cluster_matrix (np.ndarray): The frame by cluster matrix
         save_path (Path): The save path
         velocity_data (np.ndarray): The velocity data
         similar_homings (bool, optional): Whether to select similar homings. Defaults to False

     Returns:
         (object) The single trial regression object with two important attributes such as:
             - design_matrix (pd.DataFrame): The design matrix
             - targets_df (pd.DataFrame): The potential dependent variables
    """

    def __init__(
        self,
        video_df,
        homings_obj,
        video_and_spike_data,
        frame_by_cluster_matrix,
        save_path,
        velocity_data,
        similar_homings=False,
    ):
        logger.info("Initializing the single trial regression preprocessing object")
        self.homings_obj = homings_obj
        self.video_and_spike_data = video_and_spike_data
        self.frame_by_cluster_matrix = frame_by_cluster_matrix
        self.save_path = save_path
        self.video_df = self.remove_columns_from_video_df(video_df)

        # Settings
        self.similar_homings = similar_homings

        # Unit tests
        UnitTests.check_attributes_of_homing_dic(self.homings_obj)
        UnitTests.check_index_is_valid(self.compute_index)

        # Preprocessing homing data
        homing_df_s1 = self.preprocess_homing_data(select_similar_homings=self.similar_homings)

        # Add the dependent variable to the data
        homing_df_s2 = self.add_dependent_index_variable_to_homing_info(homing_data_single_dataframe=homing_df_s1)

        # Add the velocity data to the homing data
        self.homing_data_single_dataframe = self.add_velocity_data_to_homing_data(homing_df_s2, velocity_data)

        # Create the design matrix
        self.design_matrix = self.create_the_design_matrix(self.homing_data_single_dataframe, self.frame_by_cluster_matrix)
        self.targets_df = self.create_dependent_dataframe(self.homing_data_single_dataframe)

        # Descriptive plots
        self.plot_homing_durations()
        self.plot_y_coords_distribution()
        self.plot_the_index_distribution()
        self.plot_the_index_per_homing()

        logger.success("The single trial regression preprocessing object has been initialized")

    # ------- Descriptive exploratory plots based on preprocessing -------------------------

    def plot_neural_firing_rate_over_time(self, design_matrix: pd.DataFrame) -> None:
        """Given we have a fit for variables that increase over time, we can plot the neural firing rate over time to see if it is increasing over time"""

        # Drop the homing id column
        design_matrix = design_matrix.drop(columns=["homing_id"])

        # Plot mean firing rate over time
        mean_firing_rate = design_matrix.mean(axis=1)

        # Smoothing the mean firing rate
        mean_firing_rate = mean_firing_rate.rolling(window=40).mean()

        # Plot a line for each neuron with some trnasparency
        for neuron in design_matrix.columns:
            neural_data = design_matrix[neuron]
            smoothed_neural_data = neural_data.rolling(window=40).mean()
            plt.plot(smoothed_neural_data, alpha=0.1)

        # Plot the mean firing rate
        plt.plot(mean_firing_rate, color="black", label="Mean firing rate")
        plt.xlabel("Frames")
        plt.ylabel("Firing rate")
        plt.title("Firing rate over time")
        plt.legend()
        plt.savefig(self.save_path / "firing_rate_over_time.png")

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
        index = self.homing_data_single_dataframe["index_south"].to_numpy()
        plt.hist(index, bins=20)
        plt.xlabel("Index")
        plt.ylabel("Number of frames")
        plt.title("Distribution of the south index")
        plt.savefig(self.save_path / "index_south_distribution.png")
        plt.close()

        index = self.homing_data_single_dataframe["index_north"].to_numpy()
        plt.hist(index, bins=20)
        plt.xlabel("Index")
        plt.ylabel("Number of frames")
        plt.title("Distribution of the north index")
        plt.savefig(self.save_path / "north_index_distribution.png")
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
            plt.title(f"Index for homing {homing_id}. Arrow is hdir, text is index. Blue is north, red is south")

            # convert to pandas for the text
            homing_pd = homing.to_pandas()

            # Add the index every 3rd frame to reduce clutter
            for i, row in homing_pd.iloc[::3].iterrows():
                plt.text(
                    x=row["mouse_x_position"] + 20, y=row["mouse_y_position"] + 10, s=str(np.around(row["index_south"], 1)), color="red", fontsize=6
                )
                plt.text(
                    x=row["mouse_x_position"] + 100, y=row["mouse_y_position"] + 10, s=str(np.around(row["index_north"], 1)), color="blue", fontsize=6
                )
                # Add velocity
                plt.text(
                    x=row["mouse_x_position"] + 150, y=row["mouse_y_position"] + 20, s=str(np.around(row["velocity"], 1)), color="green", fontsize=6
                )

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

        for dependent_variable in self.targets_df.columns:
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

    # ----------------- Functions for extracting the homing data ---------------------

    def extract_data_from_homings(self, homing_object: dict, video_df: pl.DataFrame) -> list:
        """Extract the associated behavioural data between homing onsets and offsets.

        Returns:
             (list) of homing dataframes for each homing period"""

        assert UnitTests.check_frame_indexes_are_incremental(video_df["frames"].to_numpy()), "Frames are missing in the homing information"
        assert video_df["frames"].to_numpy()[0] == 1, "The frames do not start at 1"

        homing_info = []
        for onset, offset in zip(homing_object.onset_frames, homing_object.offset_frames):
            homing = video_df[int(onset) - 1 : int(offset) - 1]  # Substract 1 to prevent off by one error
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

    def add_homing_id_to_homing_data(self, extracted_homing_info: list) -> list:
        """Adding the homing id (abitrary ascending interger) to the homing data.
        This is needed for the group cross validation object

        Args:
            extracted_homing_info (list): A list of homing dataframes for each homing period

        Returns:
            (list) of homing dataframes with the homing id added as a column
        """
        for idx, homing in enumerate(extracted_homing_info):
            updated_homing = homing.with_columns(pl.lit(idx).alias("homing_id"))
            extracted_homing_info[idx] = updated_homing
        return extracted_homing_info

    def concatenate_the_homing_data(self, homing_info: list) -> pl.DataFrame:
        """Take each element of the list and concatenate them into a single dataframe"""
        for idx, homing in enumerate(homing_info):
            if idx == 0:
                homing_data = homing
            else:
                homing_data = homing_data.vstack(homing)
        return homing_data

    def preprocess_homing_data(self, select_similar_homings) -> pl.DataFrame:
        """Preprocessing the data into a single dataframe for regression analysis"""
        extracted_homing_info = self.extract_data_from_homings(homing_object=self.homings_obj, video_df=self.video_df)
        if select_similar_homings:
            self.homing_info = self.select_similar_homings(extracted_homing_info)
            extracted_homing_info = self.homing_info
        homing_info = self.add_homing_id_to_homing_data(extracted_homing_info)
        return self.concatenate_the_homing_data(homing_info)

    def add_velocity_data_to_homing_data(self, homing_data_single_dataframe: pl.DataFrame, velocity_data: np.ndarray) -> pd.DataFrame:
        """Adding the velocity data to the homing data
        
        Args:
            homing_data_single_dataframe (pl.DataFrame): The homing data
            velocity_data (np.ndarray): The velocity data taken from loading the tracking data"""

        # Add zero to start of the velocity data to make it the same length as the homing data
        velocity_data = np.insert(arr = velocity_data, obj = 0, values = 0) # obj is the index to insert the value

        # Extract velocity at the corresponding frames
        frames = homing_data_single_dataframe["frames"].to_numpy()
        velocity_data = velocity_data[frames]
        assert len(velocity_data) == len(homing_data_single_dataframe), "The length of the velocity data is not the same as the homing data"

        return homing_data_single_dataframe.with_columns(pl.Series("velocity", velocity_data))

    # ----------------- Functions for computing the index --------------------------------

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

    def add_dependent_index_variable_to_homing_info(self, homing_data_single_dataframe: pl.DataFrame):
        """Add two index variables to the homing data one for each subgoal.

        The index ranges from -1 which is when the mouse is facing the shelter
        and 1 when the mouse is facing the goal. This is a normalised index."""

        hsa = homing_data_single_dataframe["hsa"].to_numpy().copy()
        south_goal = homing_data_single_dataframe["h_bar_south_a"].to_numpy().copy()
        north_goal = homing_data_single_dataframe["h_bar_north_a"].to_numpy().copy()
        UnitTests.check_angles_are_between_minus_pi_and_pi(hsa, south_goal, north_goal)

        # if values negative radians then add 2pi to make them positive and easier to work with
        hsa = np.where(hsa < 0, hsa + 2 * np.pi, hsa)
        south_goal = np.where(south_goal < 0, south_goal + 2 * np.pi, south_goal)
        north_goal = np.where(north_goal < 0, north_goal + 2 * np.pi, north_goal)

        south_index = self.compute_index(hsa, south_goal)
        north_index = self.compute_index(hsa, north_goal)
        UnitTests.check_index_values_are_valid(south_index, north_index)

        result = homing_data_single_dataframe.with_columns(
            [
                (pl.Series("index_south", south_index)),
                (pl.Series("index_north", north_index))
            ]
        )
        return result

    # ------------ Create design matrix and targets-----------------------------

    def create_dependent_dataframe(self, homing_data_single_dataframe: pl.DataFrame) -> pl.DataFrame:
        """Create a polars dataframe containing different dependent variables as columns"""
        homing_data_single_dataframe = homing_data_single_dataframe.to_pandas()
        # add a random vector to the dependent variables to check if the model is overfitting
        homing_data_single_dataframe["random"] = np.random.rand(len(homing_data_single_dataframe)) * 50
        return homing_data_single_dataframe

    def normalise_design_matrix(self, design_matrix: np.array) -> pd.DataFrame:
        """Normalise the design matrix using a simple scale by the standard deviation"""
        np_std = np.std(design_matrix, axis=0)
        np_std[np_std == 0] = 1  # Avoid division by zero
        normalised_design_matrix = np.divide(design_matrix, np_std)
        return pd.DataFrame(normalised_design_matrix)

    def create_the_design_matrix(self, homing_data: pl.DataFrame, frame_by_cluster_matrix: np.ndarray) -> np.ndarray:
        """Creating the design matrix of shape (n_frames, n_neurons) for the regression analysis.
        
        Remembering that the each homing has been concatenated together so now we need to index the corresponding
        spike data for each homing period. NOTE: The homing id is added to the design matrix for the group cross validation
        and must be removed before running the regression analysis.
        
        Args:
            data (pl.DataFrame): The homing data
            frame_by_cluster_matrix (np.ndarray): The frame by cluster matrix with smoothed spike counts in each cell"""

        # Initialising the design matrix
        data = homing_data
        total_frames = len(data)
        total_features = frame_by_cluster_matrix.shape[1] # The number of neurons
        design_matrix = np.zeros((total_frames, total_features)) # (F, N)

        counter = 0
        homing_ids_len = len(np.unique(data["homing_id"]))
        for idx in range(homing_ids_len):

            # Get the frames for the homing id for slicing
            frames = data.filter(data["homing_id"] == idx)["frames"].to_numpy()

            # Get the corresponding frame by cluster matrix
            # minus 1 to prevent off by one error, +1 to include the last frame
            spike_data = frame_by_cluster_matrix[frames[0] - 1 : frames[-1] -1 + 1] # left the -1 + 1 in the second index to make it more readable

            # Add the spike data to the design matrix
            design_matrix[counter : counter + len(spike_data)] = spike_data
            counter += len(spike_data)

        design_matrix = self.normalise_design_matrix(design_matrix)
        design_matrix["homing_id"] = data["homing_id"].to_numpy() 

        return design_matrix


class SingleTrialRegression:
    """A class that performs single trial regression analysis on the data"""

    def __init__(self, design_matrix: pd.DataFrame, save_path: Path, dependents_df: pd.DataFrame):
        logger.info("Initializing the single trial regression analysis object")
        self.design_matrix = design_matrix
        self.save_path = save_path
        self.dependent_names = list(dependents_df.columns)
        logger.info("The dependent variables to run in this regression are: {}", self.dependent_names)
        self.dependents_df = dependents_df
        self.number_of_neurons = self.design_matrix.shape[1] - 1  # Subtract 1 for the homing id column
        self.angular_dependent_vars = [
            "h_bar_north_a",
            "hdir",
            "hsa",
            "h_bar_south_a",
        ]  # define here all potential angular variables so we can switch between coordinate systems
        self.encompasing_set = ["h_bar_north_a", "hdir", "hsa", "h_bar_south_a", "mouse_y_position", "velocity"]

        sig_index_coefficients_indices = self.run(
            run_all_dependent_variables=True, shift_neural_data=False, explore_coeffs_with_other_predictors=True, run_hiarchical_regression=True
        )

        # Plot a heatplot of the design matrix
        self.plot_clustered_heatmap(self.design_matrix, dependents_df=self.dependents_df, significant_neuron_ids=sig_index_coefficients_indices)

    # --------------------- Functions for running the regression ---------------------

    def run(
        self, run_all_dependent_variables: bool, shift_neural_data: bool, explore_coeffs_with_other_predictors: bool, run_hiarchical_regression: bool
    ) -> None:
        """This function runs the regression with different modes depending on the analysis you want to conduct:

        Modes:
        - run_all_dependent_variables: Run the model for all dependent variables
        - shift_neural_data: Shift the neural data and run the model to see how the R2 score changes with chance
        - explore_coeffs_with_other_predictors: Explore the coefficients with other predictors to see how they change"""

        if run_all_dependent_variables:
            r2_score_for_all_dependents, _, _, mse = self.run_all_dependent_variables()
            self.plot_the_r2_scores_for_all_dependents(r2_scores=r2_score_for_all_dependents)
            logger.success("The model has been run for all dependent variables")

        if shift_neural_data:
            # NOTE - Still only works for one index location at the moment
            shifts, og_r2, shifted_r2_ols = self.run_all_shifts()
            self.plot_shift_results(shifts, og_r2, shifted_r2_ols)
            logger.success("The model has been run for all shifts")

        if explore_coeffs_with_other_predictors:

            # Where og is the original gangster and comp is the comparison model
            INDEX = "index_north"
            # index = "index_south"

            og_r2_score, og_coefficients, og_p_values = self.run_just_one_dependent_variable(INDEX, self.design_matrix)
            comparison_design_matrix = self.add_other_predictors_to_design_matrix(self.design_matrix, self.encompasing_set)
            comp_r2_score, comp_predictors_coeffs, comp_predictors_p_value = self.run_just_one_dependent_variable(INDEX, comparison_design_matrix)

            # Select only the neural data coefficients and p values (Excluding intercept and non neural coeffs)
            og_coefficients = og_coefficients[: self.number_of_neurons]
            og_p_values = og_p_values[: self.number_of_neurons]
            comp_predictors_coeffs = comp_predictors_coeffs[: self.number_of_neurons]
            comp_predictors_p_value = comp_predictors_p_value[: self.number_of_neurons]

            # Take the indexes that are significant in both models
            sig_in_both_models = np.where((og_p_values < 0.05) & (comp_predictors_p_value < 0.05))[0]

            sig_index_coefficients_indices = np.where(og_p_values < 0.05)  # Which coefficients are significant in the original model
            self.plot_proportion_of_coeffs_that_remain_significant(og_p_values, comp_predictors_p_value, og_r2_score, comp_r2_score)
            # Check whether the significant coefficients change significantly between the two models
            x1 = np.ones(len(og_coefficients))
            x2 = 2 * np.ones(len(comp_predictors_coeffs))
            self.plot_coefficients_between_models(x1, x2, og_coefficients, comp_predictors_coeffs, sig_index_coefficients_indices)
            logger.success("The model has been run to compare base model with encompassing model")

        if run_hiarchical_regression:
            h_results = self.run_hiararchical_regression(INDEX)
            self.plot_adjusted_r2_scores_for_hierarchy(h_results)
            logger.success("The model has been run for the hierarchical regression")

        # check if sig_index_coefficients_indices is defined
        assert "sig_index_coefficients_indices" in locals(), "sig_index_coefficients_indices is not defined"

        return sig_in_both_models

    def run_just_one_dependent_variable(self, dependent_var_name: str, design_matrix: pd.DataFrame) -> tuple:
        """Run the model for just one dependent variable"""

        # Create a save location for the dependent variable
        make_directory(self.save_path / "ols_regression" / dependent_var_name)

        ols_fold_results = self.run_ols_model_with_cross_val(
            design_matrix=design_matrix, dependent_variable=self.dependents_df[dependent_var_name].to_numpy(), dependent_var_name=dependent_var_name
        )
        dic = self.unpack_fold_results_and_average(ols_fold_results)
        r2_scores, coefficients, p_values = dic["mean_r2"], dic["mean_coefficients"], dic["mean_p_values"]

        return r2_scores, coefficients, p_values

    def run_all_dependent_variables(self):
        """Run the model for different dependent variables and store the mean R2 scores
        to see which dependent variable is best predicted by the neural data alone"""

        # Init some storage vars
        r2_scores = {}
        p_values = {}
        coefficients = {}
        mse = {}

        # Extract the dependent variables and make save directories for each of them to store the results
        for dependent_var_name in self.dependent_names:
            make_directory(self.save_path / "ols_regression" / dependent_var_name)

        # Run the model for each dependent variable
        for var_idx, var_name in enumerate(self.dependent_names):
            logger.info("Running the model for dependent variable: {}", var_name)
            multiple_dependent_scenario = False

            # If the dependent variable is not angular dependent then run a standard OLS model
            if var_name in self.angular_dependent_vars:
                multiple_dependent_scenario = True

            # A dictionary where each key is a fold index
            ols_fold_results = self.run_ols_model_with_cross_val(
                design_matrix=self.design_matrix,
                dependent_variable=self.dependents_df[var_name].to_numpy(),
                dependent_var_name=var_name,
                multi_dependent=multiple_dependent_scenario,
            )

            # Unpack the results and average them
            dic = self.unpack_fold_results_and_average(ols_fold_results)
            r2_scores[var_name], p_values[var_name], coefficients[var_name], mse[var_name] = (
                dic["mean_r2"],
                dic["mean_p_values"],
                dic["mean_coefficients"],
                dic["mean_mse"],
            )

        return r2_scores, p_values, coefficients, mse

    def run_hiararchical_regression(self, INDEX):
        """Conduct a looping hierarchical regression analysis where predictors are added one by one and the adjusted R2 is calculated"""
        # hardcode subsets
        s0 = []  # empty set just neural data
        s1 = ["hdir"]
        s2 = ["hdir", "velocity"]
        s3 = ["hdir", "velocity", "mouse_y_position"]
        s4 = ["hdir", "velocity", "mouse_y_position", "hsa"]
        s5 = ["hdir", "velocity", "mouse_y_position", "hsa", "h_bar_north_a"]
        full_set = [s0, s1, s2, s3, s4, s5, self.encompasing_set]

        # store the results
        hirarchical_results = {}

        for i, subset in enumerate(full_set):
            # First add the subset to the design matrix
            design_matrix = self.add_other_predictors_to_design_matrix(self.design_matrix, dependents_to_add=subset)

            # Run cross validation
            ols_fold_results = self.run_ols_model_with_cross_val(
                design_matrix=design_matrix,
                dependent_variable=self.dependents_df[INDEX].to_numpy(),
                dependent_var_name=INDEX,
            )

            unpacked_results = self.unpack_fold_results_and_average(ols_fold_results)
            unpacked_r2 = unpacked_results["mean_r2"]
            number_of_predictors = design_matrix.shape[1] - 1  # Subtract 1 for the homing id column
            avg_observations = len(self.design_matrix)
            adjusted_r2 = self.calculate_adjusted_r2(unpacked_r2, avg_observations, number_of_predictors)
            hirarchical_results[i] = adjusted_r2

        return hirarchical_results

    # ------------------------ Changing the design matrix or targets ------------------

    def add_other_predictors_to_design_matrix(self, design_matrix: pd.DataFrame, dependents_to_add: list) -> pd.DataFrame:
        """Add other predictors to the design matrix to see how they affect the coefficients and p values compared to the neural data alone

        If empty list is passed then the function will just return the original design matrix

        Logic:
        -- If the dependent variable is not angular then divide by the standard deviation
        -- If the dependent variable is angular then convert to sin and cos"""

        # Check if dependents to add is an empty list
        if dependents_to_add:
            design_matrix = design_matrix.copy()
            for dependent in dependents_to_add:

                # Scale non angular data when adding into design matrix
                if dependent not in self.angular_dependent_vars:
                    std = np.std(self.dependents_df[dependent].to_numpy())
                    design_matrix[dependent] = self.dependents_df[dependent].to_numpy() / std

                # Convert angular dependent variables to sin and cos
                else:
                    circ_var = self.dependents_df[dependent].to_numpy()
                    sin_component = np.sin(circ_var)
                    cos_component = np.cos(circ_var)
                    design_matrix[f"{dependent}_sin"] = sin_component
                    design_matrix[f"{dependent}_cos"] = cos_component

        return design_matrix

    def convert_angular_dependent_variables_to_cartesian(self, dependent_variable: np.ndarray) -> np.ndarray:
        """Split the dependent variable into two components, sin and cos"""
        sin_component = np.sin(dependent_variable)
        cos_component = np.cos(dependent_variable)
        result = np.column_stack((sin_component, cos_component))
        assert result.shape[1] == 2, "The dependent variable has not been split into two components. Just checking the shape has two columns"
        return result

    # --------------------- Cross validation functions ---------------------

    def unpack_fold_results_and_average(self, fold_results: dict):
        """For each cross validation fold, average the returned statistics to obtain a single value for each metric"""
        r2_scores = []
        coefficients = []
        p_values = []
        mse = []

        for _, results in fold_results.items():
            r2_scores.append(results["test_r2"])
            coefficients.append(results["train_coefficients"])
            p_values.append(results["train_p_values"])
            mse.append(results["test_mse"])

        # --------- below code to handle outliers of fold effecting the mean -----------------
        # Should produce a more robust mean

        # Convert to numpy arrays for easy manipulation
        r2_scores = np.array(r2_scores)

        # Identify and filter out outliers in r2_scores
        r2_median = np.median(r2_scores)
        deviation_from_median = np.abs(r2_scores - r2_median)
        median_absolute_deviation = np.median(deviation_from_median)
        # Using a threshold of 3 times the median absolute deviation to identify outliers
        threshold = 3 * median_absolute_deviation
        filtered_r2_scores = r2_scores[deviation_from_median < threshold]

        # Compute the mean of the filtered r2_scores
        mean_r2 = np.mean(filtered_r2_scores) if len(filtered_r2_scores) > 0 else np.mean(r2_scores)

        # Compute mean MSE
        mean_mse = np.mean(mse)

        # return dic of results
        return {
            "mean_r2": mean_r2,
            "mean_mse": mean_mse,
            "mean_coefficients": np.mean(coefficients, axis=0),
            "mean_p_values": np.mean(p_values, axis=0),
        }

    def run_ols_model_with_cross_val(
        self, design_matrix: pd.DataFrame, dependent_variable: np.ndarray, dependent_var_name: str, n_splits: int = 5, multi_dependent: bool = False
    ):
        """Run an OLS statsmodel regression on the data, conducting a 4-fold cross validation"""

        # drop the homing id column from the design matrix to leave only the predictors
        X = design_matrix.drop(columns=["homing_id"])
        y = dependent_variable
        groups = design_matrix["homing_id"].to_numpy()  # Grouping according to homing_id

        if multi_dependent:
            y = self.convert_angular_dependent_variables_to_cartesian(y)

        # Set up cross validation by homing id
        group_kfold = GroupKFold(n_splits)
        assert len(np.unique(groups)) == len(np.unique(design_matrix["homing_id"])), "Number of groups is not equal to the number of homing ids"

        # Saveing info
        ols_save_path = self.save_path / "ols_regression" / dependent_var_name
        ols_fold_results = {}

        # Split using GroupKFold, ensuring groups do not overlap between folds
        for fold, (train_index, test_index) in enumerate(group_kfold.split(X, y, groups)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if multi_dependent:
                ols_fold_results[fold] = self.multi_output_ols_regression(
                    X_train, y_train, fold, ols_save_path, X_test, y_test, name_of_dependent=dependent_var_name
                )

            if not multi_dependent:
                ols_fold_results[fold] = self.ols_regression_statsmodel(
                    X_train, y_train, fold, ols_save_path, X_test, y_test, name_of_dependent=dependent_var_name
                )

        return ols_fold_results

    # ------------------- Statistical functions -------------------

    def repeat_observation_ttest(
        self, indices_of_sig_coeffs: np.ndarray, original_model_coeffs: np.ndarray, comparison_model_coeffs: np.ndarray
    ) -> tuple:
        """Perform a paired t test to see if the coefficients between the original model and the comparison model are significantly different

        The null hypothesis is that the coefficients are the same between the two models.
            If the p value is less than 0.05 then the change is significant which is bad.
            If the p value is greater than 0.05 then the change is not significant which is good

        In order to counteract a mean of zero issue due to a scaling around zero we take the absolute value of the coefficients

        Returns:
            tuple: t_stat, p_value both floats"""
        og_abs = np.abs(original_model_coeffs[indices_of_sig_coeffs])
        comp_abs = np.abs(comparison_model_coeffs[indices_of_sig_coeffs])
        t_stat, p_value = ttest_rel(og_abs, comp_abs)
        return t_stat, p_value

    def calculate_adjusted_r2(self, r2_score: float, n: int, p: int) -> float:
        """Calculate the adjusted R2 score

        Args:
            r2_score (float): The R2 score
            n (int): The number of samples
            p (int): The number of predictors"""
        adjusted_r2 = 1 - (1 - r2_score) * (n - 1) / (n - p - 1)
        return adjusted_r2

    def scaled_MSE_by_MAD(self, ygt: np.ndarray, ypred: np.ndarray) -> float:
        """Calculate the scaled MSE by the MAD"""

        # If the dependent var is angular we need to handle the two components
        if len(ygt.shape) == 2:
            ygt_mad = np.mean(np.abs(ygt - np.median(ygt, axis=0)), axis=0)
            mse = mean_squared_error(ygt, ypred, multioutput="raw_values")
            smse_mad = mse / (ygt_mad**2)
            return np.mean(smse_mad)

        # If non angular then just calculate the smae
        else:
            ygt_mad = np.mean(np.abs(ygt - np.median(ygt)))
            mse = mean_squared_error(ygt, ypred)
            smse_mad = mse / (ygt_mad**2)
            return smse_mad

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

    def ols_regression_statsmodel(self, X_train, y_train, fold, save_path: Path, X_test, y_test, name_of_dependent: str) -> dict:
        """Performing the OLS regression

        NOTES:
         - first coefficient is the intercept
         - first p-value is for the intercept
         - second p-value is for the first coefficient

         Returns:
            dict: A dictionary containing the training and test R2 scores, coefficients and p-values
        """

        # Check for NaNs
        UnitTests.check_for_nans_and_inf(X_train, X_test)

        # ------------------- Training data -------------------
        X_train = sm.add_constant(X_train)
        train_mod = sm.OLS(y_train, X_train)
        train_results = train_mod.fit()  # needed for when not using regularized attribute
        # train_results = sm.OLS(y_train, X_train).fit_regularized(method='elastic_net', L1_wt=1.0) # L1_wt=1.0 is lasso, L1_wt=0.0 is ridge - Where lasso will assign weights to one of two colinear variables
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
        test_mse = self.scaled_MSE_by_MAD(y_test, test_pred)

        # Plot test predictions
        plt.subplot(122)
        plt.plot(np.arange(len(y_test)), y_test, label="True")
        plt.plot(np.arange(len(test_pred)), test_pred, label="Predicted")
        plt.legend()
        plt.title(f"Fold {fold + 1} Test Data")

        plt.suptitle(f"R2 Train: {train_r2}, R2 Test: {test_r2} - {name_of_dependent}")

        plt.savefig(save_path / f"fold_{fold + 1}_train_vs_test.png")
        plt.close()

        return {
            "train_coefficients": train_coefficients,
            "train_p_values": train_p_values,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "test_predictions": test_pred,
            "test_mse": test_mse,
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

    def multi_output_ols_regression(self, X_train, y_train, fold, save_path: Path, X_test, y_test, name_of_dependent: str) -> dict:
        """NOTE - Pvalues and coefficients are fake this is just to get the R2 score for the multi output regression model"""

        # ------------------- Training data -------------------
        reg = MultiOutputRegressor(LinearRegression()).fit(X_train, y_train)
        train_r2 = reg.score(X_train, y_train)
        pred_train = reg.predict(X_train)

        # Recombine the sin and cos components into a single angular variable
        pred_train_y = np.arctan2(pred_train[:, 0], pred_train[:, 1])  # Convert back to radians
        real_train_y = np.arctan2(y_train[:, 0], y_train[:, 1])  # Convert back to radians

        # Plot training predictions
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.plot(np.arange(len(real_train_y)), real_train_y, label="True")
        plt.plot(np.arange(len(pred_train_y)), pred_train_y, label="Predicted")
        plt.legend()
        plt.title(f"Fold {fold + 1} Train Data")

        # ------------------- Test data ------------------------------------
        pred_test = reg.predict(X_test)
        test_r2 = reg.score(X_test, y_test)
        test_mse = self.scaled_MSE_by_MAD(y_test, pred_test)

        # Recombine the sin and cos components into a single angular variable
        pred_test_y = np.arctan2(pred_test[:, 0], pred_test[:, 1])  # Convert back to radians
        real_test_y = np.arctan2(y_test[:, 0], y_test[:, 1])  # Convert back to radians

        # Plot test predictions
        plt.subplot(122)
        plt.plot(np.arange(len(real_test_y)), real_test_y, label="True")
        plt.plot(np.arange(len(pred_test_y)), pred_test_y, label="Predicted")
        plt.legend()
        plt.title(f"Fold {fold + 1} Test Data. {name_of_dependent} - Test R2: {test_r2}")
        plt.savefig(save_path / f"fold_{fold + 1}_train_vs_test.png")
        plt.close()

        # Make up the p values and coefficients in this case
        train_coefficients = np.zeros(X_train.shape[1])
        train_p_values = np.zeros(X_train.shape[1])

        return {
            "train_coefficients": train_coefficients,
            "train_p_values": train_p_values,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "test_predictions": pred_test_y,
            "test_mse": test_mse,
        }

    # ------------------- Plotting functions -------------------

    def plot_clustered_heatmap(self, design_matrix: np.ndarray, dependents_df: pd.DataFrame, significant_neuron_ids) -> None:

        segment = 4000

        print("Plotting the clustered heatmap")
        # Plot a narrow subplot above imshow to show the dependent variable varying across the frames
        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(2, height_ratios=[1, 4])  # Adjust height ratios as needed

        # Plot the first subplot
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(dependents_df["index_north"][:segment], label="index_north")
        ax1.set_title("Index North Over Time")
        ax1.set_xlabel("Frame Index")
        ax1.set_ylabel("Index North")
        ax1.legend()

        # start x axis from 0
        ax1.set_xlim(0, len(dependents_df["index_north"][:segment]))

        # Plot the second subplot
        ax2 = fig.add_subplot(gs[1, :])

        # Filter the design matrix to only include the significant neurons
        # design_matrix = design_matrix[:, significant_neuron_ids[0][0]]
        design_matrix = design_matrix.drop(columns=["homing_id"])
        design_matrix = design_matrix.iloc[:, significant_neuron_ids]

        # Max firing rate across all neurons and frames
        max_firing_rate = design_matrix.max()

        # Normalize the matrix
        normalized_matrix = design_matrix / max_firing_rate

        # Create the heatmap on the second subplot
        heatmap = ax2.imshow(normalized_matrix.T.iloc[:, :segment], cmap="viridis", aspect="auto")
        # plt.colorbar(heatmap, ax=ax[1], orientation='vertical')  # Show color scale

        # Customize the plot (optional)
        ax2.set_title("Heatmap of the Matrix")
        ax2.set_xlabel("Frame Index")
        ax2.set_ylabel("Cluster Index")

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Show the plot
        plt.show()

        # # Plot a narrow subplot above imshow to show the dependent variable varying across the frames
        # fig, ax = plt.subplots(2, figsize=(10, 10))
        # ax[0].plot(dependents_df["index_north"], label="index_north")

        # # Remove the homing id column
        # design_matrix = design_matrix.drop(columns=["homing_id"])

        # # Max firing rate across all neurons and frames
        # max_firing_rate = design_matrix.max()

        # # Normalise the matrix
        # normalized_matrix = design_matrix / max_firing_rate

        # # Create the heatmap
        # plt.imshow(normalized_matrix.T, cmap='viridis', aspect='auto')

        # # plt.imshow(design_matrix.T, cmap='viridis', aspect='auto', norm=mpl.colors.Normalize(vmin=0, vmax=5), interpolation='nearest')
        # plt.colorbar()  # Show color scale

        # # Customize the plot (optional)
        # plt.title("Heatmap of the Matrix")
        # plt.xlabel("Frame Index")
        # plt.ylabel("Cluster Index")

        # # Show the plot
        # plt.show()

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

    def plot_the_r2_scores_for_all_dependents(self, r2_scores):
        """Plot the R2 scores for the different dependent variables"""

        dependent_vars = list(r2_scores.keys())
        ols_r2_scores = [r2_scores[dependent_var] for dependent_var in dependent_vars]
        plt.figure(figsize=(10, 6))
        x = np.arange(len(dependent_vars))
        bar_width = 0.35
        plt.bar(x + bar_width / 2, ols_r2_scores, bar_width, label="R2 scores")

        # Plot the mse score as text above the bar
        for i, r2 in enumerate(ols_r2_scores):
            height = ols_r2_scores[i]
            plt.text(x[i] + bar_width / 2, height + 0.01, str(np.around(ols_r2_scores[i], 2)), color="black", ha="center")

        plt.xlabel("Dependent Variable")
        plt.ylabel("Mean R2 Score")
        plt.title("Mean R2 Score for Different Dependent Variables.")
        plt.xticks(x + bar_width / 2, dependent_vars)
        plt.xticks(rotation=20)
        plt.xticks(fontsize=10)
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

        # Make coefficients absolute
        original_model_coeffs = np.abs(original_model_coeffs)
        comparison_model_coeffs = np.abs(comparison_model_coeffs)

        # Create two axes
        fig, ax = plt.subplots(1, figsize=(10, 8))

        # Plot only the significant coefficients on the second plot
        ax.scatter(x1[sig_og_model_coeffs_indices], original_model_coeffs[sig_og_model_coeffs_indices], label="Neural data only")
        ax.scatter(
            x2[sig_og_model_coeffs_indices],
            comparison_model_coeffs[sig_og_model_coeffs_indices],
            label="Neural data + other predictors that are correlated with the index",
        )
        ax.legend()
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Base model", "Encompassing model"])
        # rotate the x labels
        plt.xticks(rotation=20)
        ax.set_ylabel("Coefficients")
        ax.set_title("A check of whether the coeffs that are sig for both models change significantly when adding other predictors")
        self.plot_lines_between_points_on_plot(ax, x1, x2, original_model_coeffs, comparison_model_coeffs, sig_og_model_coeffs_indices)

        # Put sig test of difference in title
        t_stat, p_value = self.repeat_observation_ttest(sig_og_model_coeffs_indices, original_model_coeffs, comparison_model_coeffs)
        are_results_significant = "significant" if p_value < 0.05 else "not significant"
        formatted_p_value = format(p_value, ".4f")
        rounded_p_value = round(p_value, 4)
        fig.suptitle(f"Sig coeff (absolute) p-value: {rounded_p_value}. \n Difference is {are_results_significant}")

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
            ["Base model", "Encompassing", "consistently sig"],
            [og_moel_significant_coeffs, all_predictors_significant, num_sig],
        )
        # rotate the x labels
        plt.xticks(rotation=10)
        ax.set_ylabel("Number of significant coefficients")
        ax.set_title(f" {num_sig} / {len(original_model_pvalues)} coeffs are significant under both models")
        plt.savefig(self.save_path / "proportion_of_significant_coeffs.png")

    def plot_adjusted_r2_scores_for_hierarchy(self, hirarchical_results: dict):
        """Plot the adjusted R2 scores for the hierarchical regression analysis"""

        adjusted_r2_scores = hirarchical_results.values()
        x = np.arange(len(adjusted_r2_scores))
        plt.figure(figsize=(10, 6))
        plt.bar(x, adjusted_r2_scores)
        plt.xlabel("Model Complexity")
        plt.ylabel("Adjusted R2 Score")
        plt.title("Adjusted R2 Score for Hierarchical Regression Analysis")
        plt.xticks(x, ["s0", "s1", "s2", "s3", "s4", "s5", "full_set"])
        plt.grid(True)
        plt.savefig(self.save_path / "adjusted_r2_scores.png")
        plt.close()

    # -------------------- Shifted data analysis --------------------

    def run_all_shifts(self):
        """As a quick pass, shift the neural data backwards and forwards and see how the R2 score changes with each shift"""

        shifted_r2_ols = []
        og_r2 = None
        shifts = np.arange(-400, 400, 10)  # Shift forwards and backwards by 400 frames (40 frames is 1 second)
        for shift_amount in tqdm(shifts, desc="Processing shifts"):
            if shift_amount == 0:
                shifted_design_matrix = self.design_matrix
            else:
                shifted_design_matrix = self.shift_spikes(self.design_matrix, shift_amount)
            r2, _, _ = self.run_just_one_dependent_variable("index_south", shifted_design_matrix)
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

# Note - This is a hard coded example to test the classes
# Not maintained
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

    save_pathh = Path(r"E:\efizz\JAL006\JAL006_shelter_barrier_flip_6_2024_03_28T10_54_20\processed_data\models\single_trial")

    file = r"E:\efizz\JAL006\JAL006_shelter_barrier_flip_6_2024_03_28T10_54_20\processed_data\fully_processed_tracking_data.pickle"
    with open(file, "rb") as r:
        tracking_data = pickle.load(r)

    velocity_data = tracking_data["avg_Velocity"]

    # Run the pipeline
    pp = PreprocessSingleTrialRegression(
        video_df=video_df,
        homings_obj=homings,
        video_and_spike_data=video_and_spike_data,
        frame_by_cluster_matrix=frame_by_cluster_matrix,
        save_path=save_pathh,
        velocity_data=velocity_data,
        similar_homings=False,
        orthogonalise_index=False,
    )
    SingleTrialRegression(
        design_matrix=pp.design_matrix,
        save_path=save_pathh,
        dependents_df=pp.targets_df,
    )
