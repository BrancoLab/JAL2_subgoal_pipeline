import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
import matplotlib.pyplot as plt
from pathlib import Path

from behave_analysis.utils.creating_directories import make_directory
from behave_analysis.analyze.single_trial.tests import UnitTests
from behave_analysis.analyze.filtering_data.filtering_functions import discover_condition_based_on_video_df


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
             - homing_list (list): A list of homing dataframes for each homing period
             - condition_per_homing (list): A list of conditions for each homing period
    """

    def __init__(
        self,
        video_df: pl.DataFrame,
        homings_obj: dict,
        video_and_spike_data: pl.DataFrame,
        frame_by_cluster_matrix: np.ndarray,
        save_path: Path,
        velocity_data: np.ndarray,
        barrier_location,
        similar_homings=False,
    ):
        logger.info("Initializing the single trial regression preprocessing object")
        self.homings_obj = homings_obj
        self.video_and_spike_data = video_and_spike_data
        self.frame_by_cluster_matrix = frame_by_cluster_matrix
        self.save_path = save_path
        self.video_df = self.remove_columns_from_video_df(video_df)
        self.barrier_location = barrier_location

        # Settings
        self.similar_homings = similar_homings

        # Preprocessing homing data
        UnitTests.check_attributes_of_homing_dic(self.homings_obj)
        self.homing_list, homing_df_s1, self.condition_per_homing = self.preprocess_homing_data(select_similar_homings=self.similar_homings)
        self.initial_directions = self.label_each_homing_with_an_initial_direction(self.extract_cumulative_homing_data(self.homings_obj, self.barrier_location))

        # Add the dependent variable to the data
        homing_df_s2 = self.add_dependent_index_variable_to_homing_info(homing_data_single_dataframe=homing_df_s1)
        UnitTests.check_index_is_valid(self.compute_index)

        # Add the velocity data to the homing data
        self.homing_data_single_dataframe = self.add_velocity_data_to_homing_data(homing_df_s2, velocity_data)

        # Create the design matrix
        self.design_matrix, self.spike_data_per_homing = self.create_the_design_matrix(self.homing_data_single_dataframe, self.frame_by_cluster_matrix)
        self.targets_df = self.create_dependent_dataframe(self.homing_data_single_dataframe)
        UnitTests.check_the_creation_of_the_design_matrix(self.create_the_design_matrix)

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

    def label_each_homing_with_an_initial_direction(self, cumulative_angle_data: dict) -> list:
        """Label each homing with an initial direction based on the smallest angle to either edge or goal
        
        TODO:
        - refact this to have a threshold as if the angle is between two values then it is ambiguous but this is
        an initial start"""
        
        avg_hsa = cumulative_angle_data["avg_hsa"]
        hdir_north_a = cumulative_angle_data["hdir_north_a"]
        hdir_south_a = cumulative_angle_data["hdir_south_a"]
        initial_direction = []
        
        # For each homing, which goal is the mouse facing the most
        for homing_idx in range(len(avg_hsa)):
            min_absolute_angle = np.min([np.abs(avg_hsa[homing_idx]), np.abs(hdir_north_a[homing_idx]), np.abs(hdir_south_a[homing_idx])])
            if min_absolute_angle == np.abs(avg_hsa[homing_idx]):
                initial_direction.append("shelter")
            if min_absolute_angle == np.abs(hdir_north_a[homing_idx]):
                initial_direction.append("north edge")
            if min_absolute_angle == np.abs(hdir_south_a[homing_idx]):
                initial_direction.append("south edge")

        return initial_direction

    # ----------------- Functions for extracting the homing data ---------------------

    def check_which_barrier_location_is_which_orientation(self, barrier_location) -> tuple:
        """Given the barrier location edges change in tracking data, check which orientation is north and south"""
        if barrier_location[0][0] > 512: # If the first edge x coordinate is greater than 512 then the fistr index is north edge
            return "north", "south"
        else: # Otherwise the first index is south edge
            return "south", "north"
    
    def extract_cumulative_homing_data(self, homing_object: dict, barrier_location) -> dict:
        """Extract the cumulative homing angle data for each homing period. I.e, for 
        each homing, what was the average homing angle to the north and south edge of the barrier.
        
        Args:
            homing_object (dict): The homings object
            barrier_location (np.ndarray): The barrier location from the tracking data
        
        Returns (dict):
            hdir_north_a (np.ndarray): The cumulative homing angle data for the north edge
            hdir_south_a (np.ndarray): The cumulative homing angle data for the south edge
            avg_hsa (np.ndarray): The cumulative homing angle data for the hsa"""
        edge_names = self.check_which_barrier_location_is_which_orientation(barrier_location)
        angle_data = homing_object.homing_angles_dic
        if edge_names[0] == "north":
            hdir_north_a = angle_data["avg_hdir_bar_goal1"]
            hdir_south_a = angle_data["avg_hdir_bar_goal2"]
        elif edge_names[0] == "south":
            hdir_north_a = angle_data["avg_hdir_bar_goal2"]
            hdir_south_a = angle_data["avg_hdir_bar_goal1"]
        avg_hsa = angle_data["avg_hsa"]
        dic = {"hdir_north_a": hdir_north_a, "hdir_south_a": hdir_south_a, "avg_hsa": avg_hsa}
        return dic

    def extract_data_from_homings(self, homing_object: dict, video_df: pl.DataFrame) -> list:
        """Extract the associated behavioural data between homing onsets and offsets.

        Returns:
             (list) of homing dataframes for each homing period"""

        assert UnitTests.check_frame_indexes_are_incremental(video_df["frames"].to_numpy()), "Frames are missing in the homing information"
        assert video_df["frames"].to_numpy()[0] == 1, "The frames do not start at 1"

        homing_info = []
        condition_per_homing = []
        for onset, offset in zip(homing_object.onset_frames, homing_object.offset_frames):
            homing = video_df[int(onset) - 1 : int(offset) - 1]  # Substract 1 to prevent off by one error
            condition = discover_condition_based_on_video_df(homing)
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
            condition_per_homing.append(condition)

        for homing in homing_info:
            assert UnitTests.check_frame_indexes_are_incremental(homing["frames"].to_numpy()), "Frames are missing in the homing information"

        return homing_info, condition_per_homing

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

    def preprocess_homing_data(self, select_similar_homings) -> tuple:
        """Preprocessing the data into a single dataframe for regression analysis
        
        Returns:
            (tuple) of homing_info and concatenated homing data
                -- homing_info (list): A list of homing dataframes for each homing period
                -- concatenated_homing_data (pl.DataFrame): The concatenated homing data ready for regression analysis"""
        extracted_homing_info, condition_per_homing = self.extract_data_from_homings(homing_object=self.homings_obj, video_df=self.video_df)
        if select_similar_homings:
            self.homing_info = self.select_similar_homings(extracted_homing_info)
            extracted_homing_info = self.homing_info
        homing_info = self.add_homing_id_to_homing_data(extracted_homing_info)
        cocatenated_homing_data = self.concatenate_the_homing_data(homing_info)
        return homing_info, cocatenated_homing_data, condition_per_homing

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

    def create_the_design_matrix(self, homing_data: pl.DataFrame, frame_by_cluster_matrix: np.ndarray, normalisation = True) -> np.ndarray:
        """Creating the design matrix of shape (n_frames, n_neurons) for the regression analysis.
        
        Remembering that the each homing has been concatenated together so now we need to index the corresponding
        spike data for each homing period. NOTE: The homing id is added to the design matrix for the group cross validation
        and must be removed before running the regression analysis.
        
        Args:
            data (pl.DataFrame): The homing data
            frame_by_cluster_matrix (np.ndarray): The frame by cluster matrix with smoothed spike counts in each cell
            normalisation (bool, optional): Whether to normalise the design matrix. Defaults to True - needed for testing
            
        returns:
            (pd.DataFrame) The design matrix with the homing id added as a column
            (list) A list of spike data for each homing period"""

        # Initialising the design matrix
        data = homing_data
        total_frames = len(data)
        total_features = frame_by_cluster_matrix.shape[1] # The number of neurons
        design_matrix = np.zeros((total_frames, total_features)) # (F, N)
        spike_data_per_homing = []

        counter = 0
        homing_ids_len = len(np.unique(data["homing_id"]))
        for idx in range(homing_ids_len):

            # Get the frames for the homing id for slicing
            frames = data.filter(data["homing_id"] == idx)["frames"].to_numpy()

            # Get the corresponding frame by cluster matrix
            # minus 1 to prevent off by one error, +1 to include the last frame
            spike_data = frame_by_cluster_matrix[frames[0] - 1 : frames[-1] -1 + 1] # left the -1 + 1 in the second index to make it more readable
            spike_data_per_homing.append(spike_data)

            # Add the spike data to the design matrix
            design_matrix[counter : counter + len(spike_data)] = spike_data
            counter += len(spike_data)

        if normalisation:
            design_matrix = self.normalise_design_matrix(design_matrix)

        if type(design_matrix) != pd.DataFrame:
            design_matrix = pd.DataFrame(design_matrix)

        assert design_matrix.shape == (len(homing_data), frame_by_cluster_matrix.shape[1]), "Design matrix shape is incorrect."

        design_matrix["homing_id"] = data["homing_id"].to_numpy()

        return design_matrix, spike_data_per_homing
