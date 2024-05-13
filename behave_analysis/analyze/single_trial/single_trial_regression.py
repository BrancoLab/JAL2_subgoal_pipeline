from pathlib import Path

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import dill as pickle
import polars as pl


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
    def _check_index_is_valid():
        """Check if the index is valid"""
        raise NotImplementedError


class PreprocessSingleTrialRegression:

    def __init__(self, video_df, homings_obj, video_and_spike_data, frame_by_cluster_matrix, save_path, similar_homings=False):
        self.homings_obj = homings_obj
        self.video_and_spike_data = video_and_spike_data
        self.frame_by_cluster_matrix = frame_by_cluster_matrix
        self.save_path = save_path
        self.video_df = self.remove_columns_from_video_df(video_df)
        self.similar_homings = similar_homings

        # Unit tests
        UnitTests.check_attributes_of_homing_dic(self.homings_obj)

        # Descriptive plots
        self.plot_homing_durations()
        self.plot_y_coords_distribution()

        # Preprocessing homing data
        self.homing_data_single_dataframe = self.preprocess_homing_data()

    # ------- Descriptive exploratory plots based on preprocessing -------------------------

    def plot_homing_durations(self) -> None:
        """Plotting and saving the homing durations"""
        durations = (self.homings_obj.offset_frames - self.homings_obj.onset_frames) / 40  # Hard coded 40 Hz for fps
        plt.hist(durations)
        plt.xlabel("Duration (s)")
        plt.ylabel("Number of homings")
        plt.title("Homing durations")
        plt.savefig(self.save_path / "homing_durations.png")

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

    def _plot_the_index_per_homing(self):
        """Plotting and save the index per homing"""
        raise NotImplementedError

    def _plot_the_index_distribution(self):
        """Plotting and save the index distribution"""
        raise NotImplementedError

    def create_descriptive_plots(self):
        """Creating the descriptive plots"""
        raise NotImplementedError

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

    def select_similar_homings(self) -> dict:
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

        for idx, homing in enumerate(self.homing_info):

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

    def add_homing_id_to_homing_data(self):
        """Adding the homing id to the homing data. This is needed for the group cross validation object"""

        for idx, homing in enumerate(self.homing_info):
            updated_homing = homing.with_column(pl.lit(idx).alias("homing_id"))
            self.homing_info[idx] = updated_homing

        return self.homing_info

    def concatenate_the_homing_data(self) -> pl.DataFrame:
        """Concatenating the homing data"""
        for idx, homing in enumerate(self.homing_info):
            if idx == 0:
                homing_data = homing
            else:
                homing_data = homing_data.vstack(homing)
        return homing_data

    def preprocess_homing_data(self) -> pl.DataFrame:
        """Preprocessing the data into a single dataframe for regression analysis"""
        self.homing_info = self.extract_data_from_homings()
        if self.similar_homings:
            self.homing_info = self.select_similar_homings()
        self.homing_info = self.add_homing_id_to_homing_data()
        self.homing_data_single_dataframe = self.concatenate_the_homing_data()
        return self.homing_data_single_dataframe

    def create_the_design_matrix(self):
        """Creating the design matrix"""
        raise NotImplementedError

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

    def compute_predictor(self, angle1: np.ndarray, angle2: np.ndarray) -> np.ndarray:
        """Compute a normalised predictor between -1 and 1 between two angles:

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

    def add_dependent_variable_to_data(self):
        """Adding the dependent variable to the data"""
        raise NotImplementedError


class SingleTrialRegression:

    def __init__(self, cross_val_groups):
        self.cross_val_groups = cross_val_groups

    def ols_regression(self):
        """Performing the OLS regression"""
        raise NotImplementedError

    def svr_regression(self):
        """Performing the SVR regression"""
        raise NotImplementedError


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

    pp = PreprocessSingleTrialRegression(
        video_df=video_df,
        homings_obj=homings,
        video_and_spike_data=video_and_spike_data,
        frame_by_cluster_matrix=frame_by_cluster_matrix,
        save_path=save_path,
        similar_homings=False,
    )
