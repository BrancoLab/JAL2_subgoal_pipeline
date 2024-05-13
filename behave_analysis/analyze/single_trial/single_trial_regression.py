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

    def __init__(self, video_df, homings_obj, video_and_spike_data, frame_by_cluster_matrix, save_path, similar_homings=False):
        self.homings_obj = homings_obj
        self.video_and_spike_data = video_and_spike_data
        self.frame_by_cluster_matrix = frame_by_cluster_matrix
        self.save_path = save_path
        self.video_df = self.remove_columns_from_video_df(video_df)
        self.similar_homings = similar_homings

        # Unit tests
        UnitTests.check_attributes_of_homing_dic(self.homings_obj)
        UnitTests.check_index_is_valid(self.compute_index)

        # Preprocessing homing data
        self.homing_data_single_dataframe = self.preprocess_homing_data(select_similar_homings = self.similar_homings)

        # Add the dependent variable to the data
        self.homing_data_single_dataframe = self.add_dependent_variable_to_data()
        
        # Create the design matrix
        self.design_matrix = self.create_the_design_matrix(self.homing_data_single_dataframe, self.frame_by_cluster_matrix)

        # Descriptive plots
        self.plot_homing_durations()
        self.plot_y_coords_distribution()
        self.plot_the_index_distribution()
        self.plot_the_index_per_homing()

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

            for _, row in homing_pd.iterrows():
                plt.text(x=row["mouse_x_position"] + 50, y=row["mouse_y_position"] + 10, s=str(np.around(row["index"], 1)), color="red", fontsize=6)

            plt.xlim(xs.min() - 1, xs.max() + 1)
            plt.ylim(ys.min() - 1, ys.max() + 1)
            plt.grid(True)
            plt.xlabel("X Coordinate")
            plt.ylabel("Y Coordinate")
            plt.savefig(save_path / f"index_for_homing_{homing_id}.png")
            plt.close()

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

    def add_dependent_variable_to_data(self):
        """Adding the dependent variable to the data"""

        hsa = self.homing_data_single_dataframe["hsa"].to_numpy().copy()
        goal = self.homing_data_single_dataframe["h_bar_south_a"].to_numpy().copy()

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

    def create_the_design_matrix(self, data: pl.DataFrame, frame_by_cluster_matrix: np.ndarray) -> np.ndarray:
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
            spike_data = frame_by_cluster_matrix[frames[0]:frames[-1]+1]
            
            # Add the spike data to the design matrix
            design_matrix[counter:counter+len(spike_data)] = spike_data
            counter += len(spike_data)
            
        # turn the design matrix into a pandas dataframe
        design_matrix = pd.DataFrame(design_matrix)
        
        np.save("refactor_design_matrix.npy", design_matrix.to_numpy())
        
        # horizontally add the homing_id and index to the design matrix
        index = data["index"].to_numpy()
        homing_id = data["homing_id"].to_numpy()
        
        design_matrix["index"] = index
        design_matrix["homing_id"] = homing_id
                
        return design_matrix

class SingleTrialRegression:
    """A class that performs single trial regression analysis on the data"""

    def __init__(self, design_matrix: pd.DataFrame, save_path: Path):
        self.design_matrix = design_matrix
        self.save_path = save_path
        
        make_directory(self.save_path / "ols_regression")
        make_directory(self.save_path / "svr_regression")
        
        # Run all the regression models
        self.main_execution(self.design_matrix)
        
    def main_execution(self, design_matrix: pd.DataFrame):
                
        # Convert Polars DataFrame to NumPy for compatibility with sklearn
        df = design_matrix
        columns_to_drop = ['index', 'homing_id']
        
        X = df.drop(columns = columns_to_drop).to_numpy()  # Feature matrix
        y = df['index'].to_numpy()  # Target variable
        groups = df['homing_id'].to_numpy()  # Grouping according to homing_id
        
        # overfit
        # reg = LinearRegression()
        # reg.fit(X, y)
        # r2 = reg.score(X, y)
        # print(f"Overfit R2 for OLS score: {np.around(r2, 2)}")
        
        # # Set up GroupKFold
        group_kfold = GroupKFold(n_splits=4)
        assert len(np.unique(groups)) == len(np.unique(df['homing_id'])), "Number of groups is not equal to the number of homing ids"
        
        # storage
        ols_test_r2 = []
        svr_test_r2 = []
        
        # Split using GroupKFold, ensuring groups do not overlap between folds
        for fold, (train_index, test_index) in enumerate(group_kfold.split(X, y, groups)):
            print(f"Fold {fold}:")
            print(f"  Train Indices: {train_index}, Train Groups: {groups[train_index]}")
            print(f"  Test Indices: {test_index}, Test Groups: {groups[test_index]}")
            
            # Extract train and test data based on GroupKFold indices
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            ols_test_r2.append(self.ols_regression(X_train, y_train, fold, self.save_path, X_test, y_test))
            svr_test_r2.append(self.svr_regression(X_train, y_train, fold, self.save_path, X_test, y_test))
            
        print(f"Mean ols test R2 score: {np.around(np.mean(ols_test_r2), 2)}")
        print(f"Mean svr test R2 score: {np.around(np.mean(svr_test_r2), 2)}")
        
    def ols_regression(self, X_train, y_train, fold, save_path: Path, X_test, y_test):
        """Performing the OLS regression"""
        
        # Create a linear regression model
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        
        # Predict and calculate R2 for training data
        train_pred = reg.predict(X_train)
        train_r2 = reg.score(X_train, y_train)
        print(f"Fold {fold + 1} Training R2 score: {np.around(train_r2, 2)}")
        
        # Plot training predictions
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.plot(np.arange(len(y_train)), y_train, label="True")
        plt.plot(np.arange(len(train_pred)), train_pred, label="Predicted")
        plt.legend()
        plt.title(f"Fold {fold + 1} Train Data")
   
        
        # Predict and calculate R2 for test data
        test_pred = reg.predict(X_test)
        test_r2 = reg.score(X_test, y_test)
        print(f"Fold {fold + 1} Testing R2 score: {np.around(test_r2, 2)}")
        
        # Plot test predictions
        plt.subplot(122)
        plt.plot(np.arange(len(y_test)), y_test, label="True")
        plt.plot(np.arange(len(test_pred)), test_pred, label="Predicted")
        plt.legend()
        plt.title(f"Fold {fold + 1} Test Data")
        plt.savefig(save_path / "ols_regression" / f"fold_{fold + 1}_train_vs_test.png")
        plt.close()
        
        return test_r2
        
    def svr_regression(self, X_train, y_train, fold, save_path: Path, X_test, y_test):
        """Performing the SVR regression"""
        
        svr_model = SVR(kernel='rbf', C=0.8, epsilon=0.05)  # You can tune these parameters
        svr_model.fit(X_train, y_train)
        
        # Predict and calculate R2 for training data
        train_pred = svr_model.predict(X_train)
        train_r2 = svr_model.score(X_train, y_train)
        print(f"Svr Fold {fold + 1} Training R2 score: {np.around(train_r2, 2)}")
        
        # test data
        test_pred = svr_model.predict(X_test)
        test_r2 = svr_model.score(X_test, y_test)
        print(f"Svr Fold {fold + 1} Testing R2 score: {np.around(test_r2, 2)}")
        
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
        plt.savefig(save_path / "svr_regression" / f"fold_{fold + 1}_train_vs_test.png")
        
        return test_r2

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
    
    print(pp.design_matrix)
    
    SingleTrialRegression(design_matrix=pp.design_matrix, save_path=save_path)
