import polars as pl
import numpy as np
from loguru import logger

class UnitTests:
    """Unit tests for the SingleTrialRegression class"""

    @staticmethod
    def check_index_values_are_valid(south_index: np.ndarray, north_index: np.ndarray):
        """Check that the index values are valid"""
        assert np.all(south_index >= -1) and np.all(south_index <= 1), "South Index values are not within the range of -1 and 1"
        assert np.all(north_index >= -1) and np.all(north_index <= 1), "North Index values are not within the range of -1 and 1"

    @staticmethod
    def check_angles_are_between_minus_pi_and_pi(hsa: np.array, south_goal: np.array, north_goal: np.array):
        """Check that the angles are between -pi and pi"""
        assert np.all(hsa >= -np.pi) and np.all(hsa <= np.pi), "hsa values are not within the range of -pi and pi"
        assert np.all(south_goal >= -np.pi) and np.all(south_goal <= np.pi), "h_bar_south_a values are not within the range of -pi and pi"
        assert np.all(north_goal >= -np.pi) and np.all(north_goal <= np.pi), "h_bar_north_a values are not within the range of -pi and pi"

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
    def check_for_nans_and_inf(X_train, Y_train):
        # Check for NaNs
        if np.any(np.isnan(X_train)) or np.any(np.isnan(Y_train)):
            raise ValueError("NaNs in the training data")

        # Check for infinite values
        if np.any(np.isinf(X_train)) or np.any(np.isinf(Y_train)):
            raise ValueError("Infinite values in the training data")
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

    @staticmethod
    def check_the_creation_of_the_design_matrix(func):
        """Check the creation of the design matrix"""

        # Create a tiny fake homing_data DataFrame
        data = {
            "homing_id": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "frames": [1, 2, 3, 4, 5, 6, 7, 8, 9]
        }
        homing_data = pl.DataFrame(data)
        
        # Create a frame by cluster matrix with all zeros apart from 3 rows
        frame_by_cluster_matrix = np.zeros((10, 5))  # 10 frames, 5 neurons
        frame_by_cluster_matrix[1] = [1, 0, 0, 0, 0]  # Second frame
        frame_by_cluster_matrix[4] = [0, 1, 0, 0, 0]  # Fifth frame
        frame_by_cluster_matrix[7] = [0, 0, 1, 0, 0]  # Eighth frame

        design_matrix, _ = func(homing_data, frame_by_cluster_matrix, normalisation=False)
        assert design_matrix.shape == (9, 6), "There should be 9 data points and 5 neurons + homing_id column"
        assert np.array_equal(np.asarray(design_matrix.iloc[0]), [0, 0, 0, 0, 0, 0]), "The first row is incorrect"
        assert np.array_equal(np.asarray(design_matrix.iloc[1]), [1, 0, 0, 0, 0, 0]), "The 2nd row is incorrect"
        assert np.array_equal(np.asarray(design_matrix.iloc[2]), [0, 0, 0, 0, 0, 0]), "The 3rd row is incorrect"
        assert np.array_equal(np.asarray(design_matrix.iloc[4]), [0, 1, 0, 0, 0, 1]), "The 4th row is incorrect"
