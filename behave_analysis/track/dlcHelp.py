# OS libraries
import glob
import pandas as pd
import os
from loguru import logger
import yaml
import numpy as np
import scipy.ndimage

from behave_analysis.database.computer_ID import get_computer_specific_paths


class DLC:
    """
    A class to handle the DLC tracking data. This class is used to extract the tracking data
    from the DLC outputted .h5 file and save it to a dictionary. The class also creates a 3D array
    of tracking data from DLC of length number of frames. The main functions are then to
    process and log poor tracking data.
    """

    def run_deeplabcut_tracking(self, session) -> None:
        """Check if DLC has been run on a video before, if not run analyze videos. If a DLC
        file already exists don't run DLC again. It requires the DLC settings file
        to contain resnet in its name which may not be the case for all DLC models
        and versions.

        Args:
            session (object): A data class containing relevant information for tracking contained within settings_track.py
        """
        dlc_already_run = bool(
            glob.glob(os.path.join(session.base_path, session.processed_path, "*resnet*"))
        )  # Does a file exist with this token in the name?

        _, dlc_settings_file = get_computer_specific_paths()

        if dlc_already_run:
            logger.info("DeepLabCut has already been run for this session: {} - {}".format(session.number, session.name))

        else:
            logger.info("Running DeepLabCut tracking for session: {} - {}".format(session.number, session.name))
            from deeplabcut.pose_estimation_tensorflow import analyze_videos

            video_file = os.path.join(session.base_path, session.file_path, session.video.camFilePath)
            analyze_videos(dlc_settings_file, video_file)
            for files in glob.glob(os.path.join(session.base_path, session.file_path, "*resnet*")):
                os.rename(files, os.path.join(session.base_path, session.processed_path, os.path.basename(files)))
        if self.settings.save_labeled_video:
            from deeplabcut import create_labeled_video

            # move dlc files to folder with avi
            run_name = os.path.split(session.file_path)[1]
            for files in glob.glob(os.path.join(session.base_path,session.processed_path,"*" + run_name + "*")):
                    os.rename(files,os.path.join(session.base_path, session.file_path,os.path.basename(files)))
            # create labeled video
            path = os.path.join(session.base_path,session.file_path,session.video.camFilePath)      
            create_labeled_video(dlc_settings_file, path) # for some ungodly reason these settings don't work: fastmode=True,save_frames=True,keypoints_only=True
            # move dlc files back to processed path
            for files in glob.glob(os.path.join(session.base_path,session.file_path, "*resnet*")):
                    os.rename(files,os.path.join(session.base_path,session.processed_path,os.path.basename(files)))
            
            # video_file = os.path.join(session.base_path,session.file_path,session.video.camFilePath) 
            # create_labeled_video(dlc_settings_file, video_file, save_frames = True, keypoints_only=True)
    
    def create_dlc_tracking_array(self, session) -> None:
        """
        Create and fill an array of tracking data from DLC.
        """
        self.extract_data_from_dlc_file(session)
        self.create_array_with_dlc_tracking_data(session)

        return None

    def extract_data_from_dlc_file(self, session) -> None:
        """
        Ingests a H5 file outputted from DLC analysis, body parts, and
        model name. Changing the string in dlc network name maybe necessary if using different model
        type.

        The function saves the body parts tracked by DLC to the tracking data dictionary.

        Args:
            session (obejct): Session settings object data class
        """

        # Load DLC Tracking Data
        dlc_tracking_file = glob.glob(os.path.join(session.base_path, session.processed_path, "*.h5"))[0]  # Selects the .h5 file in video dir
        self.dlc_output = pd.read_hdf(dlc_tracking_file)  # Converts .h5 to pandas

        # Load DLC Config from settings dataclass
        _, dlc_settings_file = get_computer_specific_paths()
        with open(dlc_settings_file) as file:
            dlc_settings = yaml.safe_load(file)

        self.tracking_data_body_parts = {}  # init dictionary
        self.tracking_data_body_parts["bodyparts"] = dlc_settings["bodyparts"]

        # fix names to make consistent across models
        if "right_hindpaw" in self.tracking_data_body_parts["bodyparts"]:
            self.dlc_output = self.dlc_output.rename(columns={"right_hindpaw": "right_hind_limb"})
            self.tracking_data_body_parts["bodyparts"] = list(
                map(lambda x: x.replace("right_hindpaw", "right_hind_limb"), self.tracking_data_body_parts["bodyparts"])
            )
        if "left_hindpaw" in self.tracking_data_body_parts["bodyparts"]:
            self.dlc_output = self.dlc_output.rename(columns={"left_hindpaw": "left_hind_limb"})
            self.tracking_data_body_parts["bodyparts"] = list(
                map(lambda x: x.replace("left_hindpaw", "left_hind_limb"), self.tracking_data_body_parts["bodyparts"])
            )

        logger.info(f"The bodyparts tracked by DLC are: {self.tracking_data_body_parts['bodyparts']}")

        self.dlc_network_name = dlc_tracking_file[dlc_tracking_file.find("DLC_resnet") : -3]  # This line breaks if different model names are used
        assert self.dlc_network_name, "No DLC name found, has a different model been used?"
        logger.info(f"The DLC network name is: {self.dlc_network_name}")

        return None

    def create_array_with_dlc_tracking_data(self, session) -> None:
        """A function that creates an array of shape (number of frames, number of body parts, 3)
        where the 3 is for x, y, and likelihood. A potential refactor would be to covert into a dictionary
        where there are more clear defined keys e.g. Leave for now.
        
        TODO:
        -- Unsure whether this function will fail if the DLC fails and produces less values than Frames. Implement this check to ensure it does fail
            DLC could fail for example on HPC non critically without us knowing. This could cause issues. 
        """
        self.tracking_data_array = np.zeros((session.video.num_frames, len(self.tracking_data_body_parts["bodyparts"]), 3))

        for i, body_part in enumerate(self.tracking_data_body_parts["bodyparts"]):
            for j, axis in enumerate(["x", "y"]):
                self.tracking_data_array[:, i, j] = self.dlc_output[self.dlc_network_name][body_part][axis].values
            self.tracking_data_array[:, i, 2] = self.dlc_output[self.dlc_network_name][body_part]["likelihood"].values

        return None

    def remove_bad_tracking_data(self, session) -> None:
        """
        A function to remove poor tracking data out of an expected frame window, or to far away
        from the body.
        """
        self.correct_out_of_frame_tracking(session)
        self.log_low_confidence_points()
        self.replace_low_confidence_points_with_nan()
        self.interpolate_nan_values()
        self.apply_median_filter(filter_length=7)
        self.replace_points_far_from_median_bodypart_with_nan()
        self.interpolate_nan_values()

    def correct_out_of_frame_tracking(self, session) -> None:
        self.tracking_data_array[self.tracking_data_array < 0] = 0
        self.tracking_data_array[:, :, 0][self.tracking_data_array[:, :, 0] > (session.video.width - 1)] = session.video.width - 1
        self.tracking_data_array[:, :, 1][self.tracking_data_array[:, :, 1] > (session.video.height - 1)] = session.video.height - 1

    def replace_points_far_from_median_bodypart_with_nan(self) -> None:
        median_position_across_bodyparts = np.nanmedian(self.tracking_data_array[:, :, :2], axis=1)
        distance_from_median_position = (
            (self.tracking_data_array[:, :, 0] - median_position_across_bodyparts[:, 0:1]) ** 2
            + (self.tracking_data_array[:, :, 1] - median_position_across_bodyparts[:, 1:2]) ** 2
        ) ** 0.5
        self.tracking_data_array[distance_from_median_position > self.settings.max_deviation_from_rest_of_points, :2] = np.nan
        
    def replace_low_confidence_points_with_nan(self) -> np.ndarray:
        """Replace low confidence points with NaNs."""
        logger.info("Replacing low confidence points with NaN")
        low_confidence_points = self.tracking_data_array[:, :, 2] < self.settings.min_confidence_in_tracking
        self.tracking_data_array[low_confidence_points, :2] = np.nan
        
    def interpolate_nan_values(self):
        for i, _ in enumerate(self.tracking_data_body_parts['bodyparts']):
            self.tracking_data_array[:, i, :2] = np.array(pd.DataFrame(self.tracking_data_array[:, i, :2]).interpolate().fillna(method='bfill').fillna(method='ffill'))
        
    def log_low_confidence_points(self) -> None:
        """
        Log how many points in DLC are considered low confidence relative to an abitrary
        value set in the settings
        """

        low_confidence_points = self.tracking_data_array[:, :, 2] < self.settings.min_confidence_in_tracking
        numOflowConfidencePoints = np.count_nonzero(low_confidence_points)
        numOfTotalPoints = low_confidence_points.shape[0] * low_confidence_points.shape[1]
        perct = numOflowConfidencePoints / numOfTotalPoints

        logger.warning(
            f"Found {numOflowConfidencePoints} out of {numOfTotalPoints} points ({perct:.2f}) below the confidence threshold of {self.settings.min_confidence_in_tracking}"
        )
        assert perct < 0.35, r"This is too high. Please check your tracking data, retrain DLC."
        
    def apply_median_filter(self, filter_length=7):
        self.tracking_data_array[:, :, :2] = scipy.ndimage.median_filter(self.tracking_data_array[:, :, :2], size=(filter_length, 1, 1), mode='nearest')
