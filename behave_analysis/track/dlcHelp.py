# OS libraries
import glob
import pandas as pd
import os
import json
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
        dlc_already_run = bool(glob.glob(os.path.join(session["base_path"], session["processed_path"], "*resnet*")))  # Does a file exist with this token in the name?
        if not dlc_already_run:
            dlc_already_run = bool(glob.glob(os.path.join(session["base_path"], session["file_path"], "newDLC", "*resnet*")))  # Does a file exist with this token in the name?
            if dlc_already_run:
                for files in glob.glob(os.path.join(session["base_path"], session["file_path"], "newDLC", "*resnet*")):
                    os.rename(files, os.path.join(session["base_path"], session["processed_path"], os.path.basename(files)))

        _, dlc_settings_file = get_computer_specific_paths()

        if dlc_already_run:
            logger.info("DeepLabCut has already been run for this session: {} - {}".format(session["number"], session["name"]))

        else:
            logger.info("Running DeepLabCut tracking for session: {} - {}".format(session["number"], session["name"]))
            from deeplabcut.pose_estimation_tensorflow import analyze_videos

            video_file = os.path.join(session["base_path"], session["file_path"], session["video"]["camFilePath"])
            analyze_videos(dlc_settings_file, video_file)
            for files in glob.glob(os.path.join(session["base_path"], session["file_path"], "*resnet*")):
                os.rename(files, os.path.join(session["base_path"], session["processed_path"], os.path.basename(files)))
        if self.settings.save_labeled_video:
            from deeplabcut import create_labeled_video

            # move dlc files to folder with avi
            run_name = os.path.split(session["file_path"])[1]
            for files in glob.glob(os.path.join(session["base_path"], session["processed_path"], "*" + run_name + "*")):
                os.rename(files, os.path.join(session["base_path"], session["file_path"], os.path.basename(files)))
            # create labeled video
            path = os.path.join(session["base_path"], session["file_path"], session["video"]["camFilePath"])
            create_labeled_video(dlc_settings_file, path)  # for some ungodly reason these settings don't work: fastmode=True,save_frames=True,keypoints_only=True
            # move dlc files back to processed path
            for files in glob.glob(os.path.join(session["base_path"], session["file_path"], "*resnet*")):
                os.rename(files, os.path.join(session["base_path"], session["processed_path"], os.path.basename(files)))

            # video_file = os.path.join(session['base_path'],session["file_path"],session["video"]["camFilePath"])
            # create_labeled_video(dlc_settings_file, video_file, save_frames = True, keypoints_only=True)

    def create_dlc_tracking_array(self, session) -> None:
        """
        Create and fill an array of tracking data from DLC.
        """
        self.extract_data_from_dlc_file(session)
        self.create_array_with_dlc_tracking_data(session)
        self.set_nonvalid_time_to_nan(session)

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
        dlc_tracking_file = glob.glob(os.path.join(session["base_path"], session["processed_path"], "*.h5"))[0]  # Selects the .h5 file in video dir
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
            self.tracking_data_body_parts["bodyparts"] = list(map(lambda x: x.replace("right_hindpaw", "right_hind_limb"), self.tracking_data_body_parts["bodyparts"]))
        if "left_hindpaw" in self.tracking_data_body_parts["bodyparts"]:
            self.dlc_output = self.dlc_output.rename(columns={"left_hindpaw": "left_hind_limb"})
            self.tracking_data_body_parts["bodyparts"] = list(map(lambda x: x.replace("left_hindpaw", "left_hind_limb"), self.tracking_data_body_parts["bodyparts"]))

        logger.info(f"The bodyparts tracked by DLC are: {self.tracking_data_body_parts['bodyparts']}")

        self.dlc_network_name = dlc_tracking_file[dlc_tracking_file.find("DLC_resnet") : -3]  # This line breaks if different model names are used
        assert self.dlc_network_name, "No DLC name found, has a different model been used?"
        logger.info(f"The DLC network name is: {self.dlc_network_name}")

    def create_array_with_dlc_tracking_data(self, session) -> None:
        """A function that creates an array of shape (number of frames, number of body parts, 3)
        where the 3 is for x, y, and likelihood. A potential refactor would be to covert into a dictionary
        where there are more clear defined keys e.g. Leave for now.
        """
        assert (
            self.dlc_output.shape[0] == session["video"]["num_frames"]
        ), "The number of frames in the DLC output does not match the number of frames in the video. This could be due to a failed DLC run or a mismatch in the video file. Please check your DLC output and video file."
        self.tracking_data_array = np.zeros((session["video"]["num_frames"], len(self.tracking_data_body_parts["bodyparts"]), 3))

        for i, body_part in enumerate(self.tracking_data_body_parts["bodyparts"]):
            for j, axis in enumerate(["x", "y"]):
                self.tracking_data_array[:, i, j] = self.dlc_output[self.dlc_network_name][body_part][axis].values
            self.tracking_data_array[:, i, 2] = self.dlc_output[self.dlc_network_name][body_part]["likelihood"].values

    def set_nonvalid_time_to_nan(self, session, data_array=None) -> None:
        """
        A function to set tracking data to nan for frames that are outside of the valid time window.
        This is useful for when the video is longer than the valid time window and we want to ignore
        the tracking data outside of the valid time window.

        Args:
            session (object): Session settings object data class
            data_array (np.ndarray): Optional target array with shape [frames, bodyparts, dims].
                If None, applies masking to self.tracking_data_array.
        """
        target = self.tracking_data_array if data_array is None else data_array
        start = int(session["valid_time"][0] * 60 * session["video"]["fps"])
        target[:start, :, :] = np.nan
        if session["valid_time"][1] != -1:
            end = int(session["valid_time"][1] * 60 * session["video"]["fps"])
            target[end:, :, :] = np.nan

    def remove_bad_tracking_data(self, session) -> None:
        """
        A function to remove poor tracking data out of an expected frame window, or to far away
        from the body.
        """
        self.corrected_tracking_data_array = self.registered_tracking_data_array.astype(float)  # convert to float so we can set nan values
        nan_before_qc = self.compute_nan_fraction_by_bodypart(self.corrected_tracking_data_array)
        self.correct_out_of_arena_tracking(session)
        self.log_low_confidence_points()
        self.replace_low_confidence_points_with_nan()
        jump_qc_first = self.reject_unphysical_jumps(session)
        self.interpolate_nan_values(max_gap_frames=self.settings.interpolation_max_gap_frames)
        self.apply_median_filter(filter_length=7)
        self.replace_points_far_from_median_bodypart_with_nan()
        jump_qc_second = self.reject_unphysical_jumps(session)
        self.interpolate_nan_values(max_gap_frames=self.settings.interpolation_max_gap_frames)
        self.set_nonvalid_time_to_nan(session, data_array=self.corrected_tracking_data_array)
        self.save_qc_report(session, nan_before_qc, jump_qc_first, jump_qc_second)
        self.reshape_for_kalman()

    def correct_out_of_arena_tracking(self, session) -> None:
        """This assumes a circular arena in the center of the video frame. It corrects for tracking points that are outside of the arena by setting them to nan."""
        for i, _ in enumerate(self.tracking_data_body_parts["bodyparts"]):
            x = self.corrected_tracking_data_array[:, i, 0]
            y = self.corrected_tracking_data_array[:, i, 1]
            dist_to_center = ((x - session["video"]["width"] / 2) ** 2 + (y - session["video"]["height"] / 2) ** 2) ** 0.5  # assume circular arena in center of video frame
            self.corrected_tracking_data_array[:, i, :2][dist_to_center > (session["video"]["radius"] + self.settings.max_deviation_from_arena)] = np.nan

    def replace_points_far_from_median_bodypart_with_nan(self) -> None:
        xy = self.corrected_tracking_data_array[:, :, :2]
        # Only compute medians on frames where at least one bodypart has finite XY to avoid all-NaN warnings.
        has_any_valid_bodypart = np.isfinite(xy).all(axis=2).any(axis=1)
        median_position_across_bodyparts = np.full((xy.shape[0], 2), np.nan, dtype=float)
        if np.any(has_any_valid_bodypart):
            median_position_across_bodyparts[has_any_valid_bodypart] = np.nanmedian(xy[has_any_valid_bodypart], axis=1)

        distance_from_median_position = ((xy[:, :, 0] - median_position_across_bodyparts[:, 0:1]) ** 2 + (xy[:, :, 1] - median_position_across_bodyparts[:, 1:2]) ** 2) ** 0.5
        outlier_mask = distance_from_median_position > self.settings.max_deviation_from_rest_of_points
        self.corrected_tracking_data_array[:, :, :2][outlier_mask] = np.nan

    def replace_low_confidence_points_with_nan(self) -> np.ndarray:
        """Replace low confidence points with NaNs."""
        logger.info("Replacing low confidence points with NaN")
        low_confidence_points = self.tracking_data_array[:, :, 2] < self.settings.min_confidence_in_tracking
        self.corrected_tracking_data_array[low_confidence_points, :2] = np.nan

    def reject_unphysical_jumps(self, session) -> dict:
        """Set implausibly large frame-to-frame jumps to NaN for each bodypart independently."""
        max_jump_px = self.settings.jump_reject_max_speed_cm_s * session["video"]["pixels_per_cm"] / session["video"]["fps"]
        expand = max(int(self.settings.jump_reject_expand_frames), 0)
        total_rejected = 0
        per_bodypart = {}

        for i, bodypart in enumerate(self.tracking_data_body_parts["bodyparts"]):
            xy = self.corrected_tracking_data_array[:, i, :2]
            displacement = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
            jump_indices = np.where(np.isfinite(displacement) & (displacement > max_jump_px))[0] + 1
            if len(jump_indices) == 0:
                per_bodypart[bodypart] = 0
                continue

            reject_mask = np.zeros(xy.shape[0], dtype=bool)
            for idx in jump_indices:
                start = max(idx - expand, 0)
                stop = min(idx + expand + 1, xy.shape[0])
                reject_mask[start:stop] = True

            rejected = int(np.count_nonzero(reject_mask))
            total_rejected += rejected
            per_bodypart[bodypart] = rejected
            self.corrected_tracking_data_array[reject_mask, i, :2] = np.nan
            logger.info(f"Rejected {rejected} samples for {bodypart} due to implausible jump speed")

        logger.info(f"Total rejected samples from jump QC: {total_rejected}")
        return {"per_bodypart": per_bodypart, "total": total_rejected}

    def interpolate_nan_values(self, max_gap_frames=None):
        """Interpolate only short internal NaN gaps; leave long/edge gaps as NaN."""
        if max_gap_frames is None:
            max_gap_frames = self.settings.interpolation_max_gap_frames
        if max_gap_frames is None or max_gap_frames <= 0:
            logger.info("Skipping interpolation because interpolation_max_gap_frames <= 0")
            return

        for i, _ in enumerate(self.tracking_data_body_parts["bodyparts"]):
            part_df = pd.DataFrame(self.corrected_tracking_data_array[:, i, :2], columns=["x", "y"])
            interpolated = part_df.interpolate(method="linear", limit=max_gap_frames, limit_direction="both", limit_area="inside")
            self.corrected_tracking_data_array[:, i, :2] = interpolated.to_numpy()

    def compute_nan_fraction_by_bodypart(self, data_array: np.ndarray) -> dict:
        """Return NaN fraction per bodypart from a tracking array with shape [frames, bodyparts, xy]."""
        fractions = {}
        for i, bodypart in enumerate(self.tracking_data_body_parts["bodyparts"]):
            invalid = ~np.isfinite(data_array[:, i, :2]).all(axis=1)
            fractions[bodypart] = float(np.mean(invalid))
        return fractions

    def compute_gap_stats(self) -> dict:
        """Summarize NaN gap counts and lengths per bodypart after final QC."""
        gap_stats = {}
        for i, bodypart in enumerate(self.tracking_data_body_parts["bodyparts"]):
            invalid = ~np.isfinite(self.corrected_tracking_data_array[:, i, :2]).all(axis=1)
            padded = np.concatenate(([False], invalid, [False]))
            changes = np.diff(padded.astype(int))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            lengths = (ends - starts).astype(int)
            gap_stats[bodypart] = {
                "num_gaps": int(len(lengths)),
                "max_gap_frames": int(lengths.max()) if len(lengths) > 0 else 0,
                "mean_gap_frames": float(lengths.mean()) if len(lengths) > 0 else 0.0,
                "total_gap_frames": int(lengths.sum()) if len(lengths) > 0 else 0,
            }
        return gap_stats

    def save_qc_report(self, session, nan_before_qc: dict, jump_qc_first: dict, jump_qc_second: dict) -> None:
        """Save a compact QC summary as JSON in the processed session folder."""
        nan_after_qc = self.compute_nan_fraction_by_bodypart(self.corrected_tracking_data_array)
        gap_stats = self.compute_gap_stats()
        low_confidence_points = self.tracking_data_array[:, :, 2] < self.settings.min_confidence_in_tracking
        low_confidence_fraction = float(np.count_nonzero(low_confidence_points) / low_confidence_points.size)

        report = {
            "session": {
                "number": session["number"],
                "name": session["name"],
            },
            "video": {
                "fps": session["video"]["fps"],
                "pixels_per_cm": session["video"]["pixels_per_cm"],
                "num_frames": session["video"]["num_frames"],
            },
            "settings": {
                "min_confidence_in_tracking": self.settings.min_confidence_in_tracking,
                "max_deviation_from_rest_of_points": self.settings.max_deviation_from_rest_of_points,
                "max_deviation_from_arena": self.settings.max_deviation_from_arena,
                "interpolation_max_gap_frames": self.settings.interpolation_max_gap_frames,
                "jump_reject_max_speed_cm_s": self.settings.jump_reject_max_speed_cm_s,
                "jump_reject_expand_frames": self.settings.jump_reject_expand_frames,
                "use_kalman": self.settings.use_kalman,
            },
            "summary": {
                "low_confidence_fraction": low_confidence_fraction,
                "jump_rejected_first_pass_total": jump_qc_first["total"],
                "jump_rejected_second_pass_total": jump_qc_second["total"],
            },
            "per_bodypart": {
                bodypart: {
                    "nan_fraction_before_qc": nan_before_qc[bodypart],
                    "nan_fraction_after_qc": nan_after_qc[bodypart],
                    "jump_rejected_first_pass": jump_qc_first["per_bodypart"].get(bodypart, 0),
                    "jump_rejected_second_pass": jump_qc_second["per_bodypart"].get(bodypart, 0),
                    **gap_stats[bodypart],
                }
                for bodypart in self.tracking_data_body_parts["bodyparts"]
            },
        }

        report_path = os.path.join(session["base_path"], session["processed_path"], "tracking_qc_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved tracking QC report to {report_path}")

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
        logger.info(f"Applying median filter to tracking data (filter_length={filter_length})")
        self.corrected_tracking_data_array[:, :, :2] = scipy.ndimage.median_filter(self.corrected_tracking_data_array[:, :, :2], size=(filter_length, 1, 1), mode="nearest")

    def reshape_for_kalman(self):
        self.corrected_tracking_data = {}
        for i, bodypart in enumerate(self.tracking_data_body_parts["bodyparts"]):
            self.corrected_tracking_data[bodypart] = self.corrected_tracking_data_array[:, i, :2]
