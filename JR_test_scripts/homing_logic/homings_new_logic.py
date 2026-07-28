"""Automatic homing detection with an explicit state-machine segmentation pass.

This module keeps the original Homings dataclass and downstream property extraction,
but replaces the old threshold-smoothed frame logic with a copy that is intended to
be iterated on independently from the legacy implementation.
"""

import os
from dataclasses import dataclass
from typing import Any

import dill as pickle
import numpy as np
import pandas as pd
import polars as pl
from astropy.stats import circmean
from loguru import logger
from scipy.ndimage import gaussian_filter1d

from behave_analysis.analyze.behaviour.homings_escapes.homings import (
    Homings,
    get_avg_homing_angle_for_first15cm_of_run,
    get_avg_speed,
    get_condition_homing,
    get_start_and_end_locs,
)
from behave_analysis.analyze.behaviour.spatial_efficiency import spatial_efficiency
from behave_analysis.utils.creating_directories import make_directory
from behave_analysis.utils.polar_cartesian_projections import negative_radians_to_positive
from behave_analysis.visualize.visualize_utils import open_tracking_data

def _wrap_to_pi(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2 * np.pi) - np.pi


class get_Homings:
    """Extract homing metrics from a session using explicit segmentation logic."""

    def __init__(self, settings, session, video_df=[]):
        self.settings = settings
        self.session = session
        self.use_boris = False
        self.diagnostics: list[dict[str, Any]] = []

        if self.settings.homings_use_boris == True:
            # if we want to use manula laelling chck that the data is present
            boris_path = os.path.join(self.session.base_path, self.session.processed_path, "Borris", "scored_homings.csv")
            if os.path.isfile(boris_path):
                self.use_boris = True
            else:
                logger.warning("You want to use Borris homing labelling, but Borris file doesn't exist! Automatically detecting homings instead")

        self.get_reference_variables()
        self.tracking_data = open_tracking_data(self.session)

        if len(video_df) == 0:
            try:
                self.video_df = pl.read_csv(os.path.join(self.session.base_path, self.session.processed_path, "full_video_dataframe.csv"))
            except FileNotFoundError:
                logger.error("Video df not found, homings will not be computed")
                self.video_df = pl.DataFrame()
        else:
            self.video_df = video_df

    def get_homings(self):
        if self.use_boris:
            logger.info("Using manually labelled homings")
            self.onset_frames, self.stimulus_durations, self.offset_frames = self.load_manual_labels()
        else:
            logger.info("Extracting homings automatically with new logic...")
            self.identify_homing_runs_with_logic()

        self.get_homing_properties()

        self.homing = Homings(
            onset_frames=self.onset_frames,
            offset_frames=self.offset_frames,
            stimulus_durations=self.stimulus_durations,
            start_locs=self.start_locs,
            end_locs=self.end_locs,
            avg_speed=self.avg_speed,
            head_orientation_dic=self.homing_angles_dic,
            hdir_at_start=self.hdir_at_start,
            spatial_efficiency=self.spatial_efficiency_values,
            trajectory_length=self.trajectory_length,
            condition=self.condition,
        )

        self.save_session()
        return self.homing

    def get_reference_variables(self):
        if len(self.session.barrier_time) > 0:
            self.barrier_location = self.session.barrier_location
        else:
            self.barrier_location = [[800, 512], [224, 512], [512, 512]]

        shelter_location = [
            int(np.mean([self.session.shelter_location[0][0], self.session.shelter_location[1][0]])),
            int(np.mean([self.session.shelter_location[0][1], self.session.shelter_location[1][1]])),
        ]
        self.reference_locations = shelter_location + self.barrier_location[:-1]

    def get_homing_properties(self):
        self.start_locs, self.end_locs = get_start_and_end_locs(tracking=self.tracking_data, onset_frames=self.onset_frames, offset_frames=self.offset_frames)
        self.avg_speed = get_avg_speed(self.onset_frames, self.offset_frames, self.tracking_data, self.session)
        self.homing_angles_dic, self.hdir_at_start = get_avg_homing_angle_for_first15cm_of_run(
            self.session, self.onset_frames, self.offset_frames, self.tracking_data, self.settings.homing_run_sustained_distance
        )
        self.condition = get_condition_homing(self.video_df, self.onset_frames, self.session)
        self.spatial_efficiency_values, self.trajectory_length = spatial_efficiency(
            self.onset_frames,
            self.stimulus_durations,
            self.session,
            self.settings,
            self.condition,
            self.tracking_data,
            trial_type="Homings",
            plotting=False,
        )

    def load_manual_labels(self):
        """Load manual labels from the BORIS CSV and convert to 0-based indexing."""
        df = pd.read_csv(os.path.join(self.session.base_path, self.session.processed_path, "Borris", "scored_homings.csv"))
        columns_to_keep = ["Time", "Image index", "Behavior type"]
        fdf = df[columns_to_keep]
        time = fdf["Time"].to_numpy()
        diff = np.diff(time)
        assert np.all(diff > 0), "Time is not increasing"
        start = len(fdf[fdf["Behavior type"] == "START"])
        end = len(fdf[fdf["Behavior type"] == "STOP"])
        assert start == end, "Start and end homings are not the same length"
        logger.info("Loaded manual labels")
        logger.info("Number of homings: {}".format(start))
        onsets = fdf[fdf["Behavior type"] == "START"]["Image index"].to_numpy() - 1
        offsets = fdf[fdf["Behavior type"] == "STOP"]["Image index"].to_numpy() - 1
        assert len(onsets) == len(offsets), "Onsets and offsets are not the same length"
        assert np.diff(onsets).all() > 0, "Onsets are not increasing"
        assert np.diff(offsets).all() > 0, "Offsets are not increasing"
        durations = offsets - onsets
        durations = np.array([[x] for x in durations / self.session.video.fps])
        return onsets, durations, offsets

    def identify_homing_runs_with_logic(self):
        """Compute homing bouts using an explicit state-machine segmentation pass."""
        features = self.extract_variables()
        segments = self.find_homing_segments(features)

        if len(segments) == 0:
            logger.warning("No homing segments detected with new logic")
            self.onset_frames = np.array([], dtype=int)
            self.offset_frames = np.array([], dtype=int)
            self.stimulus_durations = np.array([], dtype=float).reshape(0, 1)
            return

        self.onset_frames = np.asarray([segment["onset"] for segment in segments], dtype=int)
        self.offset_frames = np.asarray([segment["offset"] for segment in segments], dtype=int)
        self.stimulus_durations = np.asarray([[segment["duration_s"]] for segment in segments], dtype=float)
        self.diagnostics = [segment["diagnostic"] for segment in segments]

    def extract_variables(self):
        return {
            "speed": self.get_homing_speed(),
            "speed_y": self.get_speed_along_y_axis(),
            "angles": self.get_homing_relevant_angles(self.video_df),
        }

    def get_homing_speed(self) -> np.ndarray:
        speed = np.asarray(self.tracking_data["avg_Velocity"])
        if len(speed) == 0:
            return speed
        smoothed = gaussian_filter1d(speed, sigma=max(self.session.video.fps / 10, 1), mode="nearest")
        if np.max(smoothed) > 120:
            logger.info("Homing speed is too high, check tracking data")
        return smoothed

    def get_speed_along_y_axis(self) -> np.ndarray:
        speed_y_pixel_per_frame = np.diff(self.tracking_data["avg_loc"][:, 1], axis=0)
        speed_y_cm_per_sec = speed_y_pixel_per_frame * self.session.video.fps / self.session.video.pixels_per_cm
        smoothed_speed_y_cm_per_sec = gaussian_filter1d(speed_y_cm_per_sec, sigma=max(self.session.video.fps / 10, 1), mode="nearest")
        speed_along_y_axis = np.concatenate((np.zeros(1), smoothed_speed_y_cm_per_sec))
        return speed_along_y_axis

    def find_homing_segments(self, features: dict[str, Any]) -> list[dict[str, Any]]:
        """Run-first segmentation: find sustained downward run bouts, then look back for onset.

        This avoids the high miss-rate caused by requiring an onset-side triple-AND gate
        (turn_fast & turn_positive & goal_alignment) which fails during the turn itself
        because the head is still rotating toward the goal and is not yet aligned.
        """
        speed = features["speed"]
        speed_y = features["speed_y"]
        angles = features["angles"]

        fps = self.session.video.fps
        pause_frames = max(1, int(round(self.settings.homing_max_pause_duration * fps)))
        run_min_frames = max(1, int(round(self.settings.homing_run_sustained_duration * fps)))
        lookback_frames = max(1, int(round(self.settings.homing_turn_to_run_window * fps)))
        min_gap = max(1, int(self.settings.homing_min_frames_between_trials))

        # Per-frame signals
        turn_fast, turn_positive = self.find_frames_were_head_turns_fast_to_shelter_or_edge(angles)
        any_goal_turn = turn_fast & turn_positive  # fast turn toward any goal
        goal_alignment = self._goal_alignment_mask(angles)
        run_mask = (speed >= self.settings.homing_run_sustained_speed_threshold) & (speed_y > 0)

        # Step 1: find all downward-run bouts (split on y-direction reversals)
        bouts = self._find_run_bouts(speed, speed_y, run_mask, pause_frames, run_min_frames)

        segments = []
        used_end = -1  # last accepted segment end (+ gap)

        for bout_start, bout_end in bouts:
            if bout_start <= used_end:
                continue

            # Step 2: look back for head turn to set onset
            lb_start = max(0, bout_start - lookback_frames)
            turns_in_window = np.where(any_goal_turn[lb_start:bout_start])[0]
            if len(turns_in_window) > 0:
                onset = lb_start + int(turns_in_window[-1])  # most recent goal-directed turn
            else:
                onset = bout_start  # no turn found; use run start as onset

            offset = bout_end

            # Step 3: validate
            diag = self._segment_diagnostic(onset, offset, speed, goal_alignment)
            if diag["accepted"]:
                segments.append(
                    {
                        "onset": onset,
                        "offset": offset,
                        "duration_s": (offset - onset + 1) / fps,
                        "diagnostic": diag,
                    }
                )
                used_end = offset + min_gap
            else:
                self.diagnostics.append(diag)

        return segments

    def _has_stationary_prelude(self, stationary_mask: np.ndarray, index: int, stationary_frames: int) -> bool:
        if index < stationary_frames:
            return False
        return bool(np.all(stationary_mask[index - stationary_frames : index]))

    def _find_next_true(self, mask: np.ndarray, start: int, stop: int) -> int | None:
        stop = min(stop, len(mask))
        hits = np.where(mask[start:stop])[0]
        if len(hits) == 0:
            return None
        return int(start + hits[0])

    def _find_run_bouts(
        self,
        speed: np.ndarray,
        speed_y: np.ndarray,
        run_mask: np.ndarray,
        pause_frames: int,
        run_min_frames: int,
    ) -> list[tuple[int, int]]:
        """Find continuous downward-run bouts, splitting when the mouse reverses upward.

        A bout starts when run_mask becomes True and ends when:
        - the mouse pauses for > pause_frames (brief stop → allowed)
        - OR the mouse moves upward at > run_sustained_speed_threshold for > pause_frames
          (direction reversal → split)

        Only bouts at least run_min_frames long are returned.
        """
        bouts = []
        n = len(run_mask)
        i = 0

        while i < n:
            if not run_mask[i]:
                i += 1
                continue

            bout_start = i
            last_run = i
            pause_count = 0
            reversal_count = 0

            j = i + 1
            while j < n:
                if run_mask[j]:
                    # Actively running downward — reset counters
                    last_run = j
                    pause_count = 0
                    reversal_count = 0
                elif speed_y[j] < -self.settings.homing_max_allowed_reversal_speed:
                    # Moving upward at sustained speed — direction reversal
                    reversal_count += 1
                    pause_count = 0
                    break  # split here
                else:
                    # Slow or sideways — brief pause
                    pause_count += 1
                    reversal_count = 0
                    if pause_count > pause_frames:
                        break  # real stop — end bout
                j += 1

            bout_end = last_run

            if (bout_end - bout_start + 1) >= run_min_frames:
                bouts.append((bout_start, bout_end))

            i = bout_end + 1

        return bouts

    def _segment_diagnostic(self, onset: int, offset: int, speed: np.ndarray, goal_alignment: np.ndarray) -> dict[str, Any]:
        """Validate a candidate homing segment and return a diagnostic dict.

        Checks (all must pass):
        - minimum duration
        - peak speed above threshold
        - net downward y displacement >= min_net_y_displacement
        - not too sideways (|dx_net| / |dy_net| <= max_sideways_ratio)
        - mouse does not go above its start position by more than max_upward_excursion_cm
        - goal alignment fraction over the whole segment >= min_goal_alignment_fraction
        """
        speed_seg = speed[onset : offset + 1]
        segment_y = self.tracking_data["avg_loc"][onset : offset + 1, 1]
        segment_x = self.tracking_data["avg_loc"][onset : offset + 1, 0]
        ppc = self.session.video.pixels_per_cm
        fps = self.session.video.fps

        dx_cm = (segment_x[-1] - segment_x[0]) / ppc
        dy_cm = (segment_y[-1] - segment_y[0]) / ppc
        # max upward excursion: how much the mouse's y ever dropped below its start
        # (y increases toward shelter, so y < y[0] means mouse went back up)
        max_upward_excursion_cm = float(max(0.0, (segment_y[0] - float(np.min(segment_y))) / ppc))
        path_dx_dy = max(abs(dy_cm), 1e-9)
        sideways_ratio = abs(dx_cm) / path_dx_dy
        peak_speed = float(np.max(speed_seg)) if len(speed_seg) else 0.0
        # goal alignment over the whole run window
        mean_goal_alignment = float(np.mean(goal_alignment[onset : offset + 1])) if offset >= onset else 0.0
        cumulative_dist = cum_distance(segment_x, segment_y, ppc)

        accepted = True
        rejection_reason = []
        if cumulative_dist < self.run_sustained_distance:
            accepted = False
            rejection_reason.append("too-short")
        if peak_speed < self.run_peak_speed_threshold:
            accepted = False
            rejection_reason.append("too-slow")
        if dy_cm < self.min_net_y_displacement:
            accepted = False
            rejection_reason.append("insufficient-net-y")
        if sideways_ratio > self.max_sideways_ratio:
            accepted = False
            rejection_reason.append("too-sideways")
        if max_upward_excursion_cm > self.max_upward_excursion_cm:
            accepted = False
            rejection_reason.append("direction-reversal")
        if mean_goal_alignment < self.min_goal_alignment_fraction:
            accepted = False
            rejection_reason.append("goal-misaligned")

        return {
            "accepted": accepted,
            "rejection_reason": rejection_reason,
            "onset": onset,
            "offset": offset,
            "duration_s": duration_s,
            "peak_speed": peak_speed,
            "net_dx_cm": float(dx_cm),
            "net_dy_cm": float(dy_cm),
            "sideways_ratio": float(sideways_ratio),
            "goal_alignment_fraction": mean_goal_alignment,
            "max_upward_excursion_cm": max_upward_excursion_cm,
        }

    def find_frames_were_head_turns_fast_to_shelter_or_edge(self, angular_data_frame):
        """Return frames with fast goal-directed head turns and a positive turn direction."""
        hsa_pos = np.unwrap(angular_data_frame["hsa"].to_numpy())
        h_bar_pre_flip_pos = np.unwrap(angular_data_frame["h_preflipbar_a"].to_numpy())
        h_bar_post_flip_pos = np.unwrap(angular_data_frame["h_postflipbar_a"].to_numpy())

        hsa_turn_speed = -np.diff(hsa_pos) * self.session.video.fps
        h_bar_preflip_a_turn_speed = -np.diff(h_bar_pre_flip_pos) * self.session.video.fps
        h_bar_postflip_a_turn_speed = -np.diff(h_bar_post_flip_pos) * self.session.video.fps

        hsa_turn_speed = gaussian_filter1d(hsa_turn_speed, sigma=max(self.session.video.fps / 10, 1), mode="nearest")
        h_bar_preflip_a_turn_speed = gaussian_filter1d(h_bar_preflip_a_turn_speed, sigma=max(self.session.video.fps / 10, 1), mode="nearest")
        h_bar_postflip_a_turn_speed = gaussian_filter1d(h_bar_postflip_a_turn_speed, sigma=max(self.session.video.fps / 10, 1), mode="nearest")

        hsa_turn_speed = np.concatenate((np.zeros(1), hsa_turn_speed))
        h_bar_preflip_a_turn_speed = np.concatenate((np.zeros(1), h_bar_preflip_a_turn_speed))
        h_bar_postflip_a_turn_speed = np.concatenate((np.zeros(1), h_bar_postflip_a_turn_speed))

        hsa_bool = hsa_turn_speed > self.turn_speed_threshold
        h_bar_preflip_a_bool = h_bar_preflip_a_turn_speed > self.turn_speed_threshold
        h_bar_postflip_a_bool = h_bar_postflip_a_turn_speed > self.turn_speed_threshold
        logical_speed = np.logical_or(np.logical_or(hsa_bool, h_bar_preflip_a_bool), h_bar_postflip_a_bool)

        hsa_pos_bool = hsa_turn_speed > 0
        h_bar_preflip_a_pos_bool = h_bar_preflip_a_turn_speed > 0
        h_bar_postflip_a_pos_bool = h_bar_postflip_a_turn_speed > 0
        pos_logiical_or = np.logical_or(np.logical_or(hsa_pos_bool, h_bar_preflip_a_pos_bool), h_bar_postflip_a_pos_bool)

        return logical_speed, pos_logiical_or

    def _goal_alignment_mask(self, angular_data_frame) -> np.ndarray:
        hsa = _wrap_to_pi(angular_data_frame["hsa"].to_numpy())
        h_preflip = _wrap_to_pi(angular_data_frame["h_preflipbar_a"].to_numpy())
        h_postflip = _wrap_to_pi(angular_data_frame["h_postflipbar_a"].to_numpy())
        best_alignment = np.min(np.vstack((np.abs(hsa), np.abs(h_preflip), np.abs(h_postflip))), axis=0)
        return best_alignment <= self.goal_angle_threshold

    def get_homing_relevant_angles(self, video_df) -> pl.DataFrame:
        return video_df.select(["hsa", "h_preflipbar_a", "h_postflipbar_a"])

    def save_session(self) -> None:
        folder = make_directory(os.path.join(self.session.base_path, self.session.processed_path, "homings_new_logic"))
        file_name = os.path.join(folder, "homings_obj.pkl")
        with open(file_name, "wb") as dill_file:
            pickle.dump(self.homing, dill_file)
        if len(self.diagnostics) > 0:
            diag_file = os.path.join(folder, "homing_diagnostics.pkl")
            with open(diag_file, "wb") as dill_file:
                pickle.dump(self.diagnostics, dill_file)
        logger.success("Homings object pickle saved")


# -------- HOMING FEATURE FUNCTIONS --------------
# Reused from the legacy module for downstream compatibility.

def cum_distance(x, y, pixels_per_cm) -> float:
    """Returns the frame when the cumulative distance travelled by the mouse in cm hits the threshold

    Returns:
    -- i: int, the index of the frame where the mouse has travelled cum_threshold cm
    """
    for i, (x_i, y_i) in enumerate(zip(x, y)):
        if i == 0:
            cum_dist = 0
        elif i > 0:
            x_diff = x_i - x[i - 1]
            y_diff = y_i - y[i - 1]
            dist = np.sqrt(x_diff**2 + y_diff**2) / pixels_per_cm
            cum_dist += dist

    return cum_dist