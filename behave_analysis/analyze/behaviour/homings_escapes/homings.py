"""
The point of this script is to return the class Homings, the attributes of which can
be seen below. Escapes are not removed from homings here currently. 

Note:
-- Upgraded homings 08/07/2024. Seems the last tweak would be to fix when the trajectory terminmates too early
sometimes meaning the homing is split into two. This can be fixed by creating a buffer between homings for now.
There are also anoyying homings that when they run around the edge they meet all the criteria, perhaps these can be
removed by setting a stricter set of criteria at the expense of other homings.
"""

import os
from loguru import logger
import pandas as pd
from astropy.stats import circmean
import numpy as np
import polars as pl
from dataclasses import asdict

from behave_analysis.analyze.results_database_utils import settings_to_check, check_database_for_same_run, add_run_to_database
from behave_analysis.visualize.visualize_utils import open_tracking_data
from behave_analysis.analyze.behaviour.spatial_efficiency import spatial_efficiency
from behave_analysis.analyze.behaviour.homings_escapes.analyze_homing_consolidated import HomingAnalyzer
from behave_analysis.utils.identify_condition import build_shelter_condition_bool, build_barrier_condition_bool, build_flippedbarrier_condition_bool, identify_condition_of_trial

class get_Homings:
    """Extract homings metrics from a session

    Responsible for:
    -- Creates a Homings object using manual labels or logic
    -- Saves the Homings extracted detailed as a dict

    """

    def __init__(self, settings, session):
        self.settings = settings
        self.session = session
        self.savepath = os.path.join(self.session.base_path, self.session.processed_path, "homings")
        logger.info(f"checking for existing homings results")
        if self.settings.homings_use_boris:
            self.database, self.do_analysis, self.hexaname = check_database_for_same_run(
                db_settings={"homings_use_boris": True}, # the only setting that needs to be matched!
                results_csv_name=self.savepath + os.sep + "Homing_database.csv",
                settings=self.settings,
            )
        else:
            self.database, self.do_analysis, self.hexaname = check_database_for_same_run(
                db_settings={**settings_to_check(self.settings, ["homing"])},
                results_csv_name=self.savepath + os.sep + "Homing_database.csv",
                settings=self.settings,
            )

    def get_homings(self, video_df = [], tracking_data = [], return_dict = True):

        if not self.do_analysis:
            logger.info("Homing analysis already done with these settings, loading from database...")
            filename = os.path.join(self.savepath, "homings_" + self.hexaname + "_results.npz")
            homings = np.load(filename, allow_pickle=True)
            return homings

        if self.settings.homings_use_boris:
            boris_path = os.path.join(self.session.base_path, self.session.processed_path) + "\\" + "Borris" + "\\" + "scored_homings.csv"
            if os.path.isfile(boris_path):
                use_boris = True
                logger.info("Using manually labelled homings")
                self.onset_frames, self.stimulus_durations, self.offset_frames = load_manual_labels(self.session)
            else:
                logger.warning("You want to use Borris homing labelling, but Borris file doesn't exist! Automatically detecting homings instead")
                use_boris = False
        
        if self.settings.homings_use_boris == False or use_boris == False:
            # Begin extracting variables for homings
            logger.info("Extracting homings automatically...")
            self.identify_homing_runs_with_logic(video_df = video_df, tracking_data = tracking_data)

        self.get_homing_properties(tracking_data=tracking_data, video_df = video_df)

        results = self.save_session()
        if return_dict:
            return results

    def save_session(self) -> None:
        """Save homings object as a pickle file within the session folder"""
        filename = os.path.join(self.savepath, "homings_" + self.hexaname)
        results_dict = {"onset_frames": self.onset_frames,
                        "offset_frames": self.offset_frames,
                        "stimulus_durations": self.stimulus_durations,
                        "start_locs": self.start_locs,
                        "end_locs": self.end_locs,
                        "avg_speed": self.avg_speed,
                        "head_orientation_dic": self.homing_angles_dic,
                        "hdir_at_start": self.hdir_at_start,
                        "spatial_efficiency": self.spatial_efficiency_values,
                        "trajectory_length": self.trajectory_length,
                        "condition": self.condition}
        np.savez(os.path.join(filename + "_results.npz"), **results_dict, allow_pickle=True)
        settings = asdict(self.settings)
        np.savez(filename + "_settings.npz", **settings, allow_pickle=True)
        add_run_to_database(self.database, 
                            {**settings_to_check(self.settings, ["homing"])}, 
                            self.savepath + os.sep + "Homing_database.csv", 
                            self.hexaname)
        logger.success("Homings saved")
        return results_dict

    # ------------------- SELECT FEATURES OF HOMINGS ----------------------

    def get_homing_properties(self, tracking_data = [], video_df = []):
        """Extract everything we need to know about homies"""
        if len(tracking_data) == 0:
            tracking_data = open_tracking_data(self.session)
        if len(video_df) == 0:
            video_df = pl.Dataframe({
                "frames": np.arange(1, len(tracking_data["hdir"]) + 1).astype(np.int64)
            })

        self.start_locs, self.end_locs = get_start_and_end_locs(
            tracking=tracking_data, onset_frames=self.onset_frames, offset_frames=self.offset_frames
        )
        self.avg_speed = get_avg_speed(self.onset_frames, self.offset_frames, tracking_data, self.session)
        self.homing_angles_dic, self.hdir_at_start = get_avg_homing_angle_for_first15cm_of_run(
            self.session, self.onset_frames, self.offset_frames, tracking_data, self.settings.cum_threshold
        )
        self.condition = get_condition_homing(video_df, self.onset_frames, self.session)
        self.spatial_efficiency_values, self.trajectory_length = spatial_efficiency(
            self.onset_frames, self.stimulus_durations, self.session, self.settings, self.condition, tracking_data, trial_type="Homings", plotting=False
        )

    # ------------------- IDENTIFY HOMINGS --------------------------------

    def identify_homing_runs_with_logic(self, video_df = [], tracking_data = []):
        """All the steps needed to ID homings automatically"""

        if len(tracking_data) == 0:
            tracking_data = open_tracking_data(self.session)

        if len(video_df) == 0:
            video_df = pl.read_csv(os.path.join(self.session.base_path, self.session.processed_path) + "\\" "full_video_dataframe.csv")

        analyzer = HomingAnalyzer([], settings=self.settings)
        analyzer.preloaded_session_data(video_df = video_df, tracking_data = tracking_data, session = self.session)
        analyzer.extract_runs(speed_threshold=self.settings.homings_speed_threshold, gap_tolerance_frames=self.settings.homings_gap_tolerance)
        analyzer._compute_run_features(analyzer.extracted_runs[0])

        candidates = analyzer.run_classification(use_learned_gates=True)
        # cadidates is list of tuple of onsets and offsets, so we can unpack it here
        candidates = np.array(candidates, dtype = int)
        self.onset_frames, self.offset_frames = candidates[:,0], candidates[:,1]
        self.stimulus_durations = (self.offset_frames - self.onset_frames)/self.session.video.fps  # match the format of the manual labels

##-------- HOMING FEATURE FUNCTIONS--------------
"""USED ALSO FOR ESCAPES"""

def get_avg_speed(onsets, offsets, tracking_data, session) -> np.array:
    """For each homing, compute the average speed in cm/s

    Returns:
    -- avg_speed: np.array of shape (n_runs, ) with the average speed in cm/s for each homing run"""

    avg_speed = np.zeros(len(onsets))

    for homing, (onset, offset) in enumerate(zip(onsets, offsets)):
        y_loc = tracking_data['head_loc'][onset:offset,1]
        in_shelt = np.where(y_loc > tracking_data['shelter_loc'][0][1])[0]
        trial_speed = tracking_data["avg_Velocity"][onset:offset]
        if len(in_shelt)>0:
            trial_speed = trial_speed[:in_shelt[0]]
        avg_speed[homing] = np.mean(trial_speed)

    assert len(avg_speed) == len(onsets), "Avg speed and number of homings are not the same length"
    return avg_speed

def get_start_and_end_locs(tracking: object, onset_frames: np.array, offset_frames: np.array) -> tuple:
    """Return the start and end locations of each homing run

    Returns:
    -- start_locs: np.array of shape (n_runs, 2) with the start locations of each homing run
    -- end_locs: np.array of shape (n_runs, 2) with the end locations of each homing run

    Each location is in pixels and stored as [x, y]"""
    start_locs = tracking["avg_loc"][onset_frames]
    end_locs = tracking["avg_loc"][offset_frames]
    assert len(start_locs) == len(end_locs), "Start and end locs are not the same length"
    assert len(start_locs) == len(onset_frames), "Start locs and number of homings are not the same length"
    return start_locs, end_locs

def get_condition_homing(video_df, onset_frames, session) -> list:
    """Return the experimental condition that the homing happened"""
    if "shelter" not in video_df.columns:
        video_df = video_df.hstack([pl.Series("shelter", build_shelter_condition_bool(session=session, frame_idx=np.arange(len(video_df))+1, n_frames=len(video_df)))])
    if "barrier_present" not in video_df.columns:
        video_df = video_df.hstack([pl.Series("barrier_present", build_barrier_condition_bool(session=session, frame_idx=np.arange(len(video_df))+1, n_frames=len(video_df)))])
    if "barrier_flipped" not in video_df.columns:
        video_df = video_df.hstack([pl.Series("barrier_flipped", build_flippedbarrier_condition_bool(session=session, frame_idx=np.arange(len(video_df))+1, n_frames=len(video_df)))])

    condition = []
    for onset in onset_frames:
        condition.append(identify_condition_of_trial(video_df.filter(video_df["frames"] == int(onset)), session))
    return condition

def get_avg_homing_angle_for_start_of_run(session, onsets, offsets, tracking_data, speed_thresh = 15) -> dict:
    """This takes the average head angle after the mouse starts running 
    Unlike get_avg_homing_angle_for_first15cm_of_run, it doesn't include the head turn at the start of the homing
    The initial running period is capped at .5 seconds
    
    The speed of running is 10cm/s - this may need to be adjusted, TBD

        Returns:
    -- dic: dictionary with the above arrays stored as values
    -- starting_hdir: the hdir at the start of the homing (before the head turn)
    """

    # init arrays to store the average heading angles to the pre and post flip barrier locations
    avg_hsa = np.zeros(len(onsets))  # One value per homing
    avg_pre_flip_head_angle = np.zeros(len(onsets))
    avg_post_flip_head_angle = np.zeros(len(onsets))
    avg_hdir = np.zeros(len(onsets))
    starting_hdir = np.zeros(len(onsets))

    for idx, (onset,offset) in enumerate(zip(onsets,offsets)):
        hsa = tracking_data["hdir_shelt"][onset:offset+session.video.fps]
        hbarpre = tracking_data["hdir_barrier"][onset:offset+session.video.fps,0]
        hbarpost = tracking_data["hdir_barrier"][onset:offset+session.video.fps,1]
        hdir = tracking_data["hdir"][onset:offset+session.video.fps]
        starting_hdir[idx] = tracking_data["hdir"][onset]
        when_running = tracking_data["avg_Velocity"][onset:offset+session.video.fps]>speed_thresh # this is potentially dangerous if this threshold doesn't work for other sessions
        run_start = np.where(np.diff((when_running).astype(int)) == 1)[0][0]
        run_end = np.where(np.diff((when_running).astype(int)) == -1)[0][0]
        if (run_end - run_start) > (session.video.fps/2): # never look at more than .5 second of running
            run_end = run_start + (session.video.fps/2)
        avg_hdir[idx] = np.mean(hdir[run_start:int(run_end)])
        avg_hsa[idx] = np.mean(hsa[run_start:int(run_end)])
        avg_pre_flip_head_angle[idx] = np.mean(hbarpre[run_start:int(run_end)])
        avg_post_flip_head_angle[idx] = np.mean(hbarpost[run_start:int(run_end)])

    dic = {"avg_hdir": avg_hdir, "avg_hsa": avg_hsa, "avg_pre_flip_head_angle": avg_pre_flip_head_angle, "avg_post_flip_head_angle": avg_post_flip_head_angle}
    
    return dic, starting_hdir

def get_avg_homing_angle_for_first15cm_of_run(session, onsets, offsets, tracking_data, cum_threshold) -> dict:
    """For the first 5 to 15cm of each homing, compute the average angle to each reference locations
    (shelter, pre flip goal, post flip goal).

    Note - 15cm is arbitrary and could be changed in settings_homings.py

    Args:
    -- cum_threshold: int, the distance in cm that the mouse must move before the hsa is computed
    -- onsets: np.array of shape (n_runs, ) with the onset frame of each homing run
    -- offsets: np.array of shape (n_runs, ) with the offset frame of each homing run
    -- tracking_data: dictionary of all the good stuff from the tracking file

    Creates:
    -- avg_hsa: np.array of shape (n_runs, ) with the average hsa for each homing run
    -- avg_pre_flip_head_angle: np.array of shape (n_runs, ) with the average pre_flip_head_angle for each homing run
    -- avg_post_flip_head_angle: np.array of shape (n_runs, ) with the average post_flip_head_angle for each homing run

    Returns:
    -- dic: dictionary with the above arrays stored as values
    """

    # init arrays to store the average heading angles to the pre and post flip barrier locations
    avg_hsa = np.zeros(len(onsets))  # One value per homing
    avg_pre_flip_head_angle = np.zeros(len(onsets))
    avg_post_flip_head_angle = np.zeros(len(onsets))
    avg_hdir = np.zeros(len(onsets))
    starting_hdir = np.zeros(len(onsets))

    # extract
    hsa_data = tracking_data["hdir_shelt"]
    pre_flip_head_angle = tracking_data["hdir_barrier"][:, 0]  # The first index is the pre flip barrier location
    post_flip_head_angle = tracking_data["hdir_barrier"][:, 1]  # The second index is the post flip barrier location

    for i, (onset, offset) in enumerate(zip(onsets, offsets)):

        # Jasmine hack - she wrote this not me, laurence
        if isinstance(onset, np.ndarray):
            onset = onset[0]
            offset = offset[0]

        starting_hdir[i] = tracking_data["hdir"][onset]

        # There shouldn't be a one off error here bcause the onset and offsets should start at 0
        frame_coords = tracking_data["avg_loc"][onset:offset]
        # startframe = the frame after the mouse has travelled 5cm
        # frame_index = the frame when the mouse has reched the cum_threshold
        frame_index, start_frame = cum_distance(onset, offset, frame_coords, session.video.pixels_per_cm, cum_threshold)

        if frame_index == None:
            continue
        
        avg_hdir[i] = circmean(tracking_data["hdir"][start_frame:frame_index])
        
        hsa = hsa_data[start_frame:frame_index]
        avg_hsa[i] = circmean(hsa)

        if len(session.barrier_time) > 0:
            pre_flip_window = pre_flip_head_angle[start_frame:frame_index]
            post_flip_window = post_flip_head_angle[start_frame:frame_index]
            avg_pre_flip_head_angle[i] = circmean(pre_flip_window)
            avg_post_flip_head_angle[i] = circmean(post_flip_window)

    assert len(avg_hsa) == len(onsets), "Avg hsa and number of homings are not the same length"

    dic = {"avg_hdir": avg_hdir, "avg_hsa": avg_hsa, "avg_pre_flip_head_angle": avg_pre_flip_head_angle, "avg_post_flip_head_angle": avg_post_flip_head_angle}

    return dic, starting_hdir

def cum_distance(onset, offset, frame_coords, pixels_per_cm, cum_threshold: int) -> int:
    """Returns the frame when the cumulative distance travelled by the mouse in cm hits the threshold

    Returns:
    -- i: int, the index of the frame where the mouse has travelled cum_threshold cm

    TODO: This could be improved by also finding a strating frame we want to use (instead of including the head turn movement in the avg hsa)
    """
    start_frame = []
    for i, frame in enumerate(range(int(onset), int(offset))):
        if i == 0:
            cum_dist = 0
        elif i > 0:
            x_diff = frame_coords[i][0] - frame_coords[i - 1][0]
            y_diff = frame_coords[i][1] - frame_coords[i - 1][1]
            dist = np.sqrt(x_diff**2 + y_diff**2) / pixels_per_cm
            cum_dist += dist
        if np.logical_and(cum_dist > 5, len(start_frame) == 0):
            start_frame.append(frame)
        if cum_dist >= cum_threshold:
            return frame, start_frame[0]

    # if the mouse never reachs threshold return error message
    logger.error(f"Mouse never reaches cum threshold {cum_threshold} cm")
    frame = None
    return frame, start_frame

# ------------------- Use manual labels -------------------------------

def load_manual_labels(session) -> tuple:
    """Load manual labels from a csv file.
    NB: Assumes image frames are 1 indexed and converts to 0 based indexing here."""
    df = pd.read_csv(os.path.join(session.base_path, session.processed_path) + "\\" + "Borris" + "\\" + "scored_homings.csv")
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
    onsets = fdf[fdf["Behavior type"] == "START"]["Image index"].to_numpy() - 1  # convert to 0-based index
    offsets = fdf[fdf["Behavior type"] == "STOP"]["Image index"].to_numpy() - 1  # convert to 0-based index
    assert len(onsets) == len(offsets), "Onsets and offsets are not the same length"
    assert np.diff(onsets).all() > 0, "Onsets are not increasing"
    assert np.diff(offsets).all() > 0, "Offsets are not increasing"
    durations = offsets - onsets
    durations = np.array([[x] for x in (durations) / session.video.fps])  # match the format of the automatic labels
    return onsets, durations, offsets