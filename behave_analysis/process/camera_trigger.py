# Custom libs
from behave_analysis.process.session import NEW_Session
from behave_analysis.utils.AI_dataClass_objects import Camera_trigger

# Os libs
import os
import numpy as np
from glob import glob
import dill as pickle
import json
import pandas as pd
from loguru import logger
import datetime as datetime
from pathlib import Path


def get_Camera_trigger(session: NEW_Session, drop_frames=False):
    """AI data is a 4 channel interleaved signal. The camera pulse is the first channel.
    AI stands for analog input. However, I believe as this is a pulse, it is digital.
    A pulse generated from the NI box one part goes to the camera and another back to the NI box."""

    full_file_path = Path(os.path.join(session.base_path, session.file_path))
    AI_file = list(full_file_path.glob("*analog.bin"))[0]  # need lst and idx as its a generator

    if ".bin" in str(AI_file):
        AI_data = np.fromfile(AI_file)
    else:
        with open(AI_file, "rb") as dill_file:
            AI_data = pickle.load(dill_file)

    camera_trigger_data = AI_data[np.arange(0, len(AI_data), 4)]  # four interleaved time series
    camera_trigger_num_samples = len(camera_trigger_data)
    num_frames_expected, duration_of_video, frame_trigger_onsets_idx = get_num_frames_expected(session, camera_trigger_data, drop_frames=drop_frames)
    fps = get_fps(session, num_frames_expected, duration_of_video)
    camera_trigger = Camera_trigger(camera_trigger_num_samples, num_frames_expected, frame_trigger_onsets_idx, fps)

    # Save as JSON to avoid pickle-based serialization.
    camera_trigger_dict = {
        "num_samples": int(camera_trigger.num_samples),
        "num_frames": int(camera_trigger.num_frames),
        "frame_trigger_onsets_idx": np.asarray(camera_trigger.frame_trigger_onsets_idx).tolist(),
        "fps": int(camera_trigger.fps),
    }
    meta_file = os.path.join(session.base_path, session.processed_path, "camera_trigger.json")
    with open(meta_file, "w", encoding="utf-8") as json_file:
        json.dump(camera_trigger_dict, json_file)

    return camera_trigger, camera_trigger_data


def load_camera_trigger(session: dict, trig_file_path) -> Camera_trigger:
    """Load camera trigger data from JSON, with fallback to legacy session attribute."""
    if isinstance(session, dict):
        if "camera_trigger" in session and session["camera_trigger"] is not None:
            return session["camera_trigger"]
    elif hasattr(session, "camera_trigger") and session.camera_trigger is not None:
        return session.camera_trigger

    if not os.path.isfile(trig_file_path):
        raise FileNotFoundError(f"Camera trigger data not found at {trig_file_path}. Run process step first.")

    with open(trig_file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    return Camera_trigger(
        int(data["num_samples"]),
        int(data["num_frames"]),
        np.asarray(data["frame_trigger_onsets_idx"]),
        int(data["fps"]),
    )


def get_num_frames_expected(session: NEW_Session, camera_trigger_data: object, drop_frames=False) -> int:
    """Find the onset of the frame triggers. And count the onset of pulses as expected number of frames in the camera.

    Args:
        session (Session): session object, used in process dataclass
        camera_trigger_data (object): _description_
        drop_frames (bool, optional): _description_. Defaults to False.

    Returns:
        num_frames_expected (int): The number of frames expected calculated from trigger onset
        duration of video: How long was the video in seconds
        frame trigger onset index: The indexs of pulse onsets

    """

    frame_trigger_onsets = np.diff(camera_trigger_data)
    frame_trigger_onsets_idx = np.where(frame_trigger_onsets > 1)[0] + 1  # compensate for the fact that diff shifts the index by 1

    if drop_frames == True:
        frame_trigger_onsets_idx = find_drop_frames(session, frame_trigger_onsets_idx)

    num_frames_expected = len(frame_trigger_onsets_idx)
    duration_of_video = (frame_trigger_onsets_idx[-1] - frame_trigger_onsets_idx[0]) / session.daq_sampling_rate

    logger.info(f"Number of frames expected based on TTL: {num_frames_expected}")

    return num_frames_expected, duration_of_video, frame_trigger_onsets_idx


def get_fps(session: NEW_Session, num_frames_expected: int, duration_of_video: int) -> int:
    fps = int(num_frames_expected / duration_of_video)
    return fps


def find_drop_frames(session: NEW_Session, frame_trigger_onsets_idx, for_video_reader=False):
    """Find any dropped frames in the video.

    Args:
        session (Session): session object, used in process dataclass
        frame_trigger_onsets_idx (_type_): _description_
        for_video_reader (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    full_file_path = Path(os.path.join(session.base_path, session.file_path))
    frames_csv_path = list(full_file_path.glob("*frames.csv"))[0]

    # frames_csv_path = glob(os.path.join(session.file_path, "frames*"))[-1]

    frames_csv = pd.read_csv(frames_csv_path, names=["frame number", "zero", "timestamp"])
    logger.info(f"Number of frames timestamps: {len(frames_csv['timestamp'])}")
    difference_between_frames = np.diff(frames_csv["timestamp"]) / 125000
    # TODO: figure out if it works better when we divide by 125000

    min_difference = np.min(difference_between_frames)
    dropped_frame_diff = difference_between_frames[difference_between_frames > min_difference * 2]
    num_frames_dropped = np.round(dropped_frame_diff / min_difference - 1).astype(int)
    index_dropped_frame = np.where(difference_between_frames > min_difference * 2)[0] + 1

    if for_video_reader == True:
        return num_frames_dropped, index_dropped_frame

    if len(num_frames_dropped) == 0:
        logger.info("No frames dropped in the video recording")

    elif len(num_frames_dropped) > 0:
        logger.warning(f"{len(num_frames_dropped)} frames dropped in the video recording - Realigning video...")

        for idx, drop_frame in enumerate(index_dropped_frame):
            for i in range(0, num_frames_dropped[idx]):
                frame_trigger_onsets_idx = np.delete(frame_trigger_onsets_idx, drop_frame)
            index_dropped_frame = index_dropped_frame - num_frames_dropped[idx]

    return frame_trigger_onsets_idx
