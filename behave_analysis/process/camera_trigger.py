#Custom libs
from settings.settings_process import settings_process as settings_p # So to check if pipeline includes efizz

# Additional libraries if running with efizz
if settings_p.efizz:
    from behave_analysis.utils.downsample_AI_data import remove_idx_as_per_bonsai_ttl_resample

from behave_analysis.process.session import Session

# Os libs
import os
import numpy as np
from dataclasses import dataclass
from glob import glob
import dill as pickle
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class Camera_trigger:
    num_samples: int
    num_frames: int
    frame_trigger_onsets_idx: object
    fps: int

def get_Camera_trigger(session: Session, 
                       indexs_to_remove = None, 
                       down_sample = True, 
                       drop_frames=False):
    
    """
    indexs_to_remove = self.session.ttl.choose_index from process when efizz is ran
    """
    
    AI_file = glob(os.path.join(session.file_path, "analog*"))[-1] # take the last file if there are multiple
    if '.bin' in AI_file: 
        AI_data = np.fromfile(AI_file)
    else:
        with open(AI_file, "rb") as dill_file: AI_data = pickle.load(dill_file)
    camera_trigger_data = AI_data[np.arange(0, len(AI_data), 4)] # four interleaved time series
    logger.info("Length of camera_trigger pre downsample: {}".format(len(camera_trigger_data)))

    # remove the same indexes removed from the bonsai TTL to align the signals
    if down_sample: 
        camera_trigger_data = remove_idx_as_per_bonsai_ttl_resample("video", 
                                                                    camera_trigger_data, 
                                                                    indexs_to_remove, 
                                                                    session)

    camera_trigger_num_samples = len(camera_trigger_data)

    # What is the code? To me it seems the same output regardless of the drop_frames flag
    if drop_frames == False: 
        num_frames_expected, duration_of_video, frame_trigger_onsets_idx = get_num_frames_expected(session, camera_trigger_data, drop_frames=drop_frames)
    if drop_frames == True: 
        num_frames_expected, duration_of_video, frame_trigger_onsets_idx = get_num_frames_expected(session, camera_trigger_data, drop_frames=drop_frames)

    fps = get_fps(session, num_frames_expected, duration_of_video)
    camera_trigger = Camera_trigger(camera_trigger_num_samples, num_frames_expected, frame_trigger_onsets_idx, fps)
    return camera_trigger, camera_trigger_data

def get_num_frames_expected(session: Session, camera_trigger_data: object, drop_frames=False) -> int:
    frame_trigger_onsets = np.diff(camera_trigger_data)
    # np is 0 indexes but frames are not so add 1 
    frame_trigger_onsets_idx = np.where(frame_trigger_onsets > 1)[0] + 1
    ets_idx = np.where(frame_trigger_onsets > 1)[0] + 1
    if drop_frames == True: 
        frame_trigger_onsets_idx = find_drop_frames(session, frame_trigger_onsets_idx)
    num_frames_expected = len(frame_trigger_onsets_idx)
    duration_of_video = (frame_trigger_onsets_idx[-1] - frame_trigger_onsets_idx[0])/session.daq_sampling_rate
    return num_frames_expected, duration_of_video, frame_trigger_onsets_idx

def get_fps(session: Session, num_frames_expected: int, duration_of_video: int) -> int:
    fps = int(num_frames_expected / duration_of_video)
    return fps

def find_drop_frames(session: Session, frame_trigger_onsets_idx, for_video_reader=False):
    frames_csv_path = glob(os.path.join(session.file_path, "frames*"))[-1]
    frames_csv = pd.read_csv(frames_csv_path, names=['frame number', 'zero', 'timestamp'])
    difference_between_frames = np.diff(frames_csv['timestamp'])
    min_difference = np.min(difference_between_frames)
    dropped_frame_diff = difference_between_frames[difference_between_frames>min_difference*2]
    num_frames_dropped = np.round(dropped_frame_diff/min_difference - 1).astype(int)
    index_dropped_frame = np.where(difference_between_frames>min_difference*2)[0] + 1
    if for_video_reader == True:
        return num_frames_dropped, index_dropped_frame
    [print(f" - {int(n)} frames dropped after frame {index_dropped_frame[idx]}\n - Realigning video... ") for idx,n in enumerate(num_frames_dropped)]
    for idx, drop_frame in enumerate(index_dropped_frame):
        for i in range(0,num_frames_dropped[idx]):
            frame_trigger_onsets_idx = np.delete(frame_trigger_onsets_idx, drop_frame)
        index_dropped_frame = index_dropped_frame - num_frames_dropped[idx]
    return frame_trigger_onsets_idx