#Custom libs
from behave_analysis.process.session import Session
from behave_analysis.utils.AI_dataClass_objects import Camera_trigger

# Os libs
import os
import numpy as np
from glob import glob
import dill as pickle
import pandas as pd
from loguru import logger

def get_Camera_trigger(session: Session, 
                       drop_frames = False):
    """
    AI data is a 4 channel interleaved signal. The camera pulse is the first channel.
    AI stands for analog input. However, I believe as this is a pulse, it is digital.
    A pulse generated from the NI box one part goes to the camera and another back to the NI box. 
    """
    
    AI_file = glob(os.path.join(session.file_path, "analog*"))[-1] # take the last file if there are multiple
    if '.bin' in AI_file: 
        AI_data = np.fromfile(AI_file)
    else:
        with open(AI_file, "rb") as dill_file: AI_data = pickle.load(dill_file)
    camera_trigger_data = AI_data[np.arange(0, len(AI_data), 4)] # four interleaved time series

    camera_trigger_num_samples = len(camera_trigger_data)
    num_frames_expected, duration_of_video, frame_trigger_onsets_idx = get_num_frames_expected(session, 
                                                                                               camera_trigger_data, 
                                                                                               drop_frames = drop_frames)

    fps = get_fps(session, num_frames_expected, duration_of_video)
    
    camera_trigger = Camera_trigger(camera_trigger_num_samples, 
                                    num_frames_expected, 
                                    frame_trigger_onsets_idx, 
                                    fps)
    
    return camera_trigger, camera_trigger_data

def get_num_frames_expected(session: Session, 
                            camera_trigger_data: object, 
                            drop_frames=False) -> int:
    """Find the onset of the frame triggers. And count the onset of pulses as expected number of frames in the camera.

    Args:
        session (Session): _description_
        camera_trigger_data (object): _description_
        drop_frames (bool, optional): _description_. Defaults to False.

    Returns:
        num_frames_expected (int): The number of frames expected calculated from trigger onset
        duration of video: How long was the video in seconds
        frame trigger onset index: The indexs of pulse onsets
        
    """
    
    frame_trigger_onsets = np.diff(camera_trigger_data)
    frame_trigger_onsets_idx = np.where(frame_trigger_onsets > 1)[0] + 1 # np is 0 indexes but frames are not so add 1 
    
    if drop_frames == True: 
        frame_trigger_onsets_idx = find_drop_frames(session, frame_trigger_onsets_idx)
        
    num_frames_expected = len(frame_trigger_onsets_idx)
    duration_of_video = (frame_trigger_onsets_idx[-1] - frame_trigger_onsets_idx[0]) / session.daq_sampling_rate
    
    return num_frames_expected, duration_of_video, frame_trigger_onsets_idx

def get_fps(session: Session, num_frames_expected: int, duration_of_video: int) -> int:
    fps = int(num_frames_expected / duration_of_video)
    return fps

def find_drop_frames(session: Session, frame_trigger_onsets_idx, for_video_reader=False):
    """Find any dropped frames in the video.

    Args:
        session (Session): _description_
        frame_trigger_onsets_idx (_type_): _description_
        for_video_reader (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    
    frames_csv_path = glob(os.path.join(session.file_path, "frames*"))[-1]
    frames_csv = pd.read_csv(frames_csv_path, names=['frame number', 'zero', 'timestamp'])
    difference_between_frames = np.diff(frames_csv['timestamp'])
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