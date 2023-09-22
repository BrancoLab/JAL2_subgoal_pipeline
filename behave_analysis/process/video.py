# Custom libaries
from behave_analysis.process.session import NEW_Session
from behave_analysis.track.register import Register

# OS Libraries
from dataclasses import dataclass
from loguru import logger
from glob import glob
import numpy as np
import cv2
import os
from pathlib import Path

@dataclass(frozen=True)
class Video:
    num_frames: int
    camFilePath: str
    fps: int
    height: int
    width: int
    fisheye_correction_file: str
    registration_transform: object
    registration_type: str
    registration_size: tuple
    pixels_per_cm: int
    
    #! replace these values with your own parameters
    shelter_location: tuple=(512, 921) # CHANGE IF NEEDED (x, y) coordinates of the shelter
    x_offset: int=128 # if the video frame is cropped, how far from the top left edge is it
    y_offset: int=0   # (this is for the fisheye correction step)

def get_Video(session: NEW_Session, settings: object, registration_transform: object = None) -> Video:
    """A function that searchs through the directory for a camera avi file and returns a Video object."""
    
    try:
        full_file_path = Path(os.path.join(session.base_path,session.file_path))
        # video_file = str(list(full_file_path.glob("*cam.avi"))[0]) # need lst and idx as its a generator
        datapath_parts = (full_file_path / full_file_path.name).parts
        camFilePath = datapath_parts[-1] + "_cam.avi"
    
    except IndexError:
        raise IndexError(f"No camera video file found with expected name in {session.file_path}")
    
    video_file = os.path.join(session.base_path,session.file_path,camFilePath)
    video_object = cv2.VideoCapture(video_file)
    num_frames = int(video_object.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Number of recorded camera frames : {num_frames}")
    
    fps = int(video_object.get(cv2.CAP_PROP_FPS))
    height = int(video_object.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video_object.get(cv2.CAP_PROP_FRAME_WIDTH))
    fisheye_correction_file = settings.fisheye_correction_file
    registration_size = settings.size
    registration_type = settings.registration
    pixels_per_cm = settings.pixels_per_cm
    
    video = Video(num_frames, 
                  camFilePath, 
                  fps, 
                  height, 
                  width, 
                  fisheye_correction_file, 
                  registration_transform, 
                  registration_type, 
                  registration_size, 
                  pixels_per_cm)
    
    if settings.skip_registration or (isinstance(registration_transform, np.ndarray) and not settings.create_new_registration): 
        return video

    registration_transform = Register(session, video, video_object).transform
    
    # Log the registration transform as if this is None it causing issues downstream at track
    logger.debug(f"Registration transform: {registration_transform}")
    
    video = Video(num_frames, 
                  camFilePath, 
                  fps, 
                  height, 
                  width, 
                  fisheye_correction_file, 
                  registration_transform, 
                  registration_type, 
                  registration_size, 
                  pixels_per_cm)
    
    return video