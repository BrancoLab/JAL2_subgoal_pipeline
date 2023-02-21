# Custom libaries
from behave_analysis.process.session import Session
from behave_analysis.track.register import Register

# OS Libraries
from dataclasses import dataclass
from glob import glob
import numpy as np
import cv2
import os

@dataclass(frozen=True)
class Video:
    num_frames: int
    video_file: str
    fps: int
    height: int
    width: int
    fisheye_correction_file: str
    registration_transform: object
    registration_type: str
    registration_size: tuple
    pixels_per_cm: int
    #tracking_data_file: str # the path to store the tracking data file created in the tracking pipeline step - Moving to tracking step
    
    #! replace these values with your own parameters
    shelter_location: tuple=(512, 921) # CHANGE IF NEEDED (x, y) coordinates of the shelter
    x_offset: int=128 # if the video frame is cropped, how far from the top left edge is it
    y_offset: int=0   # (this is for the fisheye correction step)

def get_Video(session: Session, settings: object, registration_transform: object=None) -> Video:
    """A function that searchs through the directory for a camera avi file and returns a Video object."""
    
    try:
        video_file = glob(os.path.join(session.file_path, "cam*avi"))[-1] # take the last file if there are multiple
    except IndexError:
        raise IndexError(f"No camera video file found with expected name in {session.file_path}")
    
    video_object = cv2.VideoCapture(video_file)
    num_frames = int(video_object.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_object.get(cv2.CAP_PROP_FPS))
    height = int(video_object.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video_object.get(cv2.CAP_PROP_FRAME_WIDTH))
    fisheye_correction_file = settings.fisheye_correction_file
    registration_size = settings.size
    registration_type = settings.registration
    pixels_per_cm = settings.pixels_per_cm
    
    # tracking_data_file = os.path.join(session.file_path, "tracking")
    
    video = Video(num_frames, 
                  video_file, 
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
    video = Video(num_frames, video_file, fps, height, width, fisheye_correction_file, registration_transform, registration_type, registration_size, pixels_per_cm, tracking_data_file)
    
    return video