from behave_analysis.process.session import Session
import numpy as np
import pandas as pd
import os
from glob import glob

def find_drop_frames(session: Session):
    frames_csv_path = glob(os.path.join(session.file_path, "frames*"))[-1]
    frames_csv = pd.read_csv(frames_csv_path, names=['frame number', 'zero', 'timestamp'])
    difference_between_frames = np.diff(frames_csv['timestamp'])
    min_difference = np.min(difference_between_frames)
    dropped_frame_diff = difference_between_frames[difference_between_frames>min_difference*2]
    num_frames_dropped = np.round(dropped_frame_diff/min_difference - 1)
    index_dropped_frame = np.where(difference_between_frames>min_difference*2)[0]
    [print(f" - {int(n)} frames dropped after frame {index_dropped_frame[idx]} ---") for idx,n in enumerate(num_frames_dropped)]