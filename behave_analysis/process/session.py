"""The Session class is created at the start of the pipeline. In essence it contains all the necessary data for processing
such as:
- What was the name of the session?
- What is the mouse called?
- Where is the path to the data?

The databank populates much of this class through the session_ID.
"""

# Custom libaries

from dataclasses import dataclass

# OS Libaries

import os
import numpy as np
import numpy.typing as npt

@dataclass(frozen=False)
class Session:
    name: str
    number: int
    mouse: str
    experiment: str
    previous_sessions: int
    file_path: str
    metadata_file: str
    daq_sampling_rate: int = 15000
    camera_trigger: object = None
    audio: object = None
    video: object = None
    ttl: object = None
    ephys: object = None
    photo_resistor: object = None
    homing: object = None
    threshold_crossing: object = None

def get_Session(session_ID: np.ndarray):
    """Populate the Session class above

    Args:
        session_ID (np.ndarray): Created by collect_session_IDs() retur

    Returns:
        class: Session class 
    """
    global_session_number = session_ID[0]
    local_session_number = session_ID[1]
    experiment = session_ID[2]
    num_previous_sessions = session_ID[3]
    file_path = session_ID[4]
    metadata_file = os.path.join(file_path, "metadata")
    session_folder_name = os.path.basename(file_path)
    mouse = session_folder_name[:4]
    name = experiment + ' ' + str(local_session_number)   
    
    return Session(name, global_session_number, mouse, experiment, num_previous_sessions, file_path, metadata_file)
