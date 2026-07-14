"""
The session class is used to store all the processed data from a single session. It ingests the data from the database
such as the mouse and experiment dataclasses
"""

# OS Libaries
from dataclasses import dataclass
from pathlib import Path
from behave_analysis.database.computer_ID import get_computer_specific_paths
import os

@dataclass(frozen=False)
class NEW_Session:
    name: str
    number: int
    mouse: str
    date: str
    experiment: str
    shelter_time: None
    barrier_time: None
    barrier_flip_time: None
    valid_time: list
    file_path: str
    base_path: str
    processed_path: str
    metadata_file: str
    shelter_location: int = None
    barrier_location: int = None
    daq_sampling_rate: int = 15000
    camera_trigger: object = None
    audio: object = None
    video: object = None
    ttl: object = None
    ephys: object = None
    homing: object = None
    threshold_crossing: object = None
    
def get_experiment(experiment_data_class):
    """
    Populate the Session class above by passing the experiment data class from the databank 
    """
    experiment_type = experiment_data_class.experiment_name
    mouse = experiment_data_class.nick_name
    experiment_repeat = experiment_data_class.experiment_idx
    experiment_description = f"Mouse: {mouse}, Experiment: {experiment_type}, Run number: {experiment_repeat}"
    base_path, _ =  get_computer_specific_paths(os.path.join(experiment_data_class.root_path,experiment_data_class.experiment_path), return_ceph = True)
    file_path = os.path.join(experiment_data_class.root_path,experiment_data_class.experiment_path)
    metadata_file = os.path.join(file_path, "processed_data", "metadata")
        
    return NEW_Session(name = experiment_description, 
                       number = experiment_repeat, # Maybe this is incorrect and the wrong number after the refactor
                       mouse = mouse, 
                       date = experiment_data_class.experiment_date,
                       experiment = experiment_type, 
                       shelter_time = experiment_data_class.shelter_time,
                       barrier_time = experiment_data_class.barrier_time,
                       barrier_flip_time = experiment_data_class.barrier_flip_time,
                       valid_time= experiment_data_class.valid_time,
                       base_path = base_path,
                       file_path = file_path, 
                       processed_path = os.path.join(file_path, "processed_data"),
                       metadata_file = metadata_file)

def confirm_session(loaded_session, session):
    
    """Take a loaded session and confirm/update any metadata that is missing or incorrect."""
    loaded_session.name = session.name
    loaded_session.number = session.number
    loaded_session.mouse = session.mouse
    loaded_session.date = session.date
    loaded_session.experiment = session.experiment
    loaded_session.shelter_time = session.shelter_time
    loaded_session.barrier_time = session.barrier_time
    loaded_session.barrier_flip_time = session.barrier_flip_time
    loaded_session.valid_time = session.valid_time
    loaded_session.base_path = session.base_path
    loaded_session.file_path = session.file_path
    loaded_session.processed_path = session.processed_path
    loaded_session.metadata_file = session.metadata_file

    return loaded_session