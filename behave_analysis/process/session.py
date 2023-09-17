"""
The session class is used to store all the processed data from a single session. It ingests the data from the database
such as the mouse and experiment dataclasses
"""

# OS Libaries
from dataclasses import dataclass
import os

@dataclass(frozen=False)
class NEW_Session:
    name: str
    number: int
    mouse: str
    experiment: str
    shelter_time: None
    barrier_time: None
    file_path: str
    processed_path: str
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
    
def get_experiment(experiment_data_class):
    """
    Populate the Session class above by passing the experiment data class from the databank 
    """
    experiment_type = experiment_data_class.experiment_name
    mouse = experiment_data_class.nick_name
    experiment_repeat = experiment_data_class.experiment_idx
    experiment_description = f"Mouse: {mouse}, Experiment: {experiment_type}, Run number: {experiment_repeat}"
    file_path = experiment_data_class.root_path.joinpath(experiment_data_class.experiment_path)
    metadata_file = os.path.join(file_path, "processed_data", "metadata")
        
    return NEW_Session(name = experiment_description, 
                       number = experiment_repeat, # Maybe this is incorrect and the wrong number after the refactor
                       mouse = mouse, 
                       experiment = experiment_type, 
                       shelter_time = experiment_data_class.shelter_only_time,
                       barrier_time = experiment_data_class.barrier_time,
                       file_path = file_path, 
                       processed_path = os.path.join(file_path, "processed_data"),
                       metadata_file = metadata_file)
