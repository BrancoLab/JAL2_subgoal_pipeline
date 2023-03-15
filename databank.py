'''A database of all the experiments and mice run in the JJAL team on the big rig'''

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Mouse:
    """A class to store mouse by mouse information"""
    nick_name: str
    total_sessions: int
    mouse_number_pyrat: str
    experiment_file_names: list # A list of the file names for each experiment
    root_path: str # The path to the local data folder

@dataclass(frozen=True)
class Experiment(Mouse):
    """A class to store experiment by experiment information"""
    experiment_name: str
    experiment_idx: int # E.G if this was Mushroom 1, this would be 1
    experiment_date: str
    experiment_time: str
    experiment_path: str # Just the name of the experiment folder e.g. 001_mushroom1_2023_03_10T07_15_15
    shelter_placement_time: float = None # When was shelter placed in seconds
    barrier_placement_time: float = None # When was barrier placed in seconds

# Mices
JAL001 = Mouse(nick_name = 'JAL001',
               total_sessions = 1,
               mouse_number_pyrat = "BAA-1102922",
               experiment_file_names = [r"001_mushroom1_2023_03_10T07_15_15"],
               root_path = Path(r"D:\efizz"))

# Experiments
mushroom1 = Experiment(# Mouse specific
                       nick_name = JAL001.nick_name,
                       total_sessions = JAL001.total_sessions,
                       mouse_number_pyrat = JAL001.mouse_number_pyrat,
                       experiment_file_names = JAL001.experiment_file_names,
                       root_path = JAL001.root_path,

                       # Experiment specific
                       experiment_name = 'mushroom1',
                       experiment_idx = 0,
                       experiment_date = "2023_03_10",
                       experiment_time = "07_15_15",
                       experiment_path = Path(r"001_mushroom1_2023_03_10T07_15_15"))

seq1 = Experiment(# Mouse specific
                    nick_name = JAL001.nick_name,
                    total_sessions = JAL001.total_sessions,
                    mouse_number_pyrat = JAL001.mouse_number_pyrat,
                    experiment_file_names = JAL001.experiment_file_names,
                    root_path = JAL001.root_path,

                    # Experiment specific
                    experiment_name = 'Seq1',
                    experiment_idx = 0,
                    experiment_date = "2023_03_11",
                    experiment_time = "08_39_25",
                    experiment_path = Path(r"001_Seq1_2023_03_11T08_39_25"))

# Place all experiments in a list and 
experiments_objects = [seq1]