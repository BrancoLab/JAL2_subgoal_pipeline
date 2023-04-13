from dataclasses import dataclass
from behave_analysis.database.mice import Mouse
from behave_analysis.database.mice import JAL001
from pathlib import Path

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

seq3 = Experiment(# Mouse specific
                    nick_name = JAL001.nick_name,
                    total_sessions = JAL001.total_sessions,
                    mouse_number_pyrat = JAL001.mouse_number_pyrat,
                    experiment_file_names = JAL001.experiment_file_names,
                    root_path = JAL001.root_path,

                    # Experiment specific
                    experiment_name = 'Seq3',
                    experiment_idx = 3,
                    experiment_date = "2023_03_17",
                    experiment_time = "08_38_03",
                    experiment_path = Path(r"001_seq1_3_2023_03_17T08_38_03"))

seq4 = Experiment(# Mouse specific
                    nick_name = JAL001.nick_name,
                    total_sessions = JAL001.total_sessions,
                    mouse_number_pyrat = JAL001.mouse_number_pyrat,
                    experiment_file_names = JAL001.experiment_file_names,
                    root_path = JAL001.root_path,

                    # Experiment specific
                    experiment_name = 'Seq4',
                    experiment_idx = 4,
                    experiment_date = "2023_03_20",
                    experiment_time = "09_41_15",
                    experiment_path = Path(r"001_seq1_4_barrierAndShelter_2023_03_20T09_41_15"))

test = Experiment(# Mouse specific
                    nick_name = JAL001.nick_name,
                    total_sessions = JAL001.total_sessions,
                    mouse_number_pyrat = JAL001.mouse_number_pyrat,
                    experiment_file_names = JAL001.experiment_file_names,
                    root_path = JAL001.root_path,

                    # Experiment specific
                    experiment_name = 'test',
                    experiment_idx = 4,
                    experiment_date = "2023_03_22",
                    experiment_time = "09_58_59",
                    experiment_path = Path(r"001_synctestmanyaudio_2023_03_22T09_58_59"))

laserTestShort = Experiment(# Mouse specific
                    nick_name = JAL001.nick_name,
                    total_sessions = JAL001.total_sessions,
                    mouse_number_pyrat = JAL001.mouse_number_pyrat,
                    experiment_file_names = JAL001.experiment_file_names,
                    root_path = JAL001.root_path,

                    # Experiment specific
                    experiment_name = 'Laser test Jazz',
                    experiment_idx = 666,
                    experiment_date = "2023_04_11",
                    experiment_time = "12_52_28",
                    experiment_path = Path(r"999_laserTestJAZZ_2023_04_11T12_52_28"))

laserTest = Experiment(# Mouse specific
                    nick_name = JAL001.nick_name,
                    total_sessions = JAL001.total_sessions,
                    mouse_number_pyrat = JAL001.mouse_number_pyrat,
                    experiment_file_names = JAL001.experiment_file_names,
                    root_path = JAL001.root_path,

                    # Experiment specific
                    experiment_name = 'Laser test Jazz Hands 2',
                    experiment_idx = 999,
                    experiment_date = "2023_04_11",
                    experiment_time = "15_51_58",
                    experiment_path = Path(r"999_laserTestJAZZhands2_2023_04_11T15_51_58"))