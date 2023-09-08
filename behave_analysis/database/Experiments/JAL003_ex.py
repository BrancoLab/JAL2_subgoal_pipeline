from behave_analysis.database.Experiments.experiment_class import Experiment
from behave_analysis.database.Mice.JAL003 import JAL003
from pathlib import Path

JAL3_flip_rot = Experiment(# Mouse specific
                    nick_name = JAL003.nick_name,
                    total_sessions = JAL003.total_sessions,
                    mouse_number_pyrat = JAL003.mouse_number_pyrat,
                    experiment_file_names = JAL003.experiment_file_names,
                    root_path = JAL003.root_path,

                    # Experiment specific
                    experiment_name = 'flip_rotated_1',
                    experiment_idx = 0,
                    experiment_date = "2023_08_25",
                    experiment_time = "09_42_06",
                    shelter_only_time = [0, 58],
                    barrier_time = [58, -1], 
                    barrier_flip_time = 184,
                    experiment_path = Path(r"003_flip_rotated_2023_08_25T09_42_06"))

flip1stSept_003 =   Experiment(# Mouse specific
                    nick_name = JAL003.nick_name,
                    total_sessions = JAL003.total_sessions,
                    mouse_number_pyrat = JAL003.mouse_number_pyrat,
                    experiment_file_names = JAL003.experiment_file_names,
                    root_path = JAL003.root_path,

                    # Experiment specific
                    experiment_name = 'flip1stSept_003',
                    experiment_idx = 0,
                    experiment_date = "2023_09_01",
                    experiment_time = "16_23_39",
                    experiment_path = Path(r"003_flip_1Sept_2023_09_01T16_23_39"),
                    shelter_only_time = [0, 60], # The time when ONLY the shelter is present
                    barrier_time = [60, -1],
                    barrier_flip_time = None) # The time when BOTH the shelter and barrier are present
