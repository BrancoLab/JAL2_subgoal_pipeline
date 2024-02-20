from behave_analysis.database.Experiments.experiment_class import Experiment
from behave_analysis.database.Mice.AllMouses import JAL003
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
                    shelter_time = [0, -1],
                    barrier_time = [58, -1], # in minutes when the barrier was present e.g. [30, -1] (if until the end of session put -1 as second in list)
                    barrier_flip_time = 184, # time in minutes when the barrier was flipped e.g. 184
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
                    shelter_time = [0, -1], 
                    barrier_time = [61, -1], # in minutes when the barrier was present e.g. [30, -1] (if until the end of session put -1 as second in list)
                    barrier_flip_time = 173) # time in minutes when the barrier was flipped e.g. 184

flip4stSept_003 =   Experiment(# Mouse specific
                    nick_name = JAL003.nick_name,
                    total_sessions = JAL003.total_sessions,
                    mouse_number_pyrat = JAL003.mouse_number_pyrat,
                    experiment_file_names = JAL003.experiment_file_names,
                    root_path = JAL003.root_path,

                    # Experiment specific
                    experiment_name = 'flip1stSept_003',
                    experiment_idx = 0,
                    experiment_date = "2023_09_04",
                    experiment_time = "13_38_31",
                    experiment_path = Path(r"003_flippuff2_2023_09_04T13_38_31"),
                    shelter_time = [0, -1], 
                    barrier_time = [58.5, -1], # in minutes when the barrier was present e.g. [30, -1] (if until the end of session put -1 as second in list)
                    barrier_flip_time = 181.5) # time in minutes when the barrier was flipped e.g. 184
