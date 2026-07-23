from behave_analysis.database.Experiments.experiment_class import Experiment
from behave_analysis.database.Mice.AllMouses import JAL003
from pathlib import Path

JAL3_shelt_17aug = Experiment(# Mouse specific
                    nick_name = JAL003.nick_name,
                    total_sessions = JAL003.total_sessions,
                    mouse_number_pyrat = JAL003.mouse_number_pyrat,
                    experiment_file_names = JAL003.experiment_file_names,
                    root_path = JAL003.root_path,

                    # Experiment specific
                    experiment_name = 'shelter',
                    experiment_idx = 0,
                    experiment_date = "2023_08_17",
                    experiment_time = "08_16_46",
                    valid_time=[.25, -1], 
                    shelter_time = [60.2, -1],
                    barrier_time = [], # in minutes when the barrier was present e.g. [30, -1] (if until the end of session put -1 as second in list)
                    barrier_flip_time = None, # time in minutes when the barrier was flipped e.g. 184
                    experiment_path = Path(r"003_baseline_2023_08_17T08_16_46"))

JAL3_mush_21aug = Experiment(# Mouse specific
                    nick_name = JAL003.nick_name,
                    total_sessions = JAL003.total_sessions,
                    mouse_number_pyrat = JAL003.mouse_number_pyrat,
                    experiment_file_names = JAL003.experiment_file_names,
                    root_path = JAL003.root_path,

                    # Experiment specific
                    experiment_name = 'mushroom',
                    experiment_idx = 0,
                    experiment_date = "2023_08_21",
                    experiment_time = "08_58_29",
                    valid_time=[.25, -1],
                    shelter_time = [90.6, -1],
                    barrier_time = [], # in minutes when the barrier was present e.g. [30, -1] (if until the end of session put -1 as second in list)
                    barrier_flip_time = None, # time in minutes when the barrier was flipped e.g. 184
                    experiment_path = Path(r"003_mushroom_2023_08_21T08_58_29"))

JAL3_flip1_22aug = Experiment(# Mouse specific
                    nick_name = JAL003.nick_name,
                    total_sessions = JAL003.total_sessions,
                    mouse_number_pyrat = JAL003.mouse_number_pyrat,
                    experiment_file_names = JAL003.experiment_file_names,
                    root_path = JAL003.root_path,

                    # Experiment specific
                    experiment_name = 'flip',
                    experiment_idx = 1,
                    experiment_date = "2023_08_22",
                    experiment_time = "08_04_56",
                    valid_time = [.1, -1],
                    shelter_time = [0, -1],
                    barrier_time = [83.15, -1], # in minutes when the barrier was present e.g. [30, -1] (if until the end of session put -1 as second in list)
                    barrier_flip_time = 202.7, # time in minutes when the barrier was flipped e.g. 184
                    experiment_path = Path(r"003_flip_2023_08_22T08_04_56"))

JAL3_flip2_25aug = Experiment(# Mouse specific
                    nick_name = JAL003.nick_name,
                    total_sessions = JAL003.total_sessions,
                    mouse_number_pyrat = JAL003.mouse_number_pyrat,
                    experiment_file_names = JAL003.experiment_file_names,
                    root_path = JAL003.root_path,

                    # Experiment specific
                    experiment_name = 'flip',
                    experiment_idx = 2,
                    experiment_date = "2023_08_25",
                    experiment_time = "09_42_06",
                    valid_time=[.2, -1],
                    shelter_time = [0, -1],
                    barrier_time = [58, -1], # in minutes when the barrier was present e.g. [30, -1] (if until the end of session put -1 as second in list)
                    barrier_flip_time = 184, # time in minutes when the barrier was flipped e.g. 184
                    experiment_path = Path(r"003_flip_rotated_2023_08_25T09_42_06"))

JAL3_flip3_29aug = Experiment(# Mouse specific
                    nick_name = JAL003.nick_name,
                    total_sessions = JAL003.total_sessions,
                    mouse_number_pyrat = JAL003.mouse_number_pyrat,
                    experiment_file_names = JAL003.experiment_file_names,
                    root_path = JAL003.root_path,

                    # Experiment specific
                    experiment_name = 'flip',
                    experiment_idx = 3,
                    experiment_date = "2023_08_29",
                    experiment_time = "10_57_43",
                    valid_time = [.25, -1],
                    shelter_time = [0, -1],
                    barrier_time = [67.8, -1], # in minutes when the barrier was present e.g. [30, -1] (if until the end of session put -1 as second in list)
                    barrier_flip_time = 178, # time in minutes when the barrier was flipped e.g. 184
                    experiment_path = Path(r"003_flip_rotated_2023_08_29T10_57_43"))

JAL3_flip4_1sept =   Experiment(# Mouse specific
                    nick_name = JAL003.nick_name,
                    total_sessions = JAL003.total_sessions,
                    mouse_number_pyrat = JAL003.mouse_number_pyrat,
                    experiment_file_names = JAL003.experiment_file_names,
                    root_path = JAL003.root_path,

                    # Experiment specific
                    experiment_name = 'flip',
                    experiment_idx = 4,
                    experiment_date = "2023_09_01",
                    experiment_time = "16_23_39",
                    experiment_path = Path(r"003_flip_1Sept_2023_09_01T16_23_39"),
                    valid_time=[.3, -1],
                    shelter_time = [0, -1], 
                    barrier_time = [61, -1], # in minutes when the barrier was present e.g. [30, -1] (if until the end of session put -1 as second in list)
                    barrier_flip_time = 173) # time in minutes when the barrier was flipped e.g. 184

JAL3_flip5_4sept =   Experiment(# Mouse specific
                    nick_name = JAL003.nick_name,
                    total_sessions = JAL003.total_sessions,
                    mouse_number_pyrat = JAL003.mouse_number_pyrat,
                    experiment_file_names = JAL003.experiment_file_names,
                    root_path = JAL003.root_path,

                    # Experiment specific
                    experiment_name = 'flip',
                    experiment_idx = 5,
                    experiment_date = "2023_09_04",
                    experiment_time = "13_38_31",
                    experiment_path = Path(r"003_flippuff2_2023_09_04T13_38_31"),
                    valid_time=[.45, -1],
                    shelter_time = [0, -1], 
                    barrier_time = [58.5, -1], # in minutes when the barrier was present e.g. [30, -1] (if until the end of session put -1 as second in list)
                    barrier_flip_time = 181.5) # time in minutes when the barrier was flipped e.g. 184

JAL3_flip6_7sept =   Experiment(# Mouse specific
                    nick_name = JAL003.nick_name,
                    total_sessions = JAL003.total_sessions,
                    mouse_number_pyrat = JAL003.mouse_number_pyrat,
                    experiment_file_names = JAL003.experiment_file_names,
                    root_path = JAL003.root_path,

                    # Experiment specific
                    experiment_name = 'flip',
                    experiment_idx = 6,
                    experiment_date = "2023_09_07",
                    experiment_time = "12_23_47",
                    experiment_path = Path(r"003_003puftrotato3_2023_09_07T12_23_47"),
                    valid_time = [.25, -1],
                    shelter_time = [0, -1],
                    barrier_time = [70.5, 289.25], # in minutes when the barrier was present e.g. [30, -1] (if until the end of session put -1 as second in list)
                    barrier_flip_time = 182) # time in minutes when the barrier was flipped e.g. 184
