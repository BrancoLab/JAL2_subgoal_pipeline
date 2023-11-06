from behave_analysis.database.Experiments.experiment_class import Experiment
from behave_analysis.database.Mice.AllMouses import JAL004
from pathlib import Path

JAL4_mush1 = Experiment(# Mouse specific
                    nick_name = JAL004.nick_name,
                    total_sessions = JAL004.total_sessions,
                    mouse_number_pyrat = JAL004.mouse_number_pyrat,
                    experiment_file_names = JAL004.experiment_file_names,
                    root_path = JAL004.root_path,

                    # Experiment specific
                    experiment_name = 'mushroom',
                    experiment_idx = 0,
                    experiment_date = "2023_08_22",
                    experiment_time = "13_13_41",
                    shelter_time = [72, -1],
                    barrier_time = [],
                    barrier_flip_time = None,
                    experiment_path = Path(r"004_mush1_2023_08_22T13_13_41"))


JAL4_3rdSept = Experiment(# Mouse specific
                    nick_name = JAL004.nick_name,
                    total_sessions = JAL004.total_sessions,
                    mouse_number_pyrat = JAL004.mouse_number_pyrat,
                    experiment_file_names = JAL004.experiment_file_names,
                    root_path = JAL004.root_path,
                    
                    # Experiment specific
                    experiment_name = 'flip',
                    experiment_idx = 0,
                    experiment_date = "2023_09_03",
                    experiment_time = "12_04_16",
                    shelter_time = [0, -1],
                    barrier_time = [54, -1],
                    barrier_flip_time = 171,
                    experiment_path = Path(r"004_flip_2023_09_03T12_04_16"))