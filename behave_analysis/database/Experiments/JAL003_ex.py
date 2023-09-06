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
                    experiment_name = 'mushroom',
                    experiment_idx = 0,
                    experiment_date = "2023_08_25",
                    experiment_time = "09_42_06",
                    shelter_only_time = [0, 58],
                    barrier_time = [58, -1], 
                    barrief_flip_time = 184,
                    experiment_path = Path(r"003_flip_rotated_2023_08_25T09_42_06"))