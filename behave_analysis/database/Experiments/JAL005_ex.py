from behave_analysis.database.Experiments.experiment_class import Experiment
from behave_analysis.database.Mice.AllMouses import JAL005 as mouse
from pathlib import Path

JAL5_mush1 = Experiment(# Mouse specific
                    nick_name = mouse.nick_name,
                    total_sessions = mouse.total_sessions,
                    mouse_number_pyrat = mouse.mouse_number_pyrat,
                    experiment_file_names = mouse.experiment_file_names,
                    root_path = mouse.root_path,

                    # Experiment specific
                    experiment_name = 'mushroom',
                    experiment_idx = 0,
                    experiment_date = "2023_10_03",
                    experiment_time = "08_11_08",
                    shelter_time = [68, -1],
                    barrier_time = [],
                    barrier_flip_time = None,
                    experiment_path = Path(r"005_mushy1_2023_10_03T08_11_08"))