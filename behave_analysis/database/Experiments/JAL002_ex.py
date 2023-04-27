from behave_analysis.database.Experiments.experiment_class import Experiment
from behave_analysis.database.Mice.JAL002 import JAL002
from pathlib import Path

firstConnection = Experiment(# Mouse specific
                    nick_name = JAL002.nick_name,
                    total_sessions = JAL002.total_sessions,
                    mouse_number_pyrat = JAL002.mouse_number_pyrat,
                    experiment_file_names = JAL002.experiment_file_names,
                    root_path = JAL002.root_path,

                    # Experiment specific
                    experiment_name = 'First connect',
                    experiment_idx = 0,
                    experiment_date = "2023_04_19",
                    experiment_time = "10_24_04",
                    experiment_path = Path(r"002_firstConnect_2023_04_19T10_24_04"),
                    shelter_time = [],
                    barrier_time = [])