from behave_analysis.database.Experiments.experiment_class import Experiment
from pathlib import Path
from behave_analysis.database.Mice.AllMouses import JR3440, JR3456, JR3457

burrow_3440 = Experiment(# Mouse specific
                    nick_name = JR3440.nick_name,
                    total_sessions = JR3440.total_sessions,
                    mouse_number_pyrat = JR3440.mouse_number_pyrat,
                    experiment_file_names = JR3440.experiment_file_names,
                    root_path = JR3440.root_path,

                    # Experiment specific
                    experiment_name = 'burrow',
                    experiment_idx = 0,
                    experiment_date = "2023_10_18",
                    experiment_time = "08_55_49",
                    experiment_path = Path(r"JR3440_burrow_2023_10_18T08_55_49"),
                    shelter_time = [0, -1],
                    barrier_time = [],
                    barrier_flip_time = None)

burrow_3456 = Experiment(# Mouse specific
                    nick_name = JR3456.nick_name,
                    total_sessions = JR3456.total_sessions,
                    mouse_number_pyrat = JR3456.mouse_number_pyrat,
                    experiment_file_names = JR3456.experiment_file_names,
                    root_path = JR3456.root_path,

                    # Experiment specific
                    experiment_name = 'burrow',
                    experiment_idx = 0,
                    experiment_date = "2023_10_19",
                    experiment_time = "10_23_14",
                    experiment_path = Path(r"JR3456_burrow_2023_10_19T10_23_14"),
                    shelter_time = [0, -1],
                    barrier_time = [],
                    barrier_flip_time = None)

burrow_3457 = Experiment(# Mouse specific
                    nick_name = JR3457.nick_name,
                    total_sessions = JR3457.total_sessions,
                    mouse_number_pyrat = JR3457.mouse_number_pyrat,
                    experiment_file_names = JR3457.experiment_file_names,
                    root_path = JR3457.root_path,

                    # Experiment specific
                    experiment_name = 'burrow',
                    experiment_idx = 0,
                    experiment_date = "2023_10_18",
                    experiment_time = "09_43_38",
                    experiment_path = Path(r"JR3457_burrow_2023_10_18T09_43_38"),
                    shelter_time = [0,-1],
                    barrier_time = [],
                    barrier_flip_time = None)

burrow_3457_2 = Experiment(# Mouse specific
                    nick_name = JR3457.nick_name,
                    total_sessions = JR3457.total_sessions,
                    mouse_number_pyrat = JR3457.mouse_number_pyrat,
                    experiment_file_names = JR3457.experiment_file_names,
                    root_path = JR3457.root_path,

                    # Experiment specific
                    experiment_name = 'burrow',
                    experiment_idx = 1,
                    experiment_date = "2023_10_19",
                    experiment_time = "09_52_06",
                    experiment_path = Path(r"JR3457_burrow2_2023_10_19T09_52_06"),
                    shelter_time = [0,-1],
                    barrier_time = [],
                    barrier_flip_time = None)