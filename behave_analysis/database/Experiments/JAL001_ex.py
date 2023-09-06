from behave_analysis.database.Experiments.experiment_class import Experiment
from pathlib import Path
from behave_analysis.database.Mice.JAL001 import JAL001

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
                    experiment_path = Path(r"001_Seq1_2023_03_11T08_39_25"),
                    shelter_only_time = [],
                    barrier_time = [],
                    barrier_flip_time = None)

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
                    experiment_path = Path(r"001_seq1_3_2023_03_17T08_38_03"),
                    shelter_only_time = [0, 30],
                    barrier_time = [30, -1],
                    barrier_flip_time = None)

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
                    experiment_path = Path(r"001_seq1_4_barrierAndShelter_2023_03_20T09_41_15"),
                    shelter_only_time = [],
                    barrier_time = [],
                    barrier_flip_time = None)

mush_3 = Experiment(# Mouse specific
                    nick_name = JAL001.nick_name,
                    total_sessions = JAL001.total_sessions,
                    mouse_number_pyrat = JAL001.mouse_number_pyrat,
                    experiment_file_names = JAL001.experiment_file_names,
                    root_path = JAL001.root_path,

                    # Experiment specific
                    experiment_name = 'Mushroom3',
                    experiment_idx = 3,
                    experiment_date = "2023_03_15",
                    experiment_time = "07_43_42",
                    experiment_path = Path(r"001_mushroom3_2023_03_15T07_43_42"),
                    shelter_only_time = [28, -1],
                    barrier_time = [],
                    barrier_flip_time = None)

seq1_2 = Experiment(# Mouse specific
                    nick_name = JAL001.nick_name,
                    total_sessions = JAL001.total_sessions,
                    mouse_number_pyrat = JAL001.mouse_number_pyrat,
                    experiment_file_names = JAL001.experiment_file_names,
                    root_path = JAL001.root_path,

                    # Experiment specific
                    experiment_name = 'Seq1',
                    experiment_idx = 2,
                    experiment_date = "2023_03_14",
                    experiment_time = "08_11_32",
                    experiment_path = Path(r"001_seq1_2_2023_03_14T08_11_32"),
                    shelter_only_time = [0, 29],
                    barrier_time = [29, -1],
                    barrier_flip_time = None)
