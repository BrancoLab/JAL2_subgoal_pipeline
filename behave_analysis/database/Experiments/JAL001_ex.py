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
                    shelter_time = [],
                    barrier_time = [])

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
                    shelter_time = [0, 30],
                    barrier_time = [30, -1])

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
                    shelter_time = [],
                    barrier_time = [])

mush5 = Experiment(# Mouse specific
                    nick_name = JAL001.nick_name,
                    total_sessions = JAL001.total_sessions,
                    mouse_number_pyrat = JAL001.mouse_number_pyrat,
                    experiment_file_names = JAL001.experiment_file_names,
                    root_path = JAL001.root_path,
                    
                    # Experiment specific
                    experiment_name = 'Mush5',
                    experiment_idx = 5,
                    experiment_date = "2023_03_21",
                    experiment_time = "08_14_30",
                    experiment_path = Path(r"001_mushroom_5_2023_03_21T08_14_30"),
                    shelter_time = [],
                    barrier_time = [])


