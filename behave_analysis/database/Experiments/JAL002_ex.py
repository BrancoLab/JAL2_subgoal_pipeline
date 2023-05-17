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

habituation_002 = Experiment(# Mouse specific
                    nick_name = JAL002.nick_name,
                    total_sessions = JAL002.total_sessions,
                    mouse_number_pyrat = JAL002.mouse_number_pyrat,
                    experiment_file_names = JAL002.experiment_file_names,
                    root_path = JAL002.root_path,


                    # Experiment specific
                    experiment_name = 'habituation',
                    experiment_idx = 0,
                    experiment_date = "2023_04_20",
                    experiment_time = "08_41_44",
                    shelter_time = [],
                    barrier_time = [],
                    experiment_path = Path(r"002_habituation_openfield_2023_04_20T08_41_44"))


mushroom1_002 = Experiment(# Mouse specific
                    nick_name = JAL002.nick_name,
                    total_sessions = JAL002.total_sessions,
                    mouse_number_pyrat = JAL002.mouse_number_pyrat,
                    experiment_file_names = JAL002.experiment_file_names,
                    root_path = JAL002.root_path,


                    # Experiment specific
                    experiment_name = 'mushroom',
                    experiment_idx = 0,
                    experiment_date = "2023_04_21",
                    experiment_time = "08_24_45",
                    shelter_time = [45, -1],
                    barrier_time = [],
                    experiment_path = Path(r"002_mushroom1_2023_04_21T08_24_45"))

mushroom4_002 = Experiment(# Mouse specific
                    nick_name = JAL002.nick_name,
                    total_sessions = JAL002.total_sessions,
                    mouse_number_pyrat = JAL002.mouse_number_pyrat,
                    experiment_file_names = JAL002.experiment_file_names,
                    root_path = JAL002.root_path,


                    # Experiment specific
                    experiment_name = 'mushroom',
                    experiment_idx = 3,
                    experiment_date = "2023_05_01",
                    experiment_time = "08_16_14",
                    shelter_time = [40, -1],
                    barrier_time = [],
                    experiment_path = Path(r"002_mushroom4_2023_05_01T08_16_14"))

seq1_3_002 = Experiment(# Mouse specific
                        nick_name = JAL002.nick_name,
                        total_sessions = JAL002.total_sessions,
                        mouse_number_pyrat = JAL002.mouse_number_pyrat,
                        experiment_file_names = JAL002.experiment_file_names,
                        root_path = JAL002.root_path,
                        
                         # Experiment specific
                    experiment_name = 'sequence1_3',
                    experiment_idx = 0,
                    experiment_date = "2023_04_28",
                    experiment_time = "08_31_30",
                    shelter_time = [0, -1],
                    barrier_time = [30, -1],
                    experiment_path = Path(r"002_Sequence1_3_2023_04_28T08_31_30"))