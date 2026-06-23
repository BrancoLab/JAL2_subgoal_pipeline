from behave_analysis.database.Experiments.experiment_class import Experiment
from behave_analysis.database.Mice.AllMouses import JAL005 as mouse
from pathlib import Path

JAL5_shelt_2Sept = Experiment(# Mouse specific
                    nick_name = mouse.nick_name,
                    total_sessions = mouse.total_sessions,
                    mouse_number_pyrat = mouse.mouse_number_pyrat,
                    experiment_file_names = mouse.experiment_file_names,
                    root_path = mouse.root_path,
                    
                    # Experiment specific
                    experiment_name='shelter',
                    experiment_idx=1,
                    experiment_date="2023_09_02",
                    experiment_time="11_00_25",
                    shelter_time=[58.3, -1],
                    barrier_time=[],
                    barrier_flip_time=None,
                    experiment_path=Path(r"005_baseline_2023_09_02T11_00_25"))

JAL5_barr_5Sept = Experiment(# Mouse specific
                    nick_name = mouse.nick_name,
                    total_sessions = mouse.total_sessions,
                    mouse_number_pyrat = mouse.mouse_number_pyrat,
                    experiment_file_names = mouse.experiment_file_names,
                    root_path = mouse.root_path,
                    
                    # Experiment specific
                    experiment_name='barrier',
                    experiment_idx=1,
                    experiment_date="2023_09_05",
                    experiment_time="07_48_58",
                    shelter_time=[62, -1],
                    barrier_time=[169, -1],
                    barrier_flip_time=None,
                    experiment_path=Path(r"005_baseline_2023_09_05T07_48_58"))

JAL5_flip1_8Sept = Experiment(# Mouse specific
                    nick_name = mouse.nick_name,
                    total_sessions = mouse.total_sessions,
                    mouse_number_pyrat = mouse.mouse_number_pyrat,
                    experiment_file_names = mouse.experiment_file_names,
                    root_path = mouse.root_path,
                    
                    # Experiment specific
                    experiment_name='flip',
                    experiment_idx=1,
                    experiment_date="2023_09_08",
                    experiment_time="07_36_54",
                    shelter_time=[0.25, -1],
                    barrier_time=[52.08, -1],
                    barrier_flip_time=204.54,
                    experiment_path=Path(r"005_flip1_2023_09_08T07_36_54"))
                    
JAL5_flip3_21Sept = Experiment(# Mouse specific
                    nick_name = mouse.nick_name,
                    total_sessions = mouse.total_sessions,
                    mouse_number_pyrat = mouse.mouse_number_pyrat,
                    experiment_file_names = mouse.experiment_file_names,
                    root_path = mouse.root_path,
                    
                    # Experiment specific
                    experiment_name = 'flip',
                    experiment_idx=2,
                    experiment_date="2023_09_21",
                    experiment_time="11_11_13",
                    shelter_time=[0.4, -1],
                    barrier_time=[69.18, -1],
                    barrier_flip_time=180.40,
                    experiment_path=Path(r"005_flippuff3_2023_09_21T11_11_13"))

JAL5_mush_3oct = Experiment(# Mouse specific
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