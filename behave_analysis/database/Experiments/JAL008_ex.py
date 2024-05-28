from behave_analysis.database.Experiments.experiment_class import Experiment
from behave_analysis.database.Mice.AllMouses import JAL008 as mouse
from pathlib import Path

JAL8_shelt_22apr = Experiment(  # Mouse specific
    nick_name=mouse.nick_name,
    total_sessions=mouse.total_sessions,
    mouse_number_pyrat=mouse.mouse_number_pyrat,
    experiment_file_names=mouse.experiment_file_names,
    root_path=mouse.root_path,
    # Experiment specific
    experiment_name="shelt",
    experiment_idx=2,
    experiment_date="2024_04_22",
    experiment_time="10_51_22",
    shelter_time=[73, -1],
    barrier_time=[],
    barrier_flip_time=None,
    experiment_path=Path(r"JAL008_empty_shelter2_2024_04_22T10_51_22"),
)

JAL8_flip1_25apr = Experiment(  # Mouse specific
    nick_name=mouse.nick_name,
    total_sessions=mouse.total_sessions,
    mouse_number_pyrat=mouse.mouse_number_pyrat,
    experiment_file_names=mouse.experiment_file_names,
    root_path=mouse.root_path,
    # Experiment specific
    experiment_name="flip",
    experiment_idx=3,
    experiment_date="2024_04_25",
    experiment_time="11_27_42",
    shelter_time=[2.5,-1],
    barrier_time=[64.5,-1],
    barrier_flip_time=167,
    experiment_path=Path(r"JAL008_shelter_barrier_flip_1_2024_04_25T11_27_42"),
)

JAL8_flip2_29apr = Experiment(  # Mouse specific
    nick_name=mouse.nick_name,
    total_sessions=mouse.total_sessions,
    mouse_number_pyrat=mouse.mouse_number_pyrat,
    experiment_file_names=mouse.experiment_file_names,
    root_path=mouse.root_path,
    # Experiment specific
    experiment_name="flip",
    experiment_idx=4,
    experiment_date="2024_04_29",
    experiment_time="12_14_54",
    shelter_time=[.25,-1],
    barrier_time=[82.25,-1],
    barrier_flip_time=192.5,
    experiment_path=Path(r"JAL008_shelter_barrier_flip_2_2024_04_29T12_14_54"),
)

JAL8_tiny_3may = Experiment(  # Mouse specific
    nick_name=mouse.nick_name,
    total_sessions=mouse.total_sessions,
    mouse_number_pyrat=mouse.mouse_number_pyrat,
    experiment_file_names=mouse.experiment_file_names,
    root_path=mouse.root_path,
    # Experiment specific
    experiment_name="tiny",
    experiment_idx=5,
    experiment_date="2024_05_03",
    experiment_time="10_02_35",
    shelter_time=[.25, -1],
    barrier_time=[92.5, -1],
    barrier_flip_time=179.5,
    experiment_path=Path(r"JAL008_shelter_tiny_barrier_flip_1_2024_05_03T10_02_35"),
)

JAL8_flip4_10may = Experiment(  # Mouse specific
    nick_name=mouse.nick_name,
    total_sessions=mouse.total_sessions,
    mouse_number_pyrat=mouse.mouse_number_pyrat,
    experiment_file_names=mouse.experiment_file_names,
    root_path=mouse.root_path,
    # Experiment specific
    experiment_name="flip",
    experiment_idx=7,
    experiment_date="2024_05_10",
    experiment_time="11_47_47",
    shelter_time=[.25,-1],
    barrier_time=[89, 329.25],
    barrier_flip_time=213.25,
    experiment_path=Path(r"JAL008_shelter_barrier_flip_4_2024_05_10T11_47_47"),
)