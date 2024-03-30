from behave_analysis.database.Experiments.experiment_class import Experiment
from behave_analysis.database.Mice.AllMouses import JAL006 as mouse
from pathlib import Path

JAL6_hab_1mar = Experiment(  # Mouse specific
    nick_name=mouse.nick_name,
    total_sessions=mouse.total_sessions,
    mouse_number_pyrat=mouse.mouse_number_pyrat,
    experiment_file_names=mouse.experiment_file_names,
    root_path=mouse.root_path,
    # Experiment specific
    experiment_name="habituation",
    experiment_idx=0,
    experiment_date="2024_03_01",
    experiment_time="13_02_08",
    shelter_time=[],
    barrier_time=[],
    barrier_flip_time=None,
    experiment_path=Path(r"JAL006_Habituation_2024_03_01T13_02_08"),
)

JAL6_flip6 = Experiment(  # Mouse specific
    nick_name=mouse.nick_name,
    total_sessions=mouse.total_sessions,
    mouse_number_pyrat=mouse.mouse_number_pyrat,
    experiment_file_names=mouse.experiment_file_names,
    root_path=mouse.root_path,
    # Experiment specific
    experiment_name="flip 6 perfect",
    experiment_idx=0,
    experiment_date="2024_03_28",
    experiment_time="10_54_20",
    shelter_time=[0, -1],
    barrier_time=[60.26, 281.18],
    barrier_flip_time=236,
    experiment_path=Path(r"JAL006_shelter_barrier_flip_6_2024_03_28T10_54_20"),
)
