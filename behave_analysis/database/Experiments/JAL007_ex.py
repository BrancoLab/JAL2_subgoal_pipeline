from behave_analysis.database.Experiments.experiment_class import Experiment
from behave_analysis.database.Mice.AllMouses import JAL007 as mouse
from pathlib import Path

JAL7_hab_1mar = Experiment(  # Mouse specific
    nick_name=mouse.nick_name,
    total_sessions=mouse.total_sessions,
    mouse_number_pyrat=mouse.mouse_number_pyrat,
    experiment_file_names=mouse.experiment_file_names,
    root_path=mouse.root_path,
    # Experiment specific
    experiment_name="habituation",
    experiment_idx=0,
    experiment_date="2024_03_01",
    experiment_time="14_13_42",
    shelter_time=[],
    barrier_time=[],
    barrier_flip_time=None,
    experiment_path=Path(r"JAL007_Habituation_2024_03_01T14_13_42"),
)
