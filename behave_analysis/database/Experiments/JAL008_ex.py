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
