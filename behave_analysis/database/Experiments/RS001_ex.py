from behave_analysis.database.Experiments.experiment_class import Experiment
from behave_analysis.database.Mice.AllMouses import RS001 as mouse
from pathlib import Path

RS001_shelt_14sep26 = Experiment(  # Mouse specific
    nick_name=mouse.nick_name,
    total_sessions=mouse.total_sessions,
    mouse_number_pyrat=mouse.mouse_number_pyrat,
    experiment_file_names=mouse.experiment_file_names,
    root_path=mouse.root_path,
    # Experiment specific
    experiment_name="shelt",
    experiment_idx=1,
    experiment_date="2026_09_14",
    experiment_time="13_53_13",
    valid_time=[3.75, -1],
    shelter_time=[3.75, -1],
    barrier_time=[],
    barrier_flip_time=None,
    experiment_path=Path(r"RS001_sesh1_2026_09_14T13_53_13_cam"),
    recording_script="JAL",
)

RS001_cran_21sep26 = Experiment(  # Mouse specific
    nick_name=mouse.nick_name,
    total_sessions=mouse.total_sessions,
    mouse_number_pyrat=mouse.mouse_number_pyrat,
    experiment_file_names=mouse.experiment_file_names,
    root_path=mouse.root_path,
    # Experiment specific
    experiment_name="shelt",
    experiment_idx=1,
    experiment_date="2026_09_21",
    experiment_time="09_17_19",
    valid_time=[.3, -1],
    shelter_time=[.3, -1],
    barrier_time=[],
    barrier_flip_time=None,
    experiment_path=Path(r"RS001_shelt_bait_2026_09_21T09_17_19"),
    recording_script="YT",
)

RS001_cran_21sep26_2 = Experiment(  # Mouse specific
    nick_name=mouse.nick_name,
    total_sessions=mouse.total_sessions,
    mouse_number_pyrat=mouse.mouse_number_pyrat,
    experiment_file_names=mouse.experiment_file_names,
    root_path=mouse.root_path,
    # Experiment specific
    experiment_name="shelt",
    experiment_idx=1,
    experiment_date="2026_09_21",
    experiment_time="09_45_08",
    valid_time=[0, -1],
    shelter_time=[0, -1],
    barrier_time=[],
    barrier_flip_time=None,
    experiment_path=Path(r"RS001_shelt_bait_2026_09_21T09_45_08"),
    recording_script="YT",
)