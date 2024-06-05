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

JAL7_empty_shelter_5mar = Experiment(  # Mouse specific
    nick_name=mouse.nick_name,
    total_sessions=mouse.total_sessions,
    mouse_number_pyrat=mouse.mouse_number_pyrat,
    experiment_file_names=mouse.experiment_file_names,
    root_path=mouse.root_path,
    # Experiment specific
    experiment_name="empty_shelter",
    experiment_idx=0,
    experiment_date="2024_03_05",
    experiment_time="13_45_47",
    shelter_time=[91.17, -1],
    barrier_time=[],
    barrier_flip_time=None,
    experiment_path=Path(r"JAL007_empty_shelter_2024_03_05T13_45_47"),
)

JAL7_flip2_12mar = Experiment(  # Mouse specific
    nick_name=mouse.nick_name,
    total_sessions=mouse.total_sessions,
    mouse_number_pyrat=mouse.mouse_number_pyrat,
    experiment_file_names=mouse.experiment_file_names,
    root_path=mouse.root_path,
    # Experiment specific
    experiment_name="sesh8",
    experiment_idx=0,
    experiment_date="2024_03_12",
    experiment_time="11_18_26",
    shelter_time=[.25, -1],
    barrier_time=[59.5, -1],
    barrier_flip_time= 189,
    experiment_path=Path(r"JAL007_barrierflip2_2024_03_12T11_18_26"),
)

JAL7_flip5_22mar = Experiment(  # Mouse specific
    nick_name=mouse.nick_name,
    total_sessions=mouse.total_sessions,
    mouse_number_pyrat=mouse.mouse_number_pyrat,
    experiment_file_names=mouse.experiment_file_names,
    root_path=mouse.root_path,
    # Experiment specific
    experiment_name="sesh8",
    experiment_idx=0,
    experiment_date="2024_03_22",
    experiment_time="11_15_43",
    shelter_time=[.5, -1],
    barrier_time=[60.75, -1],
    barrier_flip_time=214.5,
    experiment_path=Path(r"JAL007_shelter_barrier_flip_5_2024_03_22T11_15_43"),
)

JAL7_sesh8_9apr = Experiment(  # Mouse specific
    nick_name=mouse.nick_name,
    total_sessions=mouse.total_sessions,
    mouse_number_pyrat=mouse.mouse_number_pyrat,
    experiment_file_names=mouse.experiment_file_names,
    root_path=mouse.root_path,
    # Experiment specific
    experiment_name="sesh8",
    experiment_idx=0,
    experiment_date="2024_04_09",
    experiment_time="10_07_45",
    shelter_time=[0, -1],
    barrier_time=[116.5, -1],
    barrier_flip_time=227,
    experiment_path=Path(r"JAL007_shelter_barrier_flip_8_2024_04_09T10_07_45"),
)

JAL7_sesh9_16apr = Experiment(  # Mouse specific
    nick_name=mouse.nick_name,
    total_sessions=mouse.total_sessions,
    mouse_number_pyrat=mouse.mouse_number_pyrat,
    experiment_file_names=mouse.experiment_file_names,
    root_path=mouse.root_path,
    # Experiment specific
    experiment_name="sesh8",
    experiment_idx=0,
    experiment_date="2024_04_16",
    experiment_time="11_13_05",
    shelter_time=[3.75, -1],
    barrier_time=[73.5, -1],
    barrier_flip_time=200,
    experiment_path=Path(r"JAL007_shelter_barrier_flip_9_2024_04_16T11_13_05"),
)

JAL7_23apr = Experiment(  # Mouse specific
    nick_name=mouse.nick_name,
    total_sessions=mouse.total_sessions,
    mouse_number_pyrat=mouse.mouse_number_pyrat,
    experiment_file_names=mouse.experiment_file_names,
    root_path=mouse.root_path,
    # Experiment specific
    experiment_name="sesh8",
    experiment_idx=0,
    experiment_date="2024_04_23",
    experiment_time="09_59_40",
    shelter_time=[.25, -1],
    barrier_time=[73.5, -1],
    barrier_flip_time=196.3,
    experiment_path=Path(r"JAL007_shelter_barrier_flip_100_2024_04_23T09_59_40"),
)

JAL7_30apr = Experiment(  # Mouse specific
    nick_name=mouse.nick_name,
    total_sessions=mouse.total_sessions,
    mouse_number_pyrat=mouse.mouse_number_pyrat,
    experiment_file_names=mouse.experiment_file_names,
    root_path=mouse.root_path,
    # Experiment specific
    experiment_name="sesh8",
    experiment_idx=0,
    experiment_date="2024_04_30",
    experiment_time="10_57_04",
    shelter_time=[.25, -1],
    barrier_time=[63, -1],
    barrier_flip_time=174.5,
    experiment_path=Path(r"JAL007_tinnybarrier1_2024_04_30T10_57_04"),
)