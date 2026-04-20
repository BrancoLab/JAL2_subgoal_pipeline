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

JAL6_shelt_4mar = Experiment(  # Mouse specific
    nick_name=mouse.nick_name,
    total_sessions=mouse.total_sessions,
    mouse_number_pyrat=mouse.mouse_number_pyrat,
    experiment_file_names=mouse.experiment_file_names,
    root_path=mouse.root_path,
    experiment_name="empty_shelter",
    experiment_idx=0,
    experiment_date="2024_03_04",
    experiment_time="11_24_29",
    shelter_time=[67.5, -1],
    barrier_time=[67.5, -1], # Testing a way to introduce a barrier time when there is none
    barrier_flip_time=None,
    experiment_path=Path(r"JAL006_empty_shelter_2024_03_04T11_24_29"),
)

JAL6_28mar = Experiment(  # Mouse specific
    nick_name=mouse.nick_name,
    total_sessions=mouse.total_sessions,
    mouse_number_pyrat=mouse.mouse_number_pyrat,
    experiment_file_names=mouse.experiment_file_names,
    root_path=mouse.root_path,
    # Experiment specific
    experiment_name="flip",
    experiment_idx=6,
    experiment_date="2024_03_28",
    experiment_time="10_54_20",
    shelter_time=[0.25, -1],
    barrier_time=[60.26, 281.18],
    barrier_flip_time=176,
    experiment_path=Path(r"JAL006_shelter_barrier_flip_6_2024_03_28T10_54_20"),
)

JAL6_flip3_18mar = Experiment(  # Mouse specific
    nick_name=mouse.nick_name,
    total_sessions=mouse.total_sessions,
    mouse_number_pyrat=mouse.mouse_number_pyrat,
    experiment_file_names=mouse.experiment_file_names,
    root_path=mouse.root_path,
    # Experiment specific
    experiment_name="flip",
    experiment_idx=3,
    experiment_date="2024_03_18",
    experiment_time="11_53_29",
    shelter_time=[0.25, -1],
    barrier_time=[68.25, -1],
    barrier_flip_time=180.25,
    experiment_path=Path(r"JAL006_barrier_flip2_2024_03_18T11_53_29"),
)

JAL6_flip4_21mar = Experiment(  # Mouse specific
    nick_name=mouse.nick_name,
    total_sessions=mouse.total_sessions,
    mouse_number_pyrat=mouse.mouse_number_pyrat,
    experiment_file_names=mouse.experiment_file_names,
    root_path=mouse.root_path,
    # Experiment specific
    experiment_name="flip",
    experiment_idx=4,
    experiment_date="2024_03_21",
    experiment_time="11_20_34",
    shelter_time=[0.25, -1],
    barrier_time=[59.25, -1],
    barrier_flip_time=172.5,
    experiment_path=Path(r"JAL006_shelter_barrier_flip_3_2024_03_21T11_20_34"),
)

JAL6_flip5_25mar = Experiment(  # Mouse specific
    nick_name=mouse.nick_name,
    total_sessions=mouse.total_sessions,
    mouse_number_pyrat=mouse.mouse_number_pyrat,
    experiment_file_names=mouse.experiment_file_names,
    root_path=mouse.root_path,
    # Experiment specific
    experiment_name="flip",
    experiment_idx=5,
    experiment_date="2024_03_25",
    experiment_time="11_05_33",
    shelter_time=[0.25, -1],
    barrier_time=[61.5, -1],
    barrier_flip_time=169.75,
    experiment_path=Path(r"JAL006_shelter_barrier_flip_5_2024_03_25T11_05_33"),
)

JAL6_flip7_1apr = Experiment(  # Mouse specific
    nick_name=mouse.nick_name,
    total_sessions=mouse.total_sessions,
    mouse_number_pyrat=mouse.mouse_number_pyrat,
    experiment_file_names=mouse.experiment_file_names,
    root_path=mouse.root_path,
    # Experiment specific
    experiment_name="flip",
    experiment_idx=7,
    experiment_date="2024_04_01",
    experiment_time="11_24_46",
    shelter_time=[0, -1],
    barrier_time=[77, -1],
    barrier_flip_time=189.25,
    experiment_path=Path(r"JAL006_shelter_barrier_flip_7_take2_2024_04_01T11_24_46"),
)
