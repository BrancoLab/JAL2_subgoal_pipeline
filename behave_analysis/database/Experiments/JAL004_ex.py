from pathlib import Path

from behave_analysis.database.Experiments.experiment_class import Experiment
from behave_analysis.database.Mice.AllMouses import JAL004

JAL4_shelt_17aug = Experiment(  # Mouse specific
    nick_name=JAL004.nick_name,
    total_sessions=JAL004.total_sessions,
    mouse_number_pyrat=JAL004.mouse_number_pyrat,
    experiment_file_names=JAL004.experiment_file_names,
    root_path=JAL004.root_path,
    # Experiment specific
    experiment_name="shelter",
    experiment_idx=0,
    experiment_date="2023_08_17",
    experiment_time="13_41_44",
    shelter_time=[60.75, -1],
    barrier_time=[],
    barrier_flip_time=None,
    experiment_path=Path(r"004_baseline_2023_08_17T13_41_44"),
)

JAL4_mush_18aug = Experiment(  # Mouse specific
    nick_name=JAL004.nick_name,
    total_sessions=JAL004.total_sessions,
    mouse_number_pyrat=JAL004.mouse_number_pyrat,
    experiment_file_names=JAL004.experiment_file_names,
    root_path=JAL004.root_path,
    # Experiment specific
    experiment_name="mushroom",
    experiment_idx=0,
    experiment_date="2023_08_18",
    experiment_time="08_22_16",
    shelter_time=[119.85, -1],
    barrier_time=[],
    barrier_flip_time=None,
    experiment_path=Path(r"004_mushroom_2023_08_18T08_22_16"),
)

JAL4_flip1_21Aug = Experiment(  # Mouse specific
    nick_name=JAL004.nick_name,
    total_sessions=JAL004.total_sessions,
    mouse_number_pyrat=JAL004.mouse_number_pyrat,
    experiment_file_names=JAL004.experiment_file_names,
    root_path=JAL004.root_path,
    # Experiment specific
    experiment_name="barrierflip",
    experiment_idx=1,
    experiment_date="2023_08_21",
    experiment_time="12_53_10",
    shelter_time=[.75, -1],
    barrier_time=[119.6, -1],
    barrier_flip_time=227.25,
    experiment_path=Path(r"004_flip_2023_08_21T12_53_10"),
)

JAL4_mush2_22Aug = Experiment(  # Mouse specific
    nick_name=JAL004.nick_name,
    total_sessions=JAL004.total_sessions,
    mouse_number_pyrat=JAL004.mouse_number_pyrat,
    experiment_file_names=JAL004.experiment_file_names,
    root_path=JAL004.root_path,
    # Experiment specific
    experiment_name="mushroom",
    experiment_idx=2,
    experiment_date="2023_08_22",
    experiment_time="13_13_41",
    shelter_time=[72, 9496], # mouse jumped off
    barrier_time=[],
    barrier_flip_time=None,
    experiment_path=Path(r"004_mush1_2023_08_22T13_13_41"),
)

JAL4_flip3_28aug = Experiment(  # Mouse specific
    nick_name=JAL004.nick_name,
    total_sessions=JAL004.total_sessions,
    mouse_number_pyrat=JAL004.mouse_number_pyrat,
    experiment_file_names=JAL004.experiment_file_names,
    root_path=JAL004.root_path,
    # Experiment specific
    experiment_name="flip",
    experiment_idx=3,
    experiment_date="2023_08_28",
    experiment_time="09_36_04",
    shelter_time=[0.25, -1],
    barrier_time=[84.25, -1],  # seconds needed to be more precise
    barrier_flip_time=159,
    experiment_path=Path(r"JAL004_flip_rotated_2023_08_28T09_36_04"),
)

JAL4_flip4_3Sept = Experiment(  # Mouse specific
    nick_name=JAL004.nick_name,
    total_sessions=JAL004.total_sessions,
    mouse_number_pyrat=JAL004.mouse_number_pyrat,
    experiment_file_names=JAL004.experiment_file_names,
    root_path=JAL004.root_path,
    # Experiment specific
    experiment_name="flip",
    experiment_idx=4,
    experiment_date="2023_09_03",
    experiment_time="12_04_16",
    shelter_time=[0.33, -1],
    barrier_time=[54.75, -1],
    barrier_flip_time=171,
    experiment_path=Path(r"004_flip_2023_09_03T12_04_16"),
)

JAL4_flip5_11Sept = Experiment(  # Mouse specific
    nick_name=JAL004.nick_name,
    total_sessions=JAL004.total_sessions,
    mouse_number_pyrat=JAL004.mouse_number_pyrat,
    experiment_file_names=JAL004.experiment_file_names,
    root_path=JAL004.root_path,
    # Experiment specific
    experiment_name="flip",
    experiment_idx=5,
    experiment_date="2023_09_11",
    experiment_time="09_32_25",
    shelter_time=[0.33, -1],
    barrier_time=[67.08, -1],  # seconds needed to be more precise
    barrier_flip_time=233.25,
    experiment_path=Path(r"004_flip_puff2_2023_09_11T09_32_25"),
)

JAL4_flip6_19Sept = Experiment(  # Mouse specific
    nick_name=JAL004.nick_name,
    total_sessions=JAL004.total_sessions,
    mouse_number_pyrat=JAL004.mouse_number_pyrat,
    experiment_file_names=JAL004.experiment_file_names,
    root_path=JAL004.root_path,
    # Experiment specific
    experiment_name="flip",
    experiment_idx=6,
    experiment_date="2023_09_19",
    experiment_time="14_10_56",
    shelter_time=[0.6, -1],
    barrier_time=[56.36, 274.14],  # seconds needed to be more precise
    barrier_flip_time=160,
    experiment_path=Path(r"004_flipppuf19sept_2023_09_19T14_10_56"),
)

