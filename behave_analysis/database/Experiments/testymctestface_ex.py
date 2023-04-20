from behave_analysis.database.Experiments.experiment_class import Experiment
from behave_analysis.database.Mice.testMouse import Testy_mctestface
from pathlib import Path

testbonsaipulse = Experiment(  # Mouse specific
    nick_name=Testy_mctestface.nick_name,
    total_sessions=Testy_mctestface.total_sessions,
    mouse_number_pyrat=Testy_mctestface.mouse_number_pyrat,
    experiment_file_names=Testy_mctestface.experiment_file_names,
    root_path=Testy_mctestface.root_path,
    # Experiment specific
    experiment_name="Test bonsai pulses",
    experiment_idx=0,
    experiment_date="y",
    experiment_time="x",
    experiment_path=Path(r"666_pulsetest1_2023_04_19T18_04_43"),
    shelter_time=[],
    barrier_time=[],
)

testbonsaipulse2withefizz = Experiment(  # Mouse specific
    nick_name=Testy_mctestface.nick_name,
    total_sessions=Testy_mctestface.total_sessions,
    mouse_number_pyrat=Testy_mctestface.mouse_number_pyrat,
    experiment_file_names=Testy_mctestface.experiment_file_names,
    root_path=Testy_mctestface.root_path,
    # Experiment specific
    experiment_name="Test bonsai pulses with efizz",
    experiment_idx=0,
    experiment_date="y",
    experiment_time="x",
    experiment_path=Path(r"666_pulsetest2_2023_04_19T18_42_44"),
    shelter_time=[],
    barrier_time=[],
)

test_NEWgate = Experiment(  # Mouse specific
    nick_name=Testy_mctestface.nick_name,
    total_sessions=Testy_mctestface.total_sessions,
    mouse_number_pyrat=Testy_mctestface.mouse_number_pyrat,
    experiment_file_names=Testy_mctestface.experiment_file_names,
    root_path=Testy_mctestface.root_path,
    # Experiment specific
    experiment_name="new gate test",
    experiment_idx=0,
    experiment_date="y",
    experiment_time="x",
    experiment_path=Path(r"666_gatetest2_2023_04_19T19_24_53"),
    shelter_time=[],
    barrier_time=[],
)

laserTestShort = Experiment(# Mouse specific
                    nick_name = Testy_mctestface.nick_name,
                    total_sessions = Testy_mctestface.total_sessions,
                    mouse_number_pyrat = Testy_mctestface.mouse_number_pyrat,
                    experiment_file_names = Testy_mctestface.experiment_file_names,
                    root_path = Testy_mctestface.root_path,

                    # Experiment specific
                    experiment_name = 'Laser test Jazz',
                    experiment_idx = 666,
                    experiment_date = "2023_04_11",
                    experiment_time = "12_52_28",
                    experiment_path = Path(r"999_laserTestJAZZ_2023_04_11T12_52_28"),
                    shelter_time = [],
                    barrier_time = [])

laserTest = Experiment(# Mouse specific
                    nick_name = Testy_mctestface.nick_name,
                    total_sessions = Testy_mctestface.total_sessions,
                    mouse_number_pyrat = Testy_mctestface.mouse_number_pyrat,
                    experiment_file_names = Testy_mctestface.experiment_file_names,
                    root_path = Testy_mctestface.root_path,

                    # Experiment specific
                    experiment_name = 'Laser test Jazz Hands 2',
                    experiment_idx = 999,
                    experiment_date = "2023_04_11",
                    experiment_time = "15_51_58",
                    experiment_path = Path(r"999_laserTestJAZZhands2_2023_04_11T15_51_58"),
                    shelter_time = [],
                    barrier_time = [])

test = Experiment(# Mouse specific
                    nick_name = Testy_mctestface.nick_name,
                    total_sessions = Testy_mctestface.total_sessions,
                    mouse_number_pyrat = Testy_mctestface.mouse_number_pyrat,
                    experiment_file_names = Testy_mctestface.experiment_file_names,
                    root_path = Testy_mctestface.root_path,

                    # Experiment specific
                    experiment_name = 'test',
                    experiment_idx = 4,
                    experiment_date = "2023_03_22",
                    experiment_time = "09_58_59",
                    experiment_path = Path(r"001_synctestmanyaudio_2023_03_22T09_58_59"),
                    shelter_time = [],
                    barrier_time = [])