from dataclasses import dataclass

@dataclass(frozen=True)
class Settings_process:
    create_new_registration: bool=False
    skip_registration: bool=True
    registration: str='partial affine'
    fisheye_correction_file: str=None
    size: int=1024
    pixels_per_cm : int=10
    by_experiment: bool=False
    experiments: list=None
    by_session: bool=False
    sessions: list=None
    all_sessions: bool=False

@dataclass(frozen=True)
class Settings_track:
    dlc_settings_file: str = None
    inverse_fisheye_correction_file: str = None
    redo_processing_step: bool=False
    skip_processing_step: bool=False
    display_tracking_output: bool=False
    min_confidence_in_tracking: float=None
    max_deviation_from_rest_of_points: int=None
    by_experiment: bool=False
    experiments: list=None
    by_session: bool=False
    sessions: list=None
    all_sessions: bool=False

@dataclass(frozen=True)
class Settings_homings:
    fast_speed: float
    fast_angular_speed: float
    padding_duration: float
    min_change_in_dist_to_shelter: float
    max_time_within_session: float
    threat_area_width: int
    threat_area_height: int
    subgoal_locations: list
    duration_after_crossing: float
    by_experiment: bool=False
    experiments: list=None
    by_session: bool=False
    sessions: list=None
    all_sessions: bool=False

@dataclass(frozen=True)
class Settings_visualize:
    laser_trials: bool=True
    escape_trials: bool=True
    homing_trials: bool=False
    t_xing_trials: bool=True
    display_tracking:bool=False
    display_trail:bool=True
    rapid: bool=True
    display_stimulus: bool=True
    seconds_before_audio: int = 3
    seconds_before_laser: int = 3
    seconds_before_homing: int = 3
    seconds_before_threshold_crossing: int = 3
    seconds_after_audio: int = 2
    seconds_after_laser: int = 6
    seconds_after_homing: int = 3
    seconds_after_threshold_crossing: int = 3
    save_folder: str = None
    fisheye_correction_file: str = None
    by_experiment: bool=False
    experiments: list=None
    by_session: bool=False
    sessions: list=None
    all_sessions: bool=False

@dataclass(frozen=True)
class Settings_analyze_local:
    plot_escape: bool=False
    plot_laser: bool=False
    plot_targets: bool=False
    plot_homings: bool=False
    plot_t_xings: bool=False
    plot_trial: bool=False
    plot_homing: bool=False
    title: str=None
    save_folder: str=None
    experiments: list = None
    sessions: list = None
    group_1: list=None
    group_2: list=None
    group_3: list=None
    group_4: list=None
    group_5: list=None
    group_6: list=None
    group_7: list=None
    group_8: list=None
    by_experiment: bool=True
    by_session: bool=False
    compare: bool=False
    all_sessions: bool=False

@dataclass(frozen=True)
class Settings_analyze_global:
    analysis: Settings_analyze_local=None
    max_num_trials: int = 6
    max_escape_duration: int = 9
    post_laser_seconds_to_plot: int = 5
    min_distance_from_shelter: int = 10
    escape_initiation_speed: float = 20
    edge_vector_threshold: float = 0.68
    binarize_statistics: bool = False
    two_tailed_test: bool=True
    leftside_only: bool=False
    rightside_only: bool=False
    reflect_trajectories: bool=False
    save_folder: str=None
    color_by: str=''
    x_jitter: bool=True