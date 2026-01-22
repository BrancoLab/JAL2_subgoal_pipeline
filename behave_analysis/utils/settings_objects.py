"""The following script contains several dataclasses that outline the fields of several settings data classes
used within the pipeline. Each data class outlines the structure or blueprint of the settings object.
As such each class below is just a shell"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings_process:
    create_new_registration: bool = False
    registration: str = "partial affine"
    fisheye_correction_file: str = None
    size: int = 1024
    pixels_per_cm: int = 10
    by_experiment: bool = False
    experiments: list = None
    by_session: bool = False
    sessions: list = None
    all_sessions: bool = False
    efizz: bool = False
    efizzDataPath: str = None
    radius: int = 460


@dataclass(frozen=True)
class Settings_track:
    inverse_fisheye_correction_file: str = None
    redo_processing_step: bool = False
    skip_processing_step: bool = False
    display_tracking_output: bool = False
    min_confidence_in_tracking: float = None
    max_deviation_from_rest_of_points: int = None
    by_experiment: bool = False
    experiments: list = None
    by_session: bool = False
    sessions: list = None
    all_sessions: bool = False
    tracking_file_location: str = None
    save_labeled_video: bool = False
    random_points: str = None


@dataclass(frozen=True)
class Settings_homings:
    fast_speed: float
    min_frames_between_trials: int
    edge_proximity: int
    fast_angular_speed: float
    padding_duration: float
    min_change_in_dist_to_shelter: float
    max_time_within_session: float
    threat_area_width: int
    cum_threshold: int
    speed_threshold: int
    threat_area_height: int
    # subgoal_locations: list
    # duration_after_crossing: float
    # by_experiment: bool = False
    # experiments: list = None
    by_session: bool = False
    sessions: list = None
    all_sessions: bool = False
    redo_homings: bool = False
    use_boris: bool = True


@dataclass(frozen=True)
class Settings_visualize:
    escape_trials: bool = True
    homing_trials: bool = False
    # t_xing_trials: bool=True
    # explore_trial: bool=False
    display_tracking: bool = False
    display_trail: bool = True
    rapid: bool = True
    display_stimulus: bool = True
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
    by_experiment: bool = False
    by_session: bool = False
    sessions: list = None
    all_sessions: bool = False
    efizz: bool = False
    show_plots: bool = False
    cluster_type: str = ""
    stim_type: str = ""
    conditions: list = None
    user_defined_conditions: bool = False
    compartment_split: bool = False
    condition_types: str = ""


@dataclass(frozen=True)
class Settings_analyze_behave:
    stim_type: str = "None"
    show_plots: bool = False


@dataclass(frozen=True)
class Settings_analyze_efizz:
    # gen settings
    stim_type: str = "None"
    redo_compute: bool = False
    cluster_type: str = ""
    show_plots: bool = False
    condition_types: str = ""
    compartment_split: str = ""
    parallel_pool_linshit: bool = True
    # LDA settings
    epoch_num: int = 6
    number_of_bins: int = 19
    use_firing_rate: bool = True
    discriminant_type: str = ""
    PCA_process: int = 15
    exclude_proximal: bool = False
    exclude_hdir: bool = False
    dropout: bool = False
    subsampling: bool = False
    exclude_stationary: bool = False
    # rayleigh settings
    rayleigh_significance: str = ""
    single_cluster_plots: bool = True
    multi_cluster_plots: bool = False
    linear_shift: bool = True
    conditions: str = ""
    user_defined_conditions: bool = False
    analyze_only_the_period_before_shelter: bool = False
    # PCA settings
    redo_pca_preprocessing: bool = False
    # Escape Pattern settings
    escape_tuning_bins: int = 25
    escape_pattern_no_stationary: bool = False
    escape_pattern_interpolation_mult: int = 2
    ep_linshift_min_step = 120  # in seconds
    ep_linshift_step = 400  # in seconds
    ep_linshift_step_n = 100  # number of steps to do
    ep_linshift_min_homings = 5  # minimum number of homings in the central third of each condition for linear shift stats
    ep_gaussian_fitting: bool = False
    ep_compute_loo_reliability: bool = False
    ep_tuned_compare_method: str = "euclidean" # or 'cosine'
    ep_tuned_stats: str = "bootstrap"  # or 'linear_shift' (not yet implemented)
    ep_tuned_stats_samples: int = 100


@dataclass(frozen=True)
class Settings_postprocess:
    cluster_type: str = ""
    efizz: bool = False
    homings: bool = False
    response_thresh: int = 5
    regenerate_synthetic_data: bool = False
