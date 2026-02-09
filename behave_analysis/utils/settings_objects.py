"""The following script contains several dataclasses that outline the fields of several settings data classes
used within the pipeline. Each data class outlines the structure or blueprint of the settings object.
As such each class below is just a shell"""

from dataclasses import dataclass
import numpy as np

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
    # homing settings
    fast_speed: float = 10  # 15 cm/s seems quite slow no? In the paper philip used 10 cm/s
    min_frames_between_trials: int = 40  # 1 second between trials to stop double counting split homings
    edge_proximity: int = 100  # 10 pixels is 1 cm, so 100 pixels is 10 cm. Defining subgoal start homings
    fast_angular_speed: float = np.pi / 2  # Turning towards some reference location speed in rad/s with a threshold of 60 degrees
    padding_duration: float = 0.5  # How long should the box car filter be in seconds and how long should a homing event be
    min_change_in_dist_to_shelter: float = 0.3  # (maybe %) How far does the mouse have to move towards the shelter to be considered a homing event
    max_time_within_session: float = 2000  # How long is session in minutes - Ignore I think # TODO remove?
    threat_area_width: int = 820
    cum_threshold: int = 25  # How many cm does the mouse have to move when considering homing angle
    speed_threshold: int = 15 # this determined when the mouse has actually srated running (after the initial head turn) 
    threat_area_height: int = 275
    by_session: bool = True
    # sessions: list
    all_sessions: bool = False
    redo_homings: bool = False # TODO remove!
    use_boris: bool = True
    # escape_settings
    stim_type: str = "audio"
    response_thresh: int = 5


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
    epoch_num: int = 6 # number of epochs for cross validation
    number_of_bins: int = 19
    use_firing_rate: bool = True
    discriminant_type: str = "linear" # 'linear' or 'quadratic' (or 'LSTM', not implemented yet)
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
    # Replay settings
    replay_cells: str = "escape_untuned"  # 'all','hdir','escape_tuned'
    replay_search_window: int = 500  # in ms
    replay_decoder_variable: str = "escape"  # 'shelter_dist' or 'escape'
    replay_train_condition: str = "barrier_pre_flip" # "shelter_only", "barrier_pre_flip", "barrier_post_flip"
    replay_test_condition: str = "barrier_pre_flip" # "shelter_only", "barrier_pre_flip", "barrier_post_flip"
    replay_decoder_train_time_period: str = "correct_homing&escape" # 'homing&escape'
    replay_decoder_test_time_period: str = "error_homing&escape"  # 'before_homing','in_shelter_after_escape','outside_shelter','stationary_outside_shelter','in_shelter'
    replay_rank_order_corr_method: str = "first_activity" # 'first_activity', 'weighted_avg'
    occupancy_prior: str = "uniform"  # 'uniform' or 'empirical'
    replay_template_match_method: str = "SS_decoder"  # 'rank_order_corr' or 'bayesian_decoder' or 'state_space_decoder'
    replay_state_space_decoder_bin_size: float = .001  # in seconds, default 1ms

@dataclass(frozen=True)
class Settings_postprocess:
    cluster_type: str = ""
    efizz: bool = False
    homings: bool = False
    regenerate_synthetic_data: bool = False
