"""
A settings class for the analyze_efizz.py script which is currently used to turn on or off the different models
"""

from behave_analysis.utils.settings_objects import Settings_analyze_efizz

Settings_ae = Settings_analyze_efizz(
    # ------------- General settings --------------------------
    stim_type="audio",  # 'audio', leave as 'None' if no stims were delivered
    linear_shift=True,  # whether to run linear shift!
    parallel_pool_linshit=True,  # if True, uses parallel pool to compute linear shift
    # This does not effect Tuned, this model needs linear shift to work
    redo_compute=True,  # if True it will force recompute any pre-saved analyses (e.g. Rayleigh and LDA)
    cluster_type="good",  # 'synthetic','synthetichdir','all','good',
    show_plots=False,
    # possible experimental condition inputs: 'all_time' (don't filter based on shelter or barrier),
    #                                         'pre_shelter' (empty arena),
    #                                         'shelter_present',
    #                                         'barrier_present'
    #                                         'shelter_only',
    #                                         'barrier_pre_flip',
    #                                         'barrier_post_flip',
    #                                         "barrier_removed"
    conditions=["shelter_only", "barrier_pre_flip", "barrier_post_flip"],
    user_defined_conditions=True,  # False if you want automatically identified conditions
    condition_types="experimental_conditions",  # if 'experimental_conditions' it uses conditions listed above that start with user
        # if 'time_conditions' it compares first vs second half
        # if 'behavioral_conditions' it defines the conditions based on homing/escape behaviour of mousie - it will overrule other condition settings
        # if 'homing_number_2' it defines the conditions before and after a certain number of correct homings
    compartment_split=["all"],  # ['all','threat_zone','shelter_compartment','left_arena','right_arena']
        # If 'all' it will run the model on all data, if 'threat_zone' it will only run on the threat zone data e.g
        # If 'by_position', it will compute LDA decoding by arena position
    number_of_bins=13,  # number of bin edges for angles, e.g. 13 or 19 are good numbers
    homings = "manual",  # 'manual' or 'auto' or 'auto_curated', which homings to load to video_df,
        # if 'manual' it will load the manually detected homings, if 'auto' it will load the automatically detected homings, if 'auto_curated' it will load the automatically detected homings that have been manually curated
    # ------------- Linear Shift stats settings --------------------------
    linshift_min_step=120,  # in frames, minimum shift to consider for linear shift stats
    linshift_step=80,  # in frames, step size for linear shift stats
    linshift_step_n=100,  # number of steps to do for linear shift stats
    # ------------- PCA model settings --------------------------
    redo_pca_preprocessing=False,  # rerun if you have changed, angles, conditions, or underlying neural data
    # ------------- LDA model settings --------------------------
    use_firing_rate=True,
    exclude_proximal=5,  # this determines how far the mouse has to be from each point for head angle point decoding, if 0 LDA uses all head angles regardless of distance to the target
    exclude_hdir=False,
    dropout=False,  # this will iteratively dropout each cluster and recompute the LDA prediction accuracy to see how much that cluster matters
    PCA_process=[],  # number of PCs to use, if left empty it will run without PCA
    subsampling=False,  # whether to subsample to equalize data by angles and space
    min_speed_threshold=1,  # mouse must be moving > 1cm/s, currently only works for experimental conditions (update filtering functions if you want to use with other settings)
    # ------------ Rayleigh model settings ----------------------
    rayleigh_significance="linshit",  # "linshit" or "bootstrap"
    single_cluster_plots=True,  # True: Plot every condition in one figure
    # False: Do not plot every condition in one figure for each cluster
    multi_cluster_plots=False,  # True: Plot every cluster in one figure for one condition
    # False: Do not plot every cluster in one figure
    # ------------ Escape Pattern settings ----------------------
    ep_bins=25,  # number of bins for escape pattern tuning
    # ------------ Replay model settings ----------------------
    replay_cells="all",  # 'all','hdir','escape_tuned'
    replay_template_variable="escape",  # to make the order template of the replay sequence
    replay_decoder_variable="escape",  # 'shelter_dist' or 'escape' or 'speed' or '2D_position
    replay_train_condition="barrier_pre_flip",  # "shelter_only", "barrier_pre_flip", "barrier_post_flip"
    replay_test_condition="barrier_pre_flip",  # "shelter_only", "barrier_pre_flip", "barrier_post_flip"
    replay_decoder_train_time_period="correct_full_homing&escape",  # 'homing&escape', "correct_<>", "error_<>", "full_<>"
    replay_decoder_test_time_period="homing&escape",  #  'error_homing&escape', 'before_homing','in_shelter_after_escape','outside_shelter','stationary_outside_shelter','in_shelter'
    replay_template_match_method="SS_decoder",  # 'rank_order_corr' or 'bayesian_decoder' or 'SS_decoder'
    # ------------ Place Cells settings ----------------------
    place_cell_bin_size_pix=50,  # in pix (10cm/pix) at least 50
    place_cell_speed_threshold=2.5,  # in cm/s, threshold for excluding time points when the mouse is stationary or moving very slowly
    place_cell_smoothing_sigma=2.0,  # in bins, for smoothing the spike count and occupancy maps before computing rate maps
    place_cell_min_occupancy=0.5,  # in seconds, minimum occupancy time for a bin to be included in the analysis
    # ------------ CCA settings ----------------------
    # list of behavioral variables to include in CCA, e.g. ["speed", "hdir_velocity", "distance_to_shelter"]
    cca_behavioral_vars=["hdir", "hdir_velocity", "mouse_x_position", "mouse_y_position", "speed", "acceleration", "hsa", "h_preflipbar_a", "h_postflipbar_a", "distance_to_shelter", "distance_to_barrier1", "distance_to_barrier2"],
    cca_n_components=5,  # number of CCA components to compute
    cca_test_sets=["shelter_outing", "homing&escape"],  # list of test sets to use for CCA, e.g. ["explore_test", "homing", "escape"]
    cca_train_set= "explore",  # name of training set to use for CCA, e.g. "explore", "homing&escape", "shelter_outing"
    cca_xval_method="match_pos_shelter_outing",  # method for splitting data into train and test sets, e.g. "random split", "half", "match_pos_homings", "match_pos_hdir_homings"

)
