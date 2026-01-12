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
    cluster_type=["good"],  # ['synthetic','synthetichdir','all','good'], # Can list multiple!
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
    # NOTE - number of bin edges NOT number of bins - need to refactor this
    number_of_bins=13,  # number of bins for angles, e.g. 13 or 19 are good numbers
    # ------------- PCA model settings --------------------------
    redo_pca_preprocessing=False,  # rerun if you have changed, angles, conditions, or underlying neural data
    # ------------- LDA model settings --------------------------
    run_LDA="all_angles",  # if [] it will not run LDA
    # if 'all_angles', 'all_distance','all_vectors' it will run it for all possible angles, distances, vectors
    # else:  list of angles ['hsa','hdir','h_postflipbar_a','h_preflipbar_a','h_bar_centre_a', 'randP']
    # TODO: linear shift doesn't curently work for vect or dist because of binning and other inputs neede in linear_discriminant_analysis function
    epoch_num=6,  # number of epochs for cross validation
    use_firing_rate=True,
    discriminant_type="linear",  # 'linear' or 'quadratic' or 'LSTM'
    exclude_proximal=5,  # this determines how far the mouse has to be from each point for head angle point decoding, if 0 LDA uses all head angles regardless of distance to the target
    exclude_hdir=False,
    dropout=False,  # this will iteratively dropout each cluster and recompute the LDA prediction accuracy to see how much that cluster matters
    PCA_process=[],  # number of PCs to use, if left empty it will run without PCA
    subsampling=False,  # whether to subsample to equalize data by angles and space
    exclude_stationary=True,  # mouse must be moving > 1cm/s, currently only works for experimental conditions (update filtering functions if you want to use with other settings)
    # ------------ Rayleigh model settings ----------------------
    rayleigh_significance="linshit",  # "linshit" or "bootstrap"
    single_cluster_plots=True,  # True: Plot every condition in one figure
    # False: Do not plot every condition in one figure for each cluster
    multi_cluster_plots=False,  # True: Plot every cluster in one figure for one condition
    # False: Do not plot every cluster in one figure
    # ------------ Escape Pattern settings ----------------------
    escape_tuning_bins=25,  # number of bins for escape pattern tuning
) 
