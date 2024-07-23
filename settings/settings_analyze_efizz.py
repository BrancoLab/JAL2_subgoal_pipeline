"""
A settings class for the analyze_efizz.py script which is currently used to turn on or off the different models

Possible conditions to input as user defined conditions:

    'all_time' (don't filter based on shelter or barrier),
    'pre_shelter' (empty arena),
    'shelter_present',
    'barrier_present'
    'shelter_only',
    'barrier_pre_flip',
    'barrier_post_flip
"""

from behave_analysis.utils.settings_objects import Settings_analyze_efizz

Settings_ae = Settings_analyze_efizz(
    # ------------- General settings --------------------------
    stim_type="audio",  # 'audio', leave as 'None' if no stims were delivered
    linear_shift=False,  # whether to run linear shift!
    # This does not effect Tuned, this model needs linear shift to work
    redo_compute=False,  # if True it will force recompute any pre-saved analyses (e.g. Rayleigh and LDA)
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
    # conditions=["shelter_present",'pre_shelter'],
    conditions=["shelter_only", "barrier_pre_flip", "barrier_post_flip"],
    user_defined_conditions=True,  # False if you want automatically identified conditions
    condition_types="experimental_conditions",  # if 'experimental_conditions' it uses conditions listed above that start with user
    # if 'behavioral_conditions' it defines the conditions based on homing/escape behaviour of mousie - it will overrule other condition settings
    # if 'homing_number_2' it defines the conditions before and after a certain number of correct homings
    compartment_split=["all"],  # ['all','threat_zone','shelter_compartment','left_arena','right_arena']
    # If 'all' it will run the model on all data, if 'threat_zone' it will only run on the threat zone data e.g
    # If 'by_position', it will compute LDA decoding by arena position
    number_of_bins=9,  # number of bins for angles, e.g. 13 or 19 are good numbers
    classify_cells=False,
    # ------------------- Run single trial analysis -------------------
    run_single_trial=True,
    # ------------- PCA model settings --------------------------
    run_dim_reduction=False,
    run_pca=False,
    run_umap=False,
    #     run_pca_model=False,
    redo_pca_preprocessing=False,  # rerun if you have changed, angles, conditions, or underlying neural data
    # ------------- Tuned model settings -----------------------
    run_tunED=False,
    # ------------- Sklearn model settings -----------------------
    run_sklearn_decoders=False,
    # --------------LSTM model settings ------------------------
    run_LSTM=False,
    # ------------- LDA model settings --------------------------
    run_LDA=[],  # if [] it will not run LDA
    # if 'all_angles', 'all_distance','all_vectors' it will run it for all possible angles, distances, vectors
    # else:  list of angles ['hsa','hdir','h_bar_south_a','h_bar_north_a','h_bar_centre_a', 'randP']
    epoch_num=6,  # number of epochs for cross validation
    use_firing_rate=True,
    discriminant_type="linear",  # 'linear' or 'quadratic' or 'LSTM'
    exclude_proximal=10,  # this determines how far the mouse has to be from each point for head angle point decoding, if 0 LDA uses all head angles regardless of distance to the target
    exclude_hdir=False,
    dropout=False,  # this will iteratively dropout each cluster and recompute the LDA prediction accuracy to see how much that cluster matters
    PCA_process=[],  # numnber of PCs to use, if left empty it will run without PCA
    # ------------ Rayleigh model settings ----------------------
    run_rayleigh=False,
    rayleigh_significance="linshit",  # "linshit" or "bootstrap"
    single_cluster_plots=False,  # True: Plot every condition in one figure
    # False: Do not plot every condition in one figure for each cluster
    multi_cluster_plots=False,  # True: Plot every cluster in one figure for one condition
    # False: Do not plot every cluster in one figure
)
