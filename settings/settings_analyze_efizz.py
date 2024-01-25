"""A settings class for the analyze_efizz.py script which is currently used to turn on or off the different models"""

from behave_analysis.utils.settings_objects import Settings_analyze_efizz

Settings_ae = Settings_analyze_efizz(
    # General settings
    stim_type="None",  # 'audio', leave as 'None' if no stims were delivered
    linear_shift=False,  # whether to run linear shift!
    # This does not effect Tuned, this model needs linear shift to work
    redo_compute=True,  # if True it will force recompute any pre-saved analyses (e.g. Rayleigh and LDA)
    cluster_type=["good"],  # ['synthetic','synthetichdir','all','good'], # Can choose all, good, mua
    show_plots=False,
    # possible condition inputs: 'all_time' (don't filter based on shelter or barrier),
    #                             'pre_shelter' (empty arena),
    #                             'shelter_present',
    #                             'barrier_present'
    #                             'shelter_only',
    #                             'barrier_pre_flip',
    #                             'barrier_post_flip',
    conditions=["shelter_only", "barrier_pre_flip", "barrier_post_flip"],
    user_defined_conditions=False,  # False if you want automatically identified conditions
    learned_conditions=True,  # if True it defines the conditions based on homing/escape behaviour of mousie - it will overrule other condition settings
    number_of_bins=19,  # number of bins for angles
    # ------------- PCA model settings --------------------------
    run_pca_model=False,
    redo_pca_preprocessing=False,  # rerun if you have changed, angles, conditions, or underlying neural data
    # ------------- Tuned model settings -----------------------
    run_tunED=False,
    # ------------- LDA model settings --------------------------
    run_LDA='all',  # if [] it will not run LDA
    # if 'all' it will run it for all possible angles - else provide list of angles
    # 'hsa','hdir','h_bar_south_a','h_bar_north_a','h_bar_centre_a', 'randP'
    epoch_num=6,  # number of epochs for cross validation
    use_firing_rate=True,
    discriminant_type="linear",  # 'linear' or 'quadratic'
    PCA_process=[],  # numnber of PCs to use, if left empty it will run without PCA
    # ------------ Rayleigh model settings ----------------------
    run_rayleigh=True,
    rayleigh_significance="linshit",  # can be either linear shift or bootstrap
    single_cluster_plots=True,  # True: Plot every condition in one figure
    # False: Do not plot every condition in one figure for each cluster
    multi_cluster_plots=False,  # True: Plot every cluster in one figure for one condition
    # False: Do not plot every cluster in one figure
)
