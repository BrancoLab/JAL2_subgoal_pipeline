"""A settings class for the analyze_efizz.py script which is currently used to turn on or off the different models"""

from behave_analysis.utils.settings_objects import Settings_analyze_efizz


Settings_analyze_efizz = Settings_analyze_efizz(
    
    # General settings
    cluster_type = ['synthetichdir'],# ['synthetic','synthetichdir','all','good'], # Can choose all, good, mua
    show_plots = False,
    # possible condition inputs: 'all_time' (don't filter based on shelter or barrier),
    #                             'pre_shelter' (empty arena),
    #                             'shelter_present',
    #                             'barrier_present'
    #                             'shelter_only',
    #                             'barrier_pre_flip',
    #                             'barrier_post_flip',
    # if condition is empty all possible conditions will be analyzed
    condition = ['shelter_present'], 
    analyze_only_the_period_before_shelter = False, # If True will only analyze the period before the shelter, if false, it will analyze after the whole session
    analyze_only_the_period_before_barrier = False, # If True will only analyze the period before the barrier, if false, it will analyze the whole session
    
    # Tuned model settings
    run_tunED = False,
    
    # LDA model settings
    run_LDA = 'all', 
    # if [] it will not run LDA
    # if 'all' it will run it for all possible angles - else provide list of angles
    # 'hsa','hdir','h_bar_south_a','h_bar_north_a','h_bar_centre_a', 'randP'
    epoch_num = 6, # number of epochs for cross validation
    number_of_bins = 19, # number of bins for angles
    use_firing_rate = True,
    discriminant_type = 'linear', # 'linear' or 'quadratic'
    PCA_process = [], # numnber of PCs to use, if left empty it will run without PCA
    linear_shift = False, # whether to run linear shift!

    # Rayleigh model settings
    run_rayleigh = False,
    rayleigh_bootstrap = False, # TODO: rewrite this with linear shift stats
    single_cluster_plots = True,

)

