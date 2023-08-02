"""A settings class for the analyze_efizz.py script which is currently used to turn on or off the different models"""

from behave_analysis.utils.settings_objects import Settings_analyze_efizz


Settings_analyze_efizz = Settings_analyze_efizz(
    
    # General settings
    cluster_type = ['synthetic','synthetichdir'],#,'all','good'], # Can choose all, good, mua
    show_plots = False,
    object_present = [True], # If True will analyse data only when object present, if false, it will analyze only when object not present 
    
    # Tuned model settings
    run_tunED = False,
    
    # LDA model settings
    run_LDA = ['hsa','hdir'], # 'head_shelter_angle','hdir','h_bar_south_a','h_bar_north_a','h_bar_centre_a', 'randP'
    # run_LDA = ['hsa','hdir','h_bar_south_a','h_bar_north_a','h_bar_centre_a', 'randP'], # 'hsa','hdir','h_bar_south_a','h_bar_north_a','h_bar_centre_a', 'randP'
    epoch_num = 6, # number of epochs for cross validation
    number_of_bins = 19, # number of bins for angles
    use_firing_rate = True,
    discriminant_type = 'linear', # 'linear' or 'quadratic'
    PCA_process = [], # numnber of PCs to use, if left empty it will run without PCA
    linear_shift = True, # whether to run linear shift!

    # Consink model settings
    run_consink = False,

)

