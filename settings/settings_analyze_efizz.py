"""A settings class for the analyze_efizz.py script which is currently used to turn on or off the different models"""

from behave_analysis.utils.settings_objects import Settings_analyze_efizz


Settings_analyze_efizz = Settings_analyze_efizz(
    
    # General settings
    cluster_type = 'good', # Can choose all, good, mua
    show_plots = False,
    object_present = False, # If True will analyse data only when object present, if false, it will analyze only when object not present 
    
    # Tuned model settings
    run_tunED = True,
    
    # LDA model settings
    # run_LDA = ['head_shelter_angle','hdir'], # 'head_shelter_angle','hdir','h_bar_south_a','h_bar_north_a','h_bar_centre_a', 'randP'
    run_LDA = ['hsa','hdir','h_bar_south_a','h_bar_north_a','h_bar_centre_a','randP'], # 'head_shelter_angle','hdir','h_bar_south_a','h_bar_north_a','h_bar_centre_a', 'randP'
    
    # Consink model settings
    run_consink = False,

)

