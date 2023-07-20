"""A settings class for the analyze_efizz.py script which is currently used to turn on or off the different models"""

from behave_analysis.utils.settings_objects import Settings_analyze_efizz


Settings_analyze_efizz = Settings_analyze_efizz(
    
    # General settings
    cluster_type = 'synthetic', # Can choose all, good, mua, synthetic
    show_plots = False,
    analyze_only_the_period_before_shelter = False, # If True will only analyze the period before the shelter, if false, it will analyze after the whole session
    analyze_only_the_period_before_barrier = False, # If True will only analyze the period before the barrier, if false, it will analyze the whole session
    
    # Tuned model settings
    run_tunED = False,
    
    # LDA model settings
    # run_LDA = ['hdir', "head_shelter_angle"],
    object_present = True, # If True will analyse data only when object present, if false, it will analyze only when object not present
    
    # Consink model settings
    run_consink = False,

)

