"""A settings class for the analyze_efizz.py script which is currently used to turn on or off the different models"""

from behave_analysis.utils.settings_objects import Settings_analyze_efizz


Settings_analyze_efizz = Settings_analyze_efizz(
    
    # General settings
    cluster_type = 'good',
    show_plots = False,
    object_present = False, # if running LDA on times when the object (shelter, barrier) is present
    
    # Tuned model settings
    run_tunED = True,
    
    # LDA model settings
    run_LDA = ['head_shelter_angle','hdir'],
    
    # Consink model settings
    run_consink = True,
)

