"""A settings class for the analyze_efizz.py script which is currently used to turn on or off the different models"""

from behave_analysis.utils.settings_objects import Settings_analyze_efizz


Settings_analyze_efizz = Settings_analyze_efizz(
    cluster_type = 'synthetic',
    show_plots = False,
    run_tunED = False,
    run_LDA = ['head_shelter_angle','hdir'],
    object_present = True, # if running LDA on times when the object (shelter, barrier) is present
    run_consink = True,
)

