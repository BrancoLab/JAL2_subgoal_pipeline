"""A settings class for the analyze_efizz.py script which is currently used to turn on or off the different models"""

from behave_analysis.utils.settings_objects import Settings_analyze_efizz


Settings_analyze_efizz = Settings_analyze_efizz(
    cluster_type = 'all',
    show_plots = False,
    run_tunED = False,
    # run_LDA = ['head_shelter_angle','hdir'], # 'head_shelter_angle','hdir','h_bar_south_a','h_bar_north_a','h_bar_centre_a', 'randP'
    run_LDA = ['hsa','hdir','h_bar_south_a','h_bar_north_a','h_bar_centre_a','randP'], # 'head_shelter_angle','hdir','h_bar_south_a','h_bar_north_a','h_bar_centre_a', 'randP'
    object_present = False, # if running LDA on times when the object (shelter, barrier) is present
    run_consink = False,
)

