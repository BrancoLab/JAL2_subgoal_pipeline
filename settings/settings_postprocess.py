# Custom lib
from behave_analysis.utils.settings_objects import Settings_postprocess

defined_settings_postprocess = Settings_postprocess(
    
    cluster_type = 'good', # cluster_type: "synthetic", "synthetichdir" (only hdir cells in synthetic dataset), "synthetichdirhsa", 
                                #"all" = mua + good, "mua" or "good" (or "noise" if you're feeling funky)
    efizz = True, # false for behavior only sessions

    # find escape onset
    response_thresh = 5 # in seconds, when did mouse escape after the stim?
     
) 