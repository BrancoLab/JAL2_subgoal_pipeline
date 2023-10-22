# Custom lib
from behave_analysis.utils.settings_objects import Settings_postprocess

defined_settings_postprocess = Settings_postprocess(
    
    cluster_type = 'synthetic', # cluster_type: "synthetic", "synthetichdir" (only hdir cells in synthetic dataset), "synthetichdirhsa", 
                                #"all" = mua + good, "mua" or "good" (or "noise" if you're feeling funky)
    no_efizz = True, # true for behavior only sessions
     
) 