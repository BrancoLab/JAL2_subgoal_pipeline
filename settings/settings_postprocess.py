# Custom lib
from behave_analysis.utils.settings_objects import Settings_postprocess

defined_settings_postprocess = Settings_postprocess(
    
    cluster_type = 'all', # cluster_type: "synthetic", "all" = mua + good, "mua" or "good" (or "noise" if you're feeling funky)
     
) 