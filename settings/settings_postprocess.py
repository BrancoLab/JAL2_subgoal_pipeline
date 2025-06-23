"""Post process settings for the pipeline. 

The following are options for the cluster type:
- "synthetic" (all cells) - Synthetic data will always include head direction cells, you must hard code this out in the synthic data post processor
- "all" (all cells in the dataset) real data
- "mua" (only mua cells in the dataset) real data
- "good" (only good cells in the dataset) real data
- "noise" (only noise cells in the dataset) real data

"""

from behave_analysis.utils.settings_objects import Settings_postprocess

defined_settings_postprocess = Settings_postprocess(
    cluster_type="good",
    efizz=True,  # false for behavior only sessions
    homings=True,  # find escape onset
    response_thresh=5,  # in seconds, when did mouse escape after the stim? Cut off time
    regenerate_synthetic_data=True,  # If you have chosen synthetic data, do you want to regenerate the synthetic data?
)
