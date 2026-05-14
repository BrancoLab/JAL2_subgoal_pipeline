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
    homings=False,  # find homings, works best if you-ve manually labeled homings in boris
    regenerate_synthetic_data=True,  # If you have chosen synthetic data, do you want to regenerate the synthetic data?
    save_spike_video_parquet=False, # this will save a dataframe with the spike count for each cluster for each video frame, merged with the video data (like position and speed) large and redundant since there are also dataframes of spikes and behaviour
)
