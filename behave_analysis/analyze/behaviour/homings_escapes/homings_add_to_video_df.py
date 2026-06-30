import numpy as np
import polars as pl

from settings.settings_overrides import settings_overrides
from behave_analysis.analyze.behaviour.homings_escapes.homings import get_Homings
from behave_analysis.analyze.behaviour.homings_escapes.homing_curation_syd_viewer import remove_manually_curated

# ------------------- SAVING BOOL TO DF -------------------------------
def add_homie_to_video_df(session, video_df, homing_type = "auto", tracking_data = [], savepath = []):
    
    from settings.settings_analyze_behave import settings_ab
    if homing_type == "manual":
        settings_ab = settings_overrides(settings_ab, {"redo_compute": False, "homings_use_boris": True})
        homing = get_Homings(settings_ab, session).get_homings(video_df, tracking_data)
    elif homing_type == "auto_curated":
        settings_ab = settings_overrides(settings_ab, {"redo_compute": False})
        homing = get_Homings({**settings_ab, "homings_curated": True}, session).get_homings(video_df, tracking_data)
        homing = remove_manually_curated(homing)
    elif homing_type == "auto":
        settings_ab = settings_overrides(settings_ab, {"redo_compute": False})
        homing = get_Homings(settings_ab, session).get_homings(video_df, tracking_data)

    # if homing data is present, create a boolean array to indicate when homing is occuring in the session
    number_of_frames = len(video_df)
    homing_bool = np.zeros(number_of_frames, dtype=bool)
    onset_frames = homing["onset_frames"]
    offset_frames = homing["offset_frames"]
    for onset, offset in zip(onset_frames, offset_frames):
        homing_bool[onset: offset + 1] = True

    if "homingPeriod" in video_df.columns:
        video_df = video_df.drop("homingPeriod")
    video_df = video_df.hstack([pl.Series("homingPeriod", homing_bool)])

    # save the video dataframe
    if len(savepath) > 0:
        video_df.write_csv(savepath)

    return video_df