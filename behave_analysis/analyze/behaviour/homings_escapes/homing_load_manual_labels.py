from loguru import logger
import pandas as pd
import os
import numpy as np

# ------------------- Use manual labels -------------------------------

def load_manual_labels(session) -> tuple:
    """Load manual labels from a csv file.
    NB: Assumes image frames are 1 indexed and converts to 0 based indexing here."""
    df = pd.read_csv(os.path.join(session.base_path, session.processed_path) + "\\" + "Borris" + "\\" + "scored_homings.csv")
    columns_to_keep = ["Time", "Image index", "Behavior type"]
    fdf = df[columns_to_keep]
    time = fdf["Time"].to_numpy()
    diff = np.diff(time)
    assert np.all(diff > 0), "Time is not increasing"
    start = len(fdf[fdf["Behavior type"] == "START"])
    end = len(fdf[fdf["Behavior type"] == "STOP"])
    assert start == end, "Start and end homings are not the same length"
    logger.info("Loaded manual labels")
    logger.info("Number of homings: {}".format(start))
    onsets = fdf[fdf["Behavior type"] == "START"]["Image index"].to_numpy() - 1  # convert to 0-based index
    offsets = fdf[fdf["Behavior type"] == "STOP"]["Image index"].to_numpy() - 1  # convert to 0-based index
    assert len(onsets) == len(offsets), "Onsets and offsets are not the same length"
    assert np.diff(onsets).all() > 0, "Onsets are not increasing"
    assert np.diff(offsets).all() > 0, "Offsets are not increasing"
    durations = offsets - onsets
    durations = np.array([[x] for x in (durations) / session.video.fps])  # match the format of the automatic labels
    return onsets, durations, offsets