# Custom libs
from behave_analysis.process.session import NEW_Session
from behave_analysis.utils.get_onset_and_duration import get_onset_and_duration

# Os libs
import os
import numpy as np
from glob import glob
import dill as pickle
import pandas as pd
from loguru import logger

from dataclasses import dataclass


@dataclass(frozen=True)
class Dev_3_signal_nidaq:
    red_Laser_Signal: np.ndarray
    probe_Copy_TTL: np.ndarray
    laser_onsets: np.ndarray


def get_dev3_signals(session: NEW_Session):
    """Get all the things"""

    AI_file = list(session.file_path.glob("*analog_dev4.bin"))[0]  # need lst and idx as its a generator

    if ".bin" in str(AI_file):
        AI_data = np.fromfile(AI_file)
    else:
        with open(AI_file, "rb") as dill_file:
            AI_data = pickle.load(dill_file)

    red_Laser_Signal = AI_data[np.arange(0, len(AI_data), 2)]
    probe_Copy_TTL = AI_data[np.arange(1, len(AI_data), 2)]

    laserOn = red_Laser_Signal > 5
    min_frames_between_trials = session.daq_sampling_rate * 0.1
    data_on_idx = np.where(laserOn)[0]

    idx_since_data_on = np.append(np.inf, np.diff(data_on_idx))
    laser_onset_idx = data_on_idx[idx_since_data_on > min_frames_between_trials]

    idx_before_next_trial = np.append(-np.inf, np.diff(data_on_idx[::-1]))[::-1]
    data_offset_idx = data_on_idx[idx_before_next_trial < -min_frames_between_trials]

    laser_object = Dev_3_signal_nidaq(red_Laser_Signal, probe_Copy_TTL, laser_onset_idx)

    return laser_object

def plot_laser_sync_test_in_process(laser_signal, laser_onsets):
    # import polars as pl
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1)
    axs.plot(laser_signal)
    for onset in laser_onsets:
        axs.axvline(x=onset, color="r", linestyle="--")
    plt.show()
        
