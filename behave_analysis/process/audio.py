# Custom libs
from behave_analysis.process.session import NEW_Session
from behave_analysis.utils.get_onset_and_duration import get_onset_and_duration
from behave_analysis.utils.AI_dataClass_objects import Audio

# OS Libs
import os
import numpy as np
from glob import glob
import dill as pickle
from pathlib import Path
from loguru import logger


def get_Audio(session: NEW_Session, AI_data=None) -> Audio:
    """AI data is a 4 channel interleaved signal. The audio signal is the second channel.
    AI stands for analog input. The audio signal is an offshot of the signal sent to the speaker and
    equals a voltage recording.

    Args:
        session (Session): _description_
        indexs_to_remove (_type_, optional): _description_. Defaults to None.
        down_sample (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    assert AI_data is not None, "AI file path must be provided to load camera trigger data."

    if session.recording_script == "JAL":
        audio_data = AI_data[:,1] # microphone
        audio_on = abs(audio_data) > 3
    elif session.recording_script == "YT":
        audio_data = AI_data[:,4] # audio generation copy
        audio_on = abs(audio_data) > .025
    
    audio_num_samples = len(audio_data)
    audio_onset_frames, stimulus_durations, _ = get_onset_and_duration(
        audio_on, session, stim_type="audio", min_frames_between_trials=session.daq_sampling_rate * 5, data_type="samples"
    )

    audio = Audio(audio_num_samples, audio_onset_frames, stimulus_durations)
    logger.info(f"Auditory threats were presented {len(audio_onset_frames)} times during this session.")
    return audio
