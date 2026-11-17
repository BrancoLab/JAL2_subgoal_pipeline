# Custom libs
from loguru import logger

from behave_analysis.process.session import NEW_Session
from behave_analysis.utils.AI_dataClass_objects import Airpuff

# OS Libs
import os
import numpy as np
import json



def get_Airpuff(session: NEW_Session, dev4_file=None) -> Airpuff:
    """AI data is a 4 channel interleaved signal. The audio signal is the second channel.
    AI stands for analog input. The audio signal is an offshot of the signal sent to the speaker and
    equals a voltage recording.

    Args:
        session (Session): _description_

    Read dev4 ai bin file with the following channels:
    0. Nose-poke beam break (Dev4/ai4);
    1. lickport piezo (ai5);
    2. pump TTL copy (ai20);
    3. probe-sync copy (ai21);
    4. red-laser copy (ai19);
    5. blue-laser copy (ai18);
    6. side photodiode (ai0);
    7. airpuff TTL copy (ai22). (Dev4 channel setup)
    """
    if session.recording_script == "YT":
        assert dev4_file is not None, "AI file path must be provided to load camera trigger data."
        dev4_data = np.fromfile(dev4_file, dtype=np.float32)
        dev4_data_reshape = dev4_data.reshape(-1, 8)  # Reshape to 8 channels
        airpuff_trigger_ch = dev4_data_reshape[:, 7]  # Airpuff trigger is the 8th channel (index 7)

        frame_trigger_onsets = np.diff(airpuff_trigger_ch)
        frame_trigger_onsets_idx = np.where(frame_trigger_onsets > 1)[0] + 1
        frame_trigger_offset_idx = np.where(frame_trigger_onsets < -1)[0] + 1
        durations = frame_trigger_offset_idx - frame_trigger_onsets_idx

        airpuff = {"onsets": frame_trigger_onsets_idx.tolist(), 
                   "offsets": frame_trigger_offset_idx.tolist(), 
                   "duration": durations.tolist()}

        meta_file = os.path.join(session.base_path, session.processed_path, "airpuff.json")
        with open(meta_file, "w", encoding="utf-8") as json_file:
            json.dump(airpuff, json_file)

        airpuff = Airpuff(onset_frames=frame_trigger_onsets_idx, stimulus_durations=frame_trigger_offset_idx - frame_trigger_onsets_idx)
    elif session.recording_script == "JAL":
        airpuff = Airpuff(onset_frames=session.audio.onset_frames, stimulus_durations=np.ones_like(session.audio.onset_frames) * 0.5)  # Assuming a fixed duration of 0.1 seconds for each airpuff event
    logger.info(f"Airpuffs were delivered {len(airpuff.onset_frames)} times during this session.")

    return airpuff
