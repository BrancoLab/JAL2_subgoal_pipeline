"""_summary_

A script to return the onset of TTL pulses to align the behavioural data 
collected on the big rig with the efizz data collected on the efizz machine.

Returns an object class containing:
- ttl pulse onsets
- the raw ttl signal
"""

#Custom libaries
from behave_analysis.process.session import Session

#OS libaries
import os
import numpy as np
from dataclasses import dataclass
from glob import glob
import dill as pickle
import pandas as pd

@dataclass(frozen=True)
class TTL_Sync:
    # Storing relevant data to align big rig with efizz machine using the onset of TTL pulses
    raw_signal: float

def get_ttl_pulse_trigger(session: Session) -> TTL_Sync:
    """_summary_
    Returns the TTL_sync class containing the onset of TTL pulses.

    Args:
        session (Session): custom object containing experimental path file

    Returns:
        TTL_Sync: TTL_Sync.pulse_onset can be used to sync with another machine
    """
    AI_file = glob(os.path.join(session.file_path, "analog*"))[-1] # take the last file if there are multiple
    if '.bin' in AI_file: 
        AI_data = np.fromfile(AI_file)
    else:
        with open(AI_file, "rb") as dill_file: AI_data = pickle.load(dill_file)
    ttl_signal = AI_data[3:-1:4] #From the 3 index until the end select every 4th sample
    ttl_object = TTL_Sync(ttl_signal)
    return (ttl_object)