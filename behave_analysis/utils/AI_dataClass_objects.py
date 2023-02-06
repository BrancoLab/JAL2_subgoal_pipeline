"""The following script contains several dataclasses that outline the fields of several AI data classes 
used within the pipeline. Each data class outlines the structure or blueprint of the settings object. 
As such each class below is just a shell.

It also contains a dataclass for the TTL sync object, which is used to align the big rig with the efizz machine"""

from dataclasses import dataclass

@dataclass(frozen=False)
class TTL_Sync:
    # Storing relevant data to align big rig with efizz machine using the onset of TTL pulses
    bonsai_TTL: float # voltage recordings of ttl signal from bonsai machine
    imec_TTL: float
    sampling_rate: int # Should be 30khz for neuropixels
    bonsai_sync_onsets: int # array of ints, onset/offsets PRE RESAMPLING 
    bonsai_sync_offsets: int # array of ints, onset/offsets PRE RESAMPLING 
    ephys_sync_onsets: int # array of ints, onset/offsets PRE RESAMPLING 
    ephys_sync_offset: int # array of ints, onset/offsets PRE RESAMPLING
    
@dataclass(frozen=True)
class Camera_trigger:
    num_samples: int
    num_frames: int
    frame_trigger_onsets_idx: object
    fps: int