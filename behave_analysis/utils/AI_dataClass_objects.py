"""The following script contains several dataclasses that outline the fields of several AI data classes 
used within the pipeline. Each data class outlines the structure or blueprint of the settings object. 
As such each class below is just a shell.It also contains a dataclass for the TTL sync object, which 
is used to align the big rig with the efizz machine"""

from dataclasses import dataclass

@dataclass(frozen=False)
class TTL_Sync:
    bonsai_TTL: float 
    imec_TTL: float
    sampling_rate: int 
    bonsai_sync_onsets: int 
    bonsai_sync_offsets: int 
    ephys_sync_onsets: int 
    ephys_sync_offset: int

@dataclass(frozen=True)
class Elecetrophysiology:
    spike_times: object
    spike_clusters: object
    cluster_group: object
    TTL_bin_path: str
    
@dataclass(frozen=True)
class Camera_trigger:
    num_samples: int
    num_frames: int
    frame_trigger_onsets_idx: object
    fps: int

@dataclass(frozen=True)
class photoresistor_trigger:
    num_samples: int
    onset_frames: object
    stimulus_durations: object
    
@dataclass(frozen=True)
class Audio:
    num_samples: int
    onset_frames: object
    stimulus_durations: object