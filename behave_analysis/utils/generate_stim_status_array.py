import numpy as np

def generate_stim_status_array(onset_frames, stimulus_durations, seconds_before, seconds_after, fps) -> np.ndarray:
    stim_status = np.zeros((onset_frames[-1]-onset_frames[0])+int((seconds_before+stimulus_durations[-1]+seconds_after)*fps)) + 0.01 # 0.01 ~ in between stimuli
    stim_status[:seconds_before*fps] = np.arange(-seconds_before*fps-1, -1)/fps # pre-stimulus countdown in seconds
    for onset_frame, stimulus_duration in zip(onset_frames, stimulus_durations):
        stim_status[int(seconds_before*fps+onset_frame-onset_frames[0]):int((seconds_before+stimulus_duration)*fps)+onset_frame-onset_frames[0]]=0 # 0 ~ stimulus is ON
    stim_status[-int(seconds_after*fps):]=np.arange(1, seconds_after*fps+1)/fps  # post-stimulus countup in seconds
    return stim_status