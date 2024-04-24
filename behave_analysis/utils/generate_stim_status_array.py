import numpy as np


def generate_stim_status_array(onset_frames: np.int64, stimulus_durations, seconds_before, seconds_after, fps) -> np.ndarray:
    """No idea what this function does, but it generates a numpy array of some sort. Adding an assert because
    it seems to fail for escapes but works for homings. Need to change the output of the onsets for escapes."""
    
    # assert isinstance(onset_frames, np.int64), "onset_frames must be an integer"
    if not isinstance(onset_frames, np.int64):
        onset_frames = int(onset_frames)

    stim_status = np.zeros((0) + int((seconds_before + stimulus_durations[-1] + seconds_after) * fps)) + 0.01  # 0.01 ~ in between stimuli
    stim_status[: seconds_before * fps] = np.arange(-seconds_before * fps - 1, -1) / fps  # pre-stimulus countdown in seconds
    stim_status[
        int(seconds_before * fps + onset_frames - int(onset_frames)) : int((seconds_before + stimulus_durations) * fps)
        + onset_frames
        - int(onset_frames)
    ] = 0  # 0 ~ stimulus is ON
    stim_status[-int(seconds_after * fps) :] = np.arange(1, seconds_after * fps + 1) / fps  # post-stimulus countup in seconds
    return stim_status
