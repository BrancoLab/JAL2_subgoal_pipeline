import numpy as np

def build_shelter_condition_bool(session, frame_idx: np.array, n_frames: int) -> None:
    """INPUTS: 
        frame_idx (np.array): array of frame indices, len(frame_idx) = n_frames but it's 1-indexed, so frame_idx[0] = 1
        n_frames (int)"""
    
    if len(session.shelter_time) > 0:
        start = session.shelter_time[0] * session.video.fps * 60
        if session.shelter_time[1] == -1:  # shelter until the end of the session
            end = n_frames + 1
        else:
            end = session.shelter_time[1] * session.video.fps * 60
        
        shelter = np.logical_and(frame_idx > start, frame_idx < end)
    else:
        # there was never a shelter in the session, so shelter is always false
        shelter = np.full(n_frames, False)
    return shelter

def build_barrier_condition_bool(session, frame_idx: np.array, n_frames: int) -> None:
    # what period in the recording was there a barrier?
    if len(session.barrier_time) > 0:
        start = session.barrier_time[0] * session.video.fps * 60
        if session.barrier_time[1] == -1:  # barrier present until the end of the session
            end = n_frames + 1
        else:
            end = session.barrier_time[1] * session.video.fps * 60
        barrier_present = np.logical_and(frame_idx > start, frame_idx < end)
    else:
        barrier_present = np.full(n_frames, False)
        print("no barrier in this session")
    return barrier_present

def build_flippedbarrier_condition_bool(session, frame_idx: np.array, n_frames: int) -> None:
    # when was the barrier flipped?
    if session.barrier_flip_time:
        barrier_flipped = frame_idx > (session.barrier_flip_time * session.video.fps * 60)
    else:
        barrier_flipped = np.full(n_frames, False)
        print("barrier was not flipped in this session")
    return barrier_flipped

def identify_condition_of_trial(video_df, session) -> str:
    """
    Determine the experimental condition of a trial based on video and session data.

    This function analyzes the given DataFrame and session object to ascertain the specific condition
    under which a trial occurred. It categorizes the trial into one of several predefined conditions
    based on the status of the shelter and barrier at the time of the trial.

    Parameters:
    - video_df (DataFrame): A DataFrame containing columns 'shelter', 'barrier_present', and 'barrier_flipped'.
                             Each column is expected to have boolean values indicating the status of each element.
    - session (Object): An object representing the session, which should include the 'barrier_flip_time' attribute.

    Returns:
    - str: The identified condition of the trial, which can be 'shelter_only', 'barrier_pre_flip', 'barrier_post_flip', or 'barrier_present'.

    Raises:
    - AssertionError: If the input DataFrame does not contain the expected columns or if the session object is not in the expected format.
    """

    # Assertions to validate inputs
    assert all(
        col in video_df.columns for col in ["shelter", "barrier_present", "barrier_flipped"]
    ), "video_df must contain 'shelter', 'barrier_present', and 'barrier_flipped' columns."
    # assert hasattr(session, "barrier_flip_time"), "session must have 'barrier_flip_time' attribute."

    condition = ""

    if video_df["shelter"].to_numpy() == False:
        condition = 'pre_shelter'

    # Check if mouse is in the shelter condition
    if np.logical_and(video_df["shelter"].to_numpy() == True, video_df["barrier_present"].to_numpy() == False):
        condition = "shelter_only"
        if video_df["barrier_flipped"].to_numpy() == True:
            condition = "barrier_removed" # ATTENTION: this only works if barrier is removed after flip!!!

    # Check which barrier condition the mouse is in
    elif np.logical_and(video_df["shelter"].to_numpy() == True, video_df["barrier_present"].to_numpy() == True):
        if session.barrier_flip_time:
            # Check if the barrier has been flipped
            if video_df["barrier_flipped"].to_numpy() == False:
                condition = "barrier_pre_flip"

            # Check if the barrier has been flipped
            if np.logical_and(video_df["barrier_flipped"].to_numpy() == True, video_df["barrier_present"].to_numpy() == True):
                condition = "barrier_post_flip"

        else:
            condition = "barrier_present"
    
    else:
        condition = "pre_shelter"

    return condition