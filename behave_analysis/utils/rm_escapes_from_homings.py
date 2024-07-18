"""Because we coupled homings into the escapes object which we shouldn't have done, we need some elegant way to remove the escapes from the homings object. This is what this script does."""

import numpy as np


# Function to determine if a homing frame is outside the escape intervals
def is_outside_escape(homing_onset, homing_offset, escape_onsets, escape_offsets):
    for escape_onset, escape_offset in zip(escape_onsets, escape_offsets):
        if homing_onset >= escape_onset and homing_offset <= escape_offset:
            return False
    return True


def remove_escapes_from_homings_object(homings_object: object, escape_object: object) -> object:
    """Remove escapes onsets and offsets from homings object"""
    check_attributes_of_homing_object(homings_object)
    assert hasattr(escape_object, "escape_onset_frames"), "The escape object does not have the attribute 'escape_onset_frames"
    assert hasattr(escape_object, "stimulus_durations"), "The escape object does not have the attribute 'stimulus_durations"
    escape_onsets = escape_object.escape_onset_frames.reshape(-1)
    escape_durations = escape_object.stimulus_durations.reshape(-1)
    escape_offsets = escape_onsets + (escape_durations * 40)  # 40 is the frame rate of the video

    # Create masks for each attribute based on the condition
    mask = np.array(
        [
            is_outside_escape(onset, offset, escape_onsets, escape_offsets)
            for onset, offset in zip(homings_object.onset_frames, homings_object.offset_frames)
        ]
    )

    # Apply the mask to each attribute
    homings_object.onset_frames = homings_object.onset_frames[mask]
    homings_object.offset_frames = homings_object.offset_frames[mask]
    homings_object.stimulus_durations = homings_object.stimulus_durations[mask]
    homings_object.avg_speed = homings_object.avg_speed[mask]
    homings_object.start_locs = homings_object.start_locs[mask]
    homings_object.end_locs = homings_object.end_locs[mask]
    homings_object.homing_angles_dic = {key: homings_object.homing_angles_dic[key][mask] for key in homings_object.homing_angles_dic.keys()}

    # Check that the lengths of the attributes are the same
    check_attributes_of_homing_object(homings_object)
    return homings_object


def check_attributes_of_homing_object(homings_object: object) -> None:
    """Check the attributes of the homings object"""
    assert hasattr(homings_object, "onset_frames"), "The homings object does not have the attribute 'onset_frames"
    assert hasattr(homings_object, "offset_frames"), "The homings object does not have the attribute 'offset_frames"
    assert hasattr(homings_object, "homing_angles_dic"), "The homings object does not have the attribute 'homing_angles_dic"
    assert hasattr(homings_object, "stimulus_durations"), "The homings object does not have the attribute 'stimulus_durations"
    assert hasattr(homings_object, "avg_speed"), "The homings object does not have the attribute 'avg_speed"
    assert hasattr(homings_object, "start_locs"), "The homings object does not have the attribute 'start_locs"
    assert hasattr(homings_object, "end_locs"), "The homings object does not have the attribute 'end_locs"

    attributes = ["onset_frames", "offset_frames", "stimulus_durations", "avg_speed", "start_locs", "end_locs"]
    # check the len of the attributes all match
    for attribute in attributes:
        assert len(getattr(homings_object, attribute)) == len(
            homings_object.onset_frames
        ), f"The attribute {attribute} does not have the same length as onset_frames"
    # check the values of the homing angles dictionary are the same length as the onset_frames
    for key in homings_object.homing_angles_dic.keys():
        assert len(homings_object.homing_angles_dic[key]) == len(
            homings_object.onset_frames
        ), f"The homing angles dictionary key {key} does not have the same length as onset_frames"
