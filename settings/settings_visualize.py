"""
In this script you can set things such as:
+ Whether to show the trail behind the mouse
+ How fast the videos should be
+ What time before and after would you like to view the trials
+ What even is a trial (laser, escape, homing)
"""
# Standard import
from pathlib import Path

# Custom lib
from behave_analysis.utils.settings_objects import Settings_visualize

defined_settings_visualize = Settings_visualize(
    stim_type="audio",  # leave as 'None' if no stims were delivered
    display_trail=True,
    display_tracking=True,
    display_stimulus=True,
    rapid=True,  # Make the videos faster or slower for debugging
    seconds_before_audio=3,
    seconds_before_laser=3,
    seconds_before_homing=3,
    seconds_before_threshold_crossing=3,
    seconds_after_audio=2,
    seconds_after_laser=6,
    seconds_after_homing=3,
    seconds_after_threshold_crossing=3,
    save_folder=Path.cwd(),
    by_experiment=False,
    by_session=True,
    sessions=[0],  # This session must be the one indexed in the databank
    all_sessions=False,
    efizz=True,  # if you want to visualize efizz also
    show_plots=False,  # if this is false, it will make and save the plots without showing them
    cluster_type="synthetic",
    # cluster_type: "synthetic", "synthetichdir" (only hdir cells in synthetic dataset),
    # "synthetichdirhsa", "all" = mua + good, "mua" or "good" (or "noise" if you're feeling funky)
    conditions = ["shelter_only", "barrier_pre_flip", "barrier_post_flip"],  # JAL3-5
    # conditions=["all_time", "shelter_only", "barrier_present"], # JAL1-2
    user_defined_conditions=False, # False if you want automatically identified conditions 
    
)
