
from syd import make_viewer
import numpy as np
import polars as pl
from pathlib import Path
import os
from loguru import logger
import matplotlib.pyplot as plt

from behave_analysis.analyze.behaviour.homings_escapes.homing_load_manual_labels import load_manual_labels
from behave_analysis.analyze.results_database_utils import check_database_for_same_run, settings_to_check
from behave_analysis.utils.identify_condition import identify_condition_of_trial
from behave_analysis.visualize.visualize_utils import open_tracking_data
from behave_analysis.utils.arena_plotting import Arena

def homing_curation_syd_viewer(
    homing_dict: dict,
    settings: object,
    session: object,
    video_df: pl.DataFrame = [],
    tracking_data: dict = [],
    include_manual_events: bool = True,
    manual_curation = True,
):
    """
    Create interactive Syd viewer for exploring extracted runs and manual labels.
    Optional: save removed runs to CSV.
    
    Args:
        candidates: List of (onset, offset) tuples if only plotting one session, or a dict of {session_name: [(onset, offset), ...]} if plotting multiple sessions
        session_name: Filter to single session (if None, includes all), or if pasing a list of tuples give name as str
    """
    
    viewer = make_viewer()
    viewer.add_integer('speed_thresh', min=0, max=20, value=settings.homings_speed_threshold)
    
    # load data
    if len(tracking_data) == 0:
        tracking_data = open_tracking_data(session)
    if len(video_df) == 0:
        video_df_path = Path(session.base_path) / session.processed_path / 'full_video_dataframe.csv'
        video_df = pl.read_csv(video_df_path)
    fps = session.video.fps
    n_frames = len(video_df["speed"])

    # candidates
    cand_on = homing_dict['onset_frames']
    cand_off = homing_dict['offset_frames']
    cand_idx = np.arange(len(cand_on))
    candidate_mask = np.zeros(n_frames, dtype=bool)
    for on_i, off_i in zip(cand_on, cand_off):
        candidate_mask[on_i:off_i + 1] = True
    
    # load the manual
    if os.path.isfile(session.base_path + '/' + session.processed_path + '/Borris/scored_homings.csv'):
        man_on, _, man_off = load_manual_labels(session)
        # remove manual labels that coincide with candidates
        manual_mask = np.zeros(len(video_df), dtype=bool)
        overlapping_masks = np.zeros(len(man_on), dtype=bool)
        for i, (on, off) in enumerate(zip(man_on, man_off)):
            if np.sum(candidate_mask[on:off + 1]) > ((off-on + 1)*.8):
                overlapping_masks[i] = True
                continue
            manual_mask[on:off + 1] = True
        man_on = man_on[~overlapping_masks]
        man_off = man_off[~overlapping_masks]
    else:
        logger.warning("No manual events found, iterating over candidates only")
        manual_mask = np.zeros(len(video_df), dtype=bool)
        man_on = np.array([], dtype=int)
        man_off = np.array([], dtype=int)
    man_idx = np.full(len(man_on), np.nan)

    # are we showing only candidates, only manual, or both?
    if include_manual_events:
        # merge all onsets
        all_onsets = np.concatenate([cand_on, man_on]) if len(man_on) > 0 else cand_on.copy()
        all_offsets = np.concatenate([cand_off, man_off]) if len(man_off) > 0 else cand_off.copy()
        all_source = np.concatenate([
            np.zeros(len(cand_on), dtype=int),
            np.ones(len(man_on), dtype=int)
        ]) if len(man_on) > 0 else np.zeros(len(cand_on), dtype=int)
        all_indices = np.concatenate([cand_idx, man_idx]) if len(man_idx) > 0 else cand_idx.copy()
    else:
        if len(cand_on) == 0:
            logger.warning("No candidate events found...")
            all_onsets, all_offsets, all_indices, all_source = [], [], [], []
        else:
            all_onsets, all_offsets, all_indices = cand_on.copy(), cand_off.copy(), cand_idx.copy()
            all_source = np.zeros(len(cand_on), dtype=int)

    # sort the onsets
    viewer.add_integer('trial', min=0, max=len(all_onsets)-1, value=0)

    def see_next(state):
        viewer.update_integer("trial", value=state["trial"] + 1)
    viewer.add_button("see_next", label="See next", callback=see_next)

    order = np.argsort(all_onsets)
    all_onsets = all_onsets[order]
    all_offsets = all_offsets[order]
    all_source = all_source[order]
    all_indices = all_indices[order]

    if manual_curation:
        removed_runs = []  # Event IDs of removed runs
        def remove_run(state):
            idx = int(state['trial'])
            removed_runs.append(all_indices[idx])
        viewer.add_button('remove_event', label='Remove Event', callback = remove_run, replot = False)

    # figure set up
    # blank figure if no events
    if len(all_onsets) == 0:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        ax.axis('off')
        mode_txt = 'candidates + manual' if include_manual_events else 'candidates only'
        ax.text(0.02, 0.8, f"No events in ({mode_txt})", fontsize=11)

    def plot(state):
        idx = int(state['trial'])

        # check if idx is out of bounds
        if idx > len(all_onsets) - 1:
            fig, ax = plt.subplots(1, 1, figsize=(8, 3))
            ax.axis('off')
            ax.text(0.02, 0.8, f"Trial index {idx} exceeds number of events ({len(all_onsets)})")
            return fig
        
        # get the index run!
        onset = int(all_onsets[idx])
        offset = int(all_offsets[idx])

        pad = int(round(fps))
        start = max(0, onset - pad)
        stop = min(n_frames, offset + pad + 1)

        # segment the relevant behavioral data
        t = (np.arange(start, stop) - start) / fps
        x = video_df["mouse_x_position"].to_numpy()[start:stop]
        y = video_df["mouse_y_position"].to_numpy()[start:stop]
        speed = video_df["speed"].to_numpy()[start:stop]
        hdir = video_df["hdir"].to_numpy()[start:stop]
        condition = identify_condition_of_trial(video_df.filter(video_df["frames"] == int(onset)), session)

        cand_win = candidate_mask[start:stop]
        man_win = manual_mask[start:stop]

        # plot it!
        fig, axs = plt.subplots(3, 2, figsize=(10, 11), gridspec_kw={'height_ratios': [2, 1, 1]})

        axs[0, 0].plot(x, y, color='0.75', lw=1)
        axs[0, 0].scatter(x[cand_win], y[cand_win], s=8, color='tab:blue', alpha=0.8, label='candidate')
        axs[0, 0].scatter(x[0], y[0], s=35, color='green', label='window start')
        axs[0, 0].scatter(x[-1], y[-1], s=35, color='red', label='window end')
        Arena(ax = axs[0, 0], condition = condition,
            barrier_coordinates = tracking_data["barrier_loc"][:-1], 
            shelter_coordinates=tracking_data["shelter_loc"], full_image = False)

        axs[0, 1].scatter(x[man_win], y[man_win], s=8, color='tab:orange', alpha=0.8, label='manual')
        axs[0, 1].plot(x, y, color='0.85', lw=1)
        Arena(ax = axs[0, 1], condition = condition,
            barrier_coordinates = tracking_data["barrier_loc"][:-1], 
            shelter_coordinates=tracking_data["shelter_loc"], full_image = False)

        src = 'candidate' if all_source[idx] == 0 else 'manual'
        axs[0, 0].set_title(f'trial {idx} ({src})')
        axs[0, 0].set_xlim(0, 1024)
        axs[0, 0].set_ylim(0, 1024)
        axs[0, 0].set_aspect('equal')
        axs[0, 0].invert_yaxis()
        axs[0, 0].legend(loc='upper right', fontsize=8)

        axs[0, 1].set_xlim(0, 1024)
        axs[0, 1].set_ylim(0, 1024)
        axs[0, 1].set_aspect('equal')
        axs[0, 1].invert_yaxis()

        axs[1, 0].plot(t, speed, color='black', lw=1)
        axs[1, 0].fill_between(t, 0, speed, where=cand_win, color='tab:blue', alpha=0.25)
        ax_r0 = axs[1, 0].twinx()
        ax_r0.plot(t, speed > state['speed_thresh'], color='tab:red', lw=1)

        axs[1, 1].plot(t, speed, color='black', lw=1)
        axs[1, 1].fill_between(t, 0, speed, where=man_win, color='tab:orange', alpha=0.18)
        ax_r1 = axs[1, 1].twinx()
        ax_r1.plot(t, speed > state['speed_thresh'], color='tab:red', lw=1)

        axs[1, 0].axvline((onset - start) / fps, color='tab:green', ls='--', lw=1)
        axs[1, 0].axvline((offset - start) / fps, color='tab:red', ls='--', lw=1)
        axs[1, 0].set_ylabel('speed')

        axs[2, 0].plot(t, hdir, color='slategray', lw=1)
        axs[2, 0].fill_between(t, np.nanmin(hdir), np.nanmax(hdir), where=cand_win, color='tab:blue', alpha=0.12)
        axs[2, 1].plot(t, hdir, color='slategray', lw=1)
        axs[2, 1].fill_between(t, np.nanmin(hdir), np.nanmax(hdir), where=man_win, color='tab:orange', alpha=0.10)
        axs[2, 0].axvline((onset - start) / fps, color='tab:green', ls='--', lw=1)
        axs[2, 0].axvline((offset - start) / fps, color='tab:red', ls='--', lw=1)
        axs[2, 0].set_ylabel('hdir')
        axs[2, 0].set_xlabel('time (s)')

        overlap = int(np.sum(candidate_mask[onset:offset + 1] & manual_mask[onset:offset + 1]))
        manual_count = int(np.sum(manual_mask[onset:offset + 1]))
        cand_count = int(np.sum(candidate_mask[onset:offset + 1]))
        axs[0, 1].set_title(f'candidate frames={cand_count} | manual frames={manual_count} | overlap={overlap}')

        plt.tight_layout()
        return fig

    viewer.set_plot(plot)

    return viewer, removed_runs


def save_removed_runs(homings_dict: dict, removed_event_ids: list, settings: object, session) -> None:
    """Save removed runs to dict and to databse"""
    removed_runs = np.zeros(homings_dict['onset_frames'].shape, dtype=bool)
    if len(removed_event_ids) > 0:
        removed_runs[removed_event_ids] = True
    homings_dict['removed_runs'] = removed_runs

    # find the hexname and load the database
    savepath = os.path.join(session.base_path, session.processed_path, "homings")
    database, _, hexaname = check_database_for_same_run(
                db_settings={**settings_to_check(settings, ["homing"])},
                results_csv_name=savepath + os.sep + "Homing_database.csv",
                settings=settings,
            )
    # in the row of the database with "name" == hexaname, set the "curated" column to True
    rowidx = database[database['name'] == hexaname].index[0]
    database.loc[rowidx, ["homings_curated"]] = True
    database.to_csv(savepath+ os.sep + "Homing_database.csv", index=False)

    # save the removed runs to a CSV file
    filename = os.path.join(savepath, "homings_" + hexaname)
    print(f"Saving curated homings to {filename}_results.npy and database entry to {savepath + os.sep + 'Homing_database.csv'}")
    np.save(os.path.join(filename + "_results.npy"), homings_dict, allow_pickle=True)

def remove_manually_curated(homings_dict: dict) -> dict:
    """Remove manually curated runs from the homings dict"""
    if 'removed_runs' in homings_dict.keys():
        removed_runs = homings_dict['removed_runs']
        for key in homings_dict.keys():
            if isinstance(homings_dict[key], list):
                homings_dict[key] = [v for i, v in enumerate(homings_dict[key]) if not removed_runs[i]]
            elif isinstance(homings_dict[key], np.ndarray) and len(homings_dict[key]) == len(removed_runs):
                homings_dict[key] = homings_dict[key][~removed_runs]
            elif isinstance(homings_dict[key], dict):
                for subkey in homings_dict[key].keys():
                    if isinstance(homings_dict[key][subkey], np.ndarray) and len(homings_dict[key][subkey]) == len(removed_runs):
                        homings_dict[key][subkey] = homings_dict[key][subkey][~removed_runs]
    return homings_dict