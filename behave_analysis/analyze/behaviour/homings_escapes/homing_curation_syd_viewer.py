
from syd import make_viewer
import numpy as np
import polars as pl
from pathlib import Path
import os
from loguru import logger
import matplotlib.pyplot as plt

from behave_analysis.analyze.behaviour.homings_escapes.homings import load_manual_labels
from behave_analysis.analyze.results_database_utils import check_database_for_same_run, settings_to_check
from behave_analysis.visualize.visualize_utils import open_tracking_data
from behave_analysis.utils.arena_plotting import Arena

def homing_curation_syd_viewer(
    homing_dict: dict,
    settings: object,
    session: object,
    include_manual_events_in_iteration: bool = True,
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
    viewer.add_integer('trial', min=0, max=500, value=0)
    viewer.add_integer('speed_thresh', min=0, max=20, value=settings.homings_speed_threshold)
    viewer.add_integer('include_manual_events', min=0, max=1, value=1 if include_manual_events_in_iteration else 0)
    
    if manual_curation:
        removed_runs = []  # Event IDs of removed runs

        def remove_run(state):
            idx = int(state['trial'])
            removed_runs.append(idx)
        viewer.add_button('remove_event', label='Remove Event', callback = remove_run, replot = False)

    # load data
    tracking_data = open_tracking_data(session)
    video_df_path = Path(session.base_path) / session.processed_path / 'full_video_dataframe.csv'
    video_df = pl.read_csv(video_df_path)
    if os.path.isfile(session.base_path + '/' + session.processed_path + '/Borris/scored_homings.csv'):
        man_on, _, man_off = load_manual_labels(session)
        manual_mask = np.zeros(len(video_df), dtype=bool)
        for on, off in zip(man_on, man_off):
            manual_mask[on:off + 1] = True
    else:
        manual_mask = np.zeros(len(video_df), dtype=bool)
        man_on = np.array([], dtype=int)
        man_off = np.array([], dtype=int)

    def plot(state):
        idx = int(state['trial'])
        fps = session.video.fps
        n_frames = len(video_df["speed"])
        cand_on = homing_dict['onset_frames']
        cand_off = homing_dict['offset_frames']

        candidate_mask = np.zeros(n_frames, dtype=bool)
        for on_i, off_i in zip(cand_on, cand_off):
            candidate_mask[on_i:off_i + 1] = True

        iterate_manual = int(state.get('include_manual_events', 1)) == 1
        if len(man_on) == 0:
            logger.warning("No manual events found, iterating over candidates only")
            iterate_manual = False
        if iterate_manual:
            all_onsets = np.concatenate([cand_on, man_on]) if len(man_on) > 0 else cand_on.copy()
            all_offsets = np.concatenate([cand_off, man_off]) if len(man_off) > 0 else cand_off.copy()
            all_source = np.concatenate([
                np.zeros(len(cand_on), dtype=int),
                np.ones(len(man_on), dtype=int)
            ]) if len(man_on) > 0 else np.zeros(len(cand_on), dtype=int)
        else:
            all_onsets = cand_on.copy()
            all_offsets = cand_off.copy()
            all_source = np.zeros(len(cand_on), dtype=int)

        # blank figure if no events
        if len(all_onsets) == 0:
            fig, ax = plt.subplots(1, 1, figsize=(8, 3))
            ax.axis('off')
            mode_txt = 'candidates + manual' if iterate_manual else 'candidates only'
            ax.text(0.02, 0.8, f"No events in ({mode_txt})", fontsize=11)
            return fig

        order = np.argsort(all_onsets)
        all_onsets = all_onsets[order]
        all_offsets = all_offsets[order]
        all_source = all_source[order]

        if idx > len(all_onsets) - 1:
            fig, ax = plt.subplots(1, 1, figsize=(8, 3))
            ax.axis('off')
            ax.text(0.02, 0.8, f"Trial index {idx} exceeds number of events ({len(all_onsets)})")
            return fig

        onset = int(all_onsets[idx])
        offset = int(all_offsets[idx])

        pad = int(round(fps))
        start = max(0, onset - pad)
        stop = min(n_frames, offset + pad + 1)

        t = (np.arange(start, stop) - start) / fps
        x = video_df["mouse_x_position"].to_numpy()[start:stop, 0]
        y = video_df["mouse_y_position"].to_numpy()[start:stop, 0]
        speed = video_df["speed"].to_numpy()[start:stop]
        hdir = video_df["hdir"].to_numpy()[start:stop]
        if video_df["barrier_present"].to_numpy()[start] == False:
                condition = 'shelter_only'
        elif video_df["barrier_present"].to_numpy()[start] == True:
            condition = 'barrier_pre_flip'
            if video_df["barrier_flipped"].to_numpy()[start] == True:
                condition = 'barrier_post_flip'

        cand_win = candidate_mask[start:stop]
        man_win = manual_mask[start:stop]

        fig, axs = plt.subplots(3, 2, figsize=(10, 11), gridspec_kw={'height_ratios': [2, 1, 1]})

        axs[0, 0].plot(x, y, color='0.75', lw=1)
        axs[0, 0].scatter(x[cand_win], y[cand_win], s=8, color='tab:blue', alpha=0.8, label='candidate')
        axs[0, 0].scatter(x[0], y[0], s=35, color='green', label='window start')
        axs[0, 0].scatter(x[-1], y[-1], s=35, color='red', label='window end')
        Arena(ax = axs[0, 0], condition = condition,
            barrier_coordinates = tracking_data.barrier_loc[:-1], full_image = False)

        axs[0, 1].scatter(x[man_win], y[man_win], s=8, color='tab:orange', alpha=0.8, label='manual')
        axs[0, 1].plot(x, y, color='0.85', lw=1)
        Arena(ax = axs[0, 1], condition = condition,
            barrier_coordinates = tracking_data.barrier_loc[:-1], full_image = False)

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

def save_removed_runs(homings_dict: dict, removed_event_ids: pl.List[int], settings: object) -> None:
    """Save removed runs to dict and to databse"""
    removed_runs = np.zeros(homings_dict['onset_frames'].shape, dtype=bool)
    removed_runs[removed_event_ids] = True
    homings_dict['removed_runs'] = removed_runs

    # find the hexname and load the database
    savepath = os.path.join(settings.session.base_path, settings.session.processed_path, "homings")
    database, do_analysis, hexaname = check_database_for_same_run(
                db_settings={**settings_to_check(settings.settings, ["homing"])},
                results_csv_name=savepath + os.sep + "Homing_database.csv",
                settings=settings.settings,
            )
    # in the row of the database with "name" == hexaname, set the "curated" column to True
    database.irow(database[database['name'] == hexaname].index[0])['homings_curated'] = True
    database.to_csv(savepath+ os.sep + "Homing_database.csv", index=False)

    # save the removed runs to a CSV file
    filename = os.path.join(savepath, "homings_" + hexaname)
    np.savez(os.path.join(filename + "_results.npz"), **v, allow_pickle=True)