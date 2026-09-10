"""A script for plotting the results of the escape pattern tuning analyses."""

import os
from behave_analysis.analyze.EscapePattern.escape_pattern_utils import compute_tuning_stat, parse_residual_string, RANGE_DICT
from behave_analysis.utils.arena_plotting import Arena
from behave_analysis.utils.creating_directories import make_directory
import matplotlib.pyplot as plt
import numpy as np


def plot_escape_tuning(ET, variable, session, cluster_type, Ids, video_df, var1ET=None):
    """The parent function that will deploy the plotting functions based on the variable being analyzed."""
    plot_path = make_directory(
        os.path.join(
            session["base_path"],
            session["processed_path"],
            "models",
            "escape_tuning",
            "plots",
            cluster_type,
            variable.replace(" ", "_").replace(":", "_").replace("&", "_").replace("-", "_"),
        )
    )

    # compute signficance of tuning curves
    stat = "zscore_peak"
    real_stat, shift_stat = compute_tuning_stat(
        stat=stat, shifted_matrix=ET["fr_shift"], shift0=int(np.shape(ET["fr_shift"])[0] / 2), neural_matrix=ET["neural_matrix"], condition=ET["condition_vector"]
    )
    sig_cells = real_stat > np.nanpercentile(shift_stat, 95, axis=0)

    if "residual" in variable:
        # check if figures exist already?
        var1, time1, var2, time2 = parse_residual_string(variable)
        plot_residual_tuning_results(ET, plot_path, Ids, sig_cells, var2, session, video_df, var1ET)
    else:
        plot_tuning_results(ET, plot_path, Ids, sig_cells)


def plot_residual_tuning_results(data, plot_path, Ids, sig_cells, var2, session, video_df, var1ET):
    """Residual tuning is computed as 'residual: <var1> in <time_period1> - <var2> in <time_period2>',
    where var1 is the variable of interest and var2 is the variable being regressed out.
    This function will make a figure for each cluster, with three columns of subplots:
    1. an overlay of the tuning to the variable of interest (e.g. frac_route) in time_period1 (e.g. homing&escape)
        and the residual tuning.
    2. the tuning to the variable that is being regressed out (e.g. 2D position) in time_period2 (e.g. explore)
    3. the tuning to the variable that is being regressed out (e.g. 2D position) in time_period1 (e.g. homing&escape)
    Each row will have these plots for each condition."""

    n_cond = data["fr_full"].shape[0]
    n_neurons = data["fr_full"].shape[1]
    assert n_neurons == len(Ids), "Number of neurons does not match number of Ids provided"

    if var2 == "2D_position":
        x_matrix, y_matrix = bin_xy_position_time2(data, video_df)

    for n in range(n_neurons):

        fig, axs = plt.subplots(nrows=n_cond, ncols=3, figsize=(15, 5 * n_cond))
        rate_map_mappable = None
        trial_pos_mappable = None

        def _safe_limits(arr):
            """Return robust color limits using 95th percentile for vmax."""
            if np.isnan(arr).all():
                return None, None
            vmin_local = np.nanmin(arr)
            vmax_local = np.nanpercentile(arr, 95)
            if not np.isfinite(vmax_local) or vmax_local <= vmin_local:
                vmax_local = np.nanmax(arr)
            return vmin_local, vmax_local

        for c in range(n_cond):
            # plot the overlay of the tuning to the variable of interest and the residual tuning
            vmin = min(np.nanmin(var1ET["fr_full"][:, n, :]), np.nanmin(data["fr_full"][:, n, :]))
            vmax = max(np.nanmax(var1ET["fr_full"][:, n, :]), np.nanmax(data["fr_full"][:, n, :]))
            ax = axs[c, 0] if n_cond > 1 else axs[0]
            bin_centers = (data["bin_edges"][:-1] + data["bin_edges"][1:]) / 2
            ax.plot(bin_centers, var1ET["fr_full"][c, n, :], label="Original Tuning Curve", color="blue")
            ax.plot(bin_centers, data["fr_full"][c, n, :], label="Residual Tuning Curve", color="orange")
            ax.set_xlabel(data["tuning_var"])
            ax.set_ylabel(f"Firing Rate (Hz) + \n in {data['all_conditions'][c]}")
            ax.set_ylim(vmin, vmax + 2)
            ax.legend()
            if sig_cells[c, n]:
                ax.set_title(f"Cluster {Ids[n]} in {data['all_conditions'][c]}: Significant Tuning")

            # plot the tuning curve for var2 (the variable that is being regressed out) in time_period2
            ax = axs[c, 1] if n_cond > 1 else axs[1]
            if var2 == "2D_position":
                mapmin, mapmax = _safe_limits(data["residual_fr_var2_t2"][:, :, :, n])
                rate_map = data["residual_fr_var2_t2"][c, :, :, n]  # shape: (bins, bins)
                Arena(
                    ax=ax,
                    dim=rate_map.shape[0] - 1,
                    condition=data["all_conditions"][c] + ("_tiny" if "tiny" in session["experiment"] else ""),
                    barrier_coordinates=session["barrier_location"][:-1] if session["barrier_location"] is not None else None,
                    shelter_coordinates=session["shelter_location"] if session["shelter_location"] is not None else None,
                    full_image=False,
                )
                if np.isnan(rate_map).all():
                    ax.text(0.5, 0.5, "No data", ha="center", va="center")
                    ax.axis("off")
                else:
                    im = ax.imshow(rate_map, vmin=mapmin, vmax=mapmax, cmap="viridis")
                    if rate_map_mappable is None:
                        rate_map_mappable = im
            else:
                vmin = np.nanmin(data["residual_fr_var2_t2"][:, n, :])
                vmax = np.nanmax(data["residual_fr_var2_t2"][:, n, :])
                ax.plot(data["residual_fr_var2_t2"][c, n, :], label=f"Tuning Curve for {var2} in {data['all_conditions'][c]}")
                ax.set_xlabel(var2)
                ax.set_ylabel(f"Firing Rate (Hz) + \n in {data['all_conditions'][c]}")
                ax.set_ylim(vmin, vmax + 2)
                ax.set_xticks(range(len(data["residual_fr_var2_t2"][c, n, :])))
                ax.set_xticklabels(RANGE_DICT[var2])

            # plot the tuning curve for the variable that is being regressed out in time_period1
            ax = axs[c, 2] if n_cond > 1 else axs[2]
            if var2 == "2D_position":
                mapmin, mapmax = _safe_limits(var1ET["mat_num_cond"][:, n, :, :])
                neural = var1ET["mat_num_cond"][c, n, :, :]  # shape: (trials, bins)
                sc = plot_2D_position_tuning(x_matrix[c, :, :], y_matrix[c, :, :], neural, ax, var1ET["all_conditions"][c], session, mapmin, mapmax)
                if trial_pos_mappable is None and sc is not None:
                    trial_pos_mappable = sc
            else:
                # TODO: not implemented...
                bla = 2

        if var2 == "2D_position":
            if rate_map_mappable is not None:
                cbar0 = fig.colorbar(rate_map_mappable, ax=axs[:, 1] if n_cond > 1 else axs[1], fraction=0.025, pad=0.02, shrink=0.8)
                cbar0.set_label("Rate")
            if trial_pos_mappable is not None:
                cbar1 = fig.colorbar(trial_pos_mappable, ax=axs[:, 2] if n_cond > 1 else axs[2], fraction=0.025, pad=0.02, shrink=0.8)
                cbar1.set_label("Rate")

        fig.savefig(os.path.join(plot_path, f"cluster_{Ids[n]}_residual_tuning.png"))
        plt.close(fig)


def plot_tuning_results(data, plot_path, Ids, sig_cells):
    """This function will make a figure for each cluster, with two columns of subplots:
    1. the tuning to the variable of interest (e.g. frac_route) in the time_period specified (e.g. homing&escape)
    2. the response on single trials (if available) to the variable of interest in the time_period specified.
    Each row will have these plots for each condition."""

    n_cond = data["fr_full"].shape[0]
    n_neurons = data["fr_full"].shape[1]
    assert n_neurons == len(Ids), "Number of neurons does not match number of Ids provided"

    for n in range(n_neurons):

        fig, axs = plt.subplots(nrows=n_cond, ncols=2, figsize=(10, 5 * n_cond))
        vmin = 0
        vmax = data["fr_full"][:, n, :].max()

        for c in range(n_cond):
            # plot the tuning curve
            ax = axs[c, 0] if n_cond > 1 else axs[0]
            bin_centers = (data["bin_edges"][:-1] + data["bin_edges"][1:]) / 2
            ax.plot(bin_centers, data["fr_full"][c, n, :], label="Tuning Curve")
            ax.set_xlabel(data["tuning_var"])
            ax.set_ylabel(f"Firing Rate (Hz) + \n in {data['all_conditions'][c]}")
            ax.set_ylim(vmin, vmax + 2)
            if sig_cells[c, n]:
                ax.set_title(f"Cluster {Ids[n]} in {data['all_conditions'][c]}: Significant Tuning")

            # make heatmap of single trial activity
            ax = axs[c, 1] if n_cond > 1 else axs[1]
            single_trial = data["mat_num_cond"][c, n, :, :]
            nan_trials = np.where(np.sum(np.isnan(single_trial), axis=1) == data["mat_num_cond"].shape[3])[0]
            if len(nan_trials) > 0:
                single_trial = np.delete(single_trial, nan_trials, axis=0)
            ax.imshow(single_trial, aspect="auto", cmap="gray_r", vmin=vmin, vmax=vmax, interpolation="none")
            ax.set_ylabel(f"Trials in {data['all_conditions'][c]}")
            # set x ticks to be the first bin center, the last bin center and two evenly spaced ticks in between
            ax.set_xticks([0, len(data["bin_edges"]) - 1, len(data["bin_edges"]) // 3, 2 * len(data["bin_edges"]) // 3])
            # round appearance to two numbers after the comma
            ax.set_xticklabels(
                [
                    f"{data['bin_edges'][0]:.2f}",
                    f"{data['bin_edges'][-1]:.2f}",
                    f"{data['bin_edges'][len(data['bin_edges'])//3]:.2f}",
                    f"{data['bin_edges'][2*len(data['bin_edges'])//3]:.2f}",
                ]
            )

        fig.savefig(os.path.join(plot_path, f"cluster_{Ids[n]}_tuning.png"))
    pass


def plot_2D_position_tuning(x_matrix, y_matrix, neural, ax, condition, session, vmin, vmax):
    from behave_analysis.utils.arena_plotting import Arena

    Arena(ax=ax, condition=condition, shelter_coordinates=session["shelter_location"], barrier_coordinates=session["barrier_location"])

    tr_tot = 0
    sc = None
    for tr in range(x_matrix.shape[0]):
        x_tr = x_matrix[tr, :]
        y_tr = y_matrix[tr, :]
        a_tr = neural[tr, :]  # neural activity for same trial/bin points

        # keep only valid points across x, y, and activity
        m = (~np.isnan(x_tr)) & (~np.isnan(y_tr)) & (~np.isnan(a_tr))
        if not np.any(m):
            continue
        tr_tot += 1
        sc = ax.scatter(x_tr[m], y_tr[m], c=a_tr[m], s=12, cmap="viridis", vmin=vmin, vmax=vmax)

    return sc


def bin_xy_position_time2(data, video_df):
    # bin the xy positions at the same rate as the ep_data so that we can plot the activity on each trial by frac route on the positions
    tuning_onsets = (((np.where(np.diff(data["homing_vector"].astype(int)) == 1))[0] + 1) / 2).astype(int)
    tuning_offsets = (((np.where(np.diff(data["homing_vector"].astype(int)) == -1))[0] + 1) / 2).astype(int)
    current_time = np.arange(len(video_df))
    multiplier = len(data["homing_vector"]) / len(video_df)
    new_time = np.arange(0, len(video_df), 1 / 2)
    y = np.interp(new_time, current_time, video_df["mouse_y_position"].to_numpy())
    x = np.interp(new_time, current_time, video_df["mouse_x_position"].to_numpy())
    assert np.equal(len(x), len(data["homing_vector"])), "Length of x does not match length of data['discretized_var']"
    offset = 0
    x_matrix = np.full(
        (data["mat_num_cond"].shape[0], data["mat_num_cond"].shape[2], data["mat_num_cond"].shape[3]), np.nan
    )  # shucks, here we gotta figure out which condition each trial is in
    y_matrix = np.full(
        (data["mat_num_cond"].shape[0], data["mat_num_cond"].shape[2], data["mat_num_cond"].shape[3]), np.nan
    )  # shucks, here we gotta figure out which condition each trial is in
    for on, off in zip(tuning_onsets, tuning_offsets):
        binned_route = data["discretized_var"][offset : offset + ((off - on) * 2)].squeeze()
        condition = int(data["condition_vector"][offset])
        next_trial = np.where(np.isnan(x_matrix[condition, :, 0]))[0][0]
        offset += (off - on) * 2
        x_route = x[on * 2 : off * 2]
        y_route = y[on * 2 : off * 2]
        for i in np.unique(binned_route).astype(int):
            x_matrix[condition, next_trial, i] = np.mean(x_route[binned_route == i])
            y_matrix[condition, next_trial, i] = np.mean(y_route[binned_route == i])
    return x_matrix, y_matrix
