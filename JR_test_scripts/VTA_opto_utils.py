"""VTA opto plotting and utils functions"""

import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd


def plot_trial_length_over_session(dot_info, fig_save_path):
    """Plot the length of each trial over the session."""
    # this is how long the dot was present. The dot disappears once the mouse has been on it for 1s.
    trial_length = dot_info["TrialEnd"].to_numpy() - dot_info["TrialStart"].to_numpy()
    plt.plot(trial_length)
    plt.xlabel("Trial Number")
    plt.ylabel("Trial Length (s)")
    plt.savefig(os.path.join(fig_save_path, "trial_length_over_session.png"))
    plt.close()

def plot_mean_distance_to_dot(dot_info, mouse_location, fig_save_path):
    """Mean distance of mouse to dot on each trial"""
    mean_dot_distance = np.empty(len(dot_info))
    # check how many frames are in each trial to make sure the array is big enough
    time_points = np.empty(len(dot_info))
    for trial in range(len(dot_info)):
        trial_start = dot_info["TrialStart"][trial]
        trial_end = dot_info["TrialEnd"][trial]
        trial_data = mouse_location[(mouse_location["Seconds"] >= trial_start) & (mouse_location["Seconds"] <= trial_end)]
        time_points[trial] = len(trial_data)
    trial_dot_distance = np.full((int(max(time_points)), len(dot_info)), np.nan)
    # iterate over trials and calculate the distance of the mouse to the dot for each time point in the trial
    for trial in range(len(dot_info)):
        trial_start = dot_info["TrialStart"][trial]
        trial_end = dot_info["TrialEnd"][trial]
        trial_data = mouse_location[(mouse_location["Seconds"] >= trial_start) & (mouse_location["Seconds"] <= trial_end)]
        for i in range(len(trial_data)):
            x1, y1 = trial_data.iloc[i]["Value.X"], trial_data.iloc[i]["Value.Y"]
            trial_dot_distance[i, trial] = ((x1 - dot_info["DotXLocation"][trial]) ** 2 + (y1 - dot_info["DotYLocation"][trial]) ** 2) ** 0.5
        mean_dot_distance[trial] = np.nanmean(trial_dot_distance[:, trial])
    plt.plot(mean_dot_distance)
    # plot the moving mean of the mean distance to the dot with a window size of 10 trials
    window_size = 10
    xpoints = np.arange(len(mean_dot_distance) - window_size + 1) + window_size // 2
    moving_mean = np.convolve(mean_dot_distance, np.ones(window_size) / window_size, mode="valid")
    plt.plot(xpoints, moving_mean)
    plt.xlabel("Trial Number")
    plt.ylabel("Mean Distance to Dot (norm a.u.)")
    plt.savefig(os.path.join(fig_save_path, "mean_distance_to_dot.png"))
    plt.close()

def probability_of_stopping_on_dot_over_time(dot_info, mouse_location):
    """Probability of stopping on the dot over time"""
    distance = np.full(len(mouse_location), np.nan)
    fake_distance = np.full(len(mouse_location), np.nan)
    for i in range(len(mouse_location)):
        x1, y1 = mouse_location.iloc[i]["Value.X"], mouse_location.iloc[i]["Value.Y"]
        s = mouse_location.iloc[i]["Seconds"]
        which_trial = np.where((dot_info["TrialStart"] <= s) & (dot_info["TrialEnd"] >= s))[0]
        if len(which_trial) > 0:
            trial = which_trial[0]
            distance[i] = ((x1 - dot_info["DotXLocation"][trial]) ** 2 + (y1 - dot_info["DotYLocation"][trial]) ** 2) ** 0.5
            trial = np.random.randint(0, len(dot_info))
            fake_distance[i] = ((x1 - dot_info["DotXLocation"][trial]) ** 2 + (y1 - dot_info["DotYLocation"][trial]) ** 2) ** 0.5
    return distance, fake_distance


def compute_speed_var(mouse_location):
    """Compute speed"""
    # 1) Smooth positions to reduce high-frequency jitter
    # Use a rolling median or moving average; odd window is easiest
    window = 10  # ~0.2 s at 50 fps
    x_smooth = pd.Series(mouse_location["Value.X"]).rolling(window, center=True, min_periods=1).median().to_numpy()
    y_smooth = pd.Series(mouse_location["Value.Y"]).rolling(window, center=True, min_periods=1).median().to_numpy()

    # 2) Differentiate the smoothed position to get velocity
    vx = np.gradient(x_smooth, mouse_location["Seconds"].to_numpy(dtype=float), edge_order=1)
    vy = np.gradient(y_smooth, mouse_location["Seconds"].to_numpy(dtype=float), edge_order=1)

    # 3) Speed magnitude
    speed = np.hypot(vx, vy)

    # 4) Optional extra smoothing on speed itself
    speed = pd.Series(speed).rolling(5, center=True, min_periods=1).median().to_numpy()

    # acceleration = change in speed over time
    accel = np.gradient(speed, mouse_location["Seconds"].to_numpy(dtype=float), edge_order=1)

    # deceleration is only negative acceleration
    decel = np.clip(-accel, 0, None)

    return speed, accel, decel


def plot_speed_distribution(speed, distance, fake_distance, fig_save_path):
    """Speed distribution for close vs far from dot"""
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    h = np.histogram(speed[distance > 0.22], bins=np.arange(0, 3, 0.05))
    h2 = np.histogram(speed[distance <= 0.22], bins=np.arange(0, 3, 0.05))
    axs[0].plot(h[1][:-1], h[0] / np.sum(h[0]), label="far from dot", color="blue")
    axs[0].plot(h2[1][:-1], h2[0] / np.sum(h2[0]), label="close to dot", color="orange")
    axs[0].legend()
    axs[0].set_xlabel("Speed")
    axs[0].set_ylabel("Probability")
    axs[0].set_title("Real distance to dot")
    h = np.histogram(speed[fake_distance > 0.22], bins=np.arange(0, 3, 0.05))
    h2 = np.histogram(speed[fake_distance <= 0.22], bins=np.arange(0, 3, 0.05))
    axs[1].plot(h[1][:-1], h[0] / np.sum(h[0]), label="far from dot", color="blue")
    axs[1].plot(h2[1][:-1], h2[0] / np.sum(h2[0]), label="close to dot", color="orange")
    axs[1].legend()
    axs[1].set_xlabel("Speed")
    axs[1].set_ylabel("Probability")
    axs[1].set_title("Distance to fake dot")
    fig.savefig(os.path.join(fig_save_path, "speed_distribution_close_vs_far_from_dot.png"))
    plt.close()

def speed_over_time_around_dot_entry(speed, distance, fake_distance, fig_save_path):
    """Look at speed profile when mouse approaches dot"""
    before = 150
    after = 150
    dot_entering = np.where(np.diff((distance <= 0.22).astype(int)) == 1)[0]
    speed_in_dot = np.full((before + after, len(dot_entering)), np.nan)
    distance_in_dot = np.full((before + after, len(dot_entering)), np.nan)
    for i, de in enumerate(dot_entering):
        speed_this_trial = speed[de - before : de + after]
        distance_this_trial = distance[de - before : de + after]
        speed_in_dot[:len(speed_this_trial), i] = speed_this_trial
        distance_in_dot[:len(distance_this_trial), i] = distance_this_trial
    # plt.imshow(speed_in_dot.T, aspect='auto', cmap='viridis')
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(np.nanmean(speed_in_dot, axis=1), color="blue", label="Mean Speed at dot entry")
    ax[1].plot(np.nanmean(distance_in_dot, axis=1), color="blue", label="Mean Distance at dot entry")
    dot_entering = np.where(np.diff((fake_distance <= 0.22).astype(int)) == 1)[0]
    speed_in_dot = np.full((before + after, len(dot_entering)), np.nan)
    distance_in_dot = np.full((before + after, len(dot_entering)), np.nan)
    for i, de in enumerate(dot_entering):
        speed_this_trial = speed[de - before : de + after]
        distance_this_trial = fake_distance[de - before : de + after]
        speed_in_dot[:len(speed_this_trial), i] = speed_this_trial
        distance_in_dot[:len(distance_this_trial), i] = distance_this_trial
    ax[0].plot(np.nanmean(speed_in_dot, axis=1), color="orange", label="Mean Speed at fake dot entry")
    ax[1].plot(np.nanmean(distance_in_dot, axis=1), color="orange", label="Mean Distance at fake dot entry")
    ax[0].axvline(before, color="red", linestyle="--", label="Dot Entry")
    ax[0].set_xlabel("Time (frames)")
    ax[0].set_ylabel("Speed")
    ax[0].legend()
    ax[1].axvline(before, color="red", linestyle="--", label="Dot Entry")
    ax[1].set_xlabel("Time (frames)")
    ax[1].set_ylabel("Distance")
    fig.savefig(os.path.join(fig_save_path, "speed_and_distance_around_dot_entry.png"))
    plt.close()

def whole_arena_occupancy_map(mouse_location, dot_info, fig_save_path):
    """Whole arena spatial occupancy map"""
    from scipy.ndimage import gaussian_filter

    bin_size = 0.1
    arena_radius = 1
    nbins = int((arena_radius * 2) / bin_size)
    one_sided_bins = np.arange(0, arena_radius + bin_size, bin_size)
    bins = np.concatenate((-one_sided_bins[::-1], one_sided_bins[1:]))
    x_bins = pd.cut(mouse_location["Value.X"], bins=bins, labels=False, retbins=False)
    y_bins = pd.cut(mouse_location["Value.Y"], bins=bins, labels=False, retbins=False)

    fps = 40  # set to your video frame rate
    sigma = 1.5  # smoothing in bins

    # attach bin labels to the dataframe
    df = mouse_location.copy()
    df["xbins"] = x_bins
    df["ybins"] = y_bins

    # keep only frames that landed in a valid bin
    df = df.dropna(subset=["xbins", "ybins"]).copy()
    df["xbins"] = df["xbins"].astype(int)
    df["ybins"] = df["ybins"].astype(int)

    nbins = len(one_sided_bins) * 2 - 1

    # count frames per bin, then convert to seconds
    occ = df.groupby(["ybins", "xbins"]).size().unstack(fill_value=0).reindex(index=range(nbins), columns=range(nbins), fill_value=0).to_numpy(dtype=float)
    occ_seconds = occ / fps

    # optional circular arena mask
    yy, xx = np.indices(occ_seconds.shape)
    center = (nbins - 1) / 2
    radius_bins = arena_radius / bin_size
    circular_mask = (xx - center) ** 2 + (yy - center) ** 2 <= radius_bins**2
    occ_seconds[~circular_mask] = np.nan

    # smooth while preserving NaN mask
    nan_mask = np.isnan(occ_seconds)
    occ_filled = np.nan_to_num(occ_seconds, nan=0.0)
    occ_smooth = gaussian_filter(occ_filled, sigma=sigma)
    occ_smooth[nan_mask] = np.nan

    plt.imshow(occ_smooth, origin="lower", extent=(-arena_radius, arena_radius, -arena_radius, arena_radius), cmap="viridis")
    unique_dots = dot_info[["DotXLocation", "DotYLocation"]].drop_duplicates()
    for i, row in unique_dots.iterrows():
        plt.scatter(row["DotXLocation"], row["DotYLocation"], color="red", s=10, label="Dot Locations")
        # add a circle around the dot location with radius 0.22
        circle = plt.Circle((row["DotXLocation"], row["DotYLocation"]), 0.22, color="red", fill=False, linestyle="--", linewidth=1)
        plt.gca().add_patch(circle)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.savefig(os.path.join(fig_save_path, "spatial_occupancy_map.png"))
    plt.close()


def trajectory_by_trial(dot_info, mouse_location, fig_save_path):
    """Trajectory plot"""
    # make color map for number of trials
    ntrials = len(dot_info)
    cmap = plt.get_cmap("viridis", ntrials)

    cols = int(np.ceil(np.sqrt(ntrials)))
    rows = int(np.ceil(ntrials / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for i in range(rows * cols):
        ts = dot_info["TrialStart"].to_numpy()[i] if i < ntrials else None
        te = dot_info["TrialEnd"].to_numpy()[i] if i < ntrials else None
        if ts is None or te is None:
            axs[i // cols, i % cols].axis("off")
            continue
        trial_data = mouse_location[(mouse_location["Seconds"] >= ts) & (mouse_location["Seconds"] <= te)]
        ax = axs[i // cols, i % cols] if rows > 1 else axs[i % cols]
        ax.plot(trial_data["Value.X"], trial_data["Value.Y"], color=cmap(np.random.randint(0, ntrials)), alpha=0.5)

        if i > 0:
            ax.scatter(dot_info["DotXLocation"].to_numpy()[i - 1], dot_info["DotYLocation"].to_numpy()[i - 1], color="grey", s=10, label="Dot Locations")
            # add a circle around the dot location with radius 0.22
            circle = plt.Circle(
                (dot_info["DotXLocation"].to_numpy()[i - 1], dot_info["DotYLocation"].to_numpy()[i - 1]), 0.22, color="grey", fill=False, linestyle="--", linewidth=1
            )
            ax.add_patch(circle)
        ax.scatter(dot_info["DotXLocation"].to_numpy()[i], dot_info["DotYLocation"].to_numpy()[i], color="red", s=10, label="Dot Locations")
        # add a circle around the dot location with radius 0.22
        circle = plt.Circle((dot_info["DotXLocation"].to_numpy()[i], dot_info["DotYLocation"].to_numpy()[i]), 0.22, color="red", fill=False, linestyle="--", linewidth=1)
        ax.add_patch(circle)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim([-1, 1])
        ax.set_xlim([-1, 1])
        ax.set_box_aspect(1)

    plt.savefig(os.path.join(fig_save_path, "single_trial_trajectory.png"))
    plt.close()


def trajectory_by_dot_location(dot_info, mouse_location, fig_save_path):
    """Trajectory plot"""
    dot_xy = np.vstack([dot_info["DotXLocation"].to_numpy(), dot_info["DotYLocation"].to_numpy()])
    dot_start = dot_info["TrialStart"].to_numpy()
    dot_end = dot_info["TrialEnd"].to_numpy()
    idx, dots = np.unique(dot_xy, axis=1, return_inverse=True)
    ndots = len(idx[0])

    cols = int(np.ceil(np.sqrt(ndots)))
    rows = int(np.ceil(ndots / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    for n in range(rows*cols):
        ntrials = np.where(dots == n)[0]
        ax = axs[n // cols, n % cols] if rows > 1 else axs[n]
        if len(ntrials) == 0:
            ax.axis('off')
            continue
        ntrials = np.where(dots == n)[0]
        cmap = plt.get_cmap('viridis', len(ntrials) if len(ntrials) > 3 else 3)
        all_ts = dot_start[ntrials]
        all_te = dot_end[ntrials]
        for i, (ts, te) in enumerate(zip(all_ts, all_te)):
            trial_data = mouse_location[(mouse_location["Seconds"] >= ts) & (mouse_location["Seconds"] <= te)]
            ax.plot(trial_data["Value.X"], trial_data["Value.Y"], color=cmap(i), alpha=0.3)

        ax.scatter(idx[0, n], idx[1, n], color="black", s=10, label="Dot Locations")
        # add a circle around the dot location with radius 0.22
        circle = plt.Circle((idx[0, n], idx[1, n]), 0.22, color="black", fill=False, linestyle="--", linewidth=1)
        ax.add_patch(circle)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim([-1, 1])
        ax.set_xlim([-1, 1])
        ax.set_box_aspect(1)
    plt.savefig(os.path.join(fig_save_path, "all_trial_trajectory.png"))
    plt.close()