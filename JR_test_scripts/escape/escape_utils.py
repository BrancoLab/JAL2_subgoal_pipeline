from behave_analysis.process.process import Process
import os
import polars as pl
import numpy as np


def load(exp):
    # load session
    session = Process(exp).load_session()
    base_path = os.path.join(session.base_path, session.processed_path)

    # spikeys
    # spike_data = pl.read_csv(os.path.join(base_path, "good_spike_data.csv"))

    # matrix
    frame_by_cluster_matrix = np.load(
        os.path.join(session.base_path, session.processed_path)
        + "\\"
        + "frame_by_good_cluster_matrix.npy"
    )

    # behavior
    video_df = pl.read_csv(os.path.join(base_path, "full_video_dataframe.csv"))
    behave = video_df["speed"].to_numpy()
    y_pos = video_df["mouse_y_position"].to_numpy()
    x_pos = video_df["mouse_x_position"].to_numpy()
    bar = video_df["barrier_present"].to_numpy()
    barflip = video_df["barrier_flipped"].to_numpy()
    return session, frame_by_cluster_matrix, behave, y_pos, x_pos, bar, barflip


def compute_escape_trajectory(xpos, ypos):
    # compute cumulative distance travelled at every time point
    distance_travelled = [0]
    for i, stim_status in enumerate(np.arange(len(xpos))):
        if i > 0:
            dist = np.sqrt((xpos[i] - xpos[i - 1]) ** 2 + (ypos[i] - ypos[i - 1]) ** 2)
            distance_travelled = np.append(
                distance_travelled, dist + distance_travelled[-1]
            )
    return distance_travelled


def compute_dist_shelt(x_pos, y_pos, cond, session):
    dist = np.zeros((len(x_pos)))
    shelter = [
        np.mean([session.shelter_location[0][0], session.shelter_location[1][0]]),
        session.shelter_location[0][1],
    ]
    bar1 = session.barrier_location[0]
    bar2 = session.barrier_location[1]
    # measure the distance of the mouse to a point in the top half of arena
    top_barrier = np.logical_and(cond == 1, y_pos < 512)
    dist[top_barrier] = np.sqrt(
        ((x_pos[top_barrier] - bar1[0]) ** 2) + ((y_pos[top_barrier] - bar1[1]) ** 2)
    )
    top_barrierflip = np.logical_and(cond == 2, y_pos < 512)
    dist[top_barrierflip] = np.sqrt(
        ((x_pos[top_barrierflip] - bar2[0]) ** 2)
        + ((y_pos[top_barrierflip] - bar2[1]) ** 2)
    )
    # measure the distance of the mouse to shelt in the bottom half of arena
    dist = dist + np.sqrt(((x_pos - shelter[0]) ** 2) + ((y_pos - shelter[1]) ** 2))
    return dist


def compress_vars(pos, neural_matrix):
    # pos is the variable we're basing the compression on
    # neural_matrix is getting compressed with it
    for i, neural_activity in enumerate(neural_matrix):
        # Step 1: Identify change points
        change_points = (
            np.where(np.diff(pos) != 0)[0] + 1
        )  # Indices where position changes
        change_points = np.insert(
            change_points, 0, 0
        )  # Include the start of the first segment
        change_points = np.append(
            change_points, len(pos)
        )  # Include the end of the last segment

        # Step 2: Compress position and neural activity
        compressed_pos = [pos[start] for start in change_points[:-1]]
        compressed_activity = [
            neural_activity[start:end].mean()  # Example: mean activity for each segment
            for start, end in zip(change_points[:-1], change_points[1:])
        ]

        # Outputs
        if i == 0:
            new_pos = np.array(compressed_pos)
            new_activity = np.array(compressed_activity)
        else:
            new_activity = np.vstack((new_activity, compressed_activity))
    return new_activity, new_pos


def discretize_x_axis(var, bin_size=10):
    bins = np.arange(0, np.amax(var), bin_size)
    disc_var = np.digitize(var, bins)
    return disc_var


def firing_by_bin(var, neural_activity, nbins):
    angles_firing = np.zeros(nbins)
    unique_groups, group_counts = np.unique(var, return_counts=True)
    # mean firing
    group_sums = np.bincount(var, weights=neural_activity)
    angles_firing[unique_groups] = group_sums[unique_groups] / group_counts
    return angles_firing[unique_groups]
