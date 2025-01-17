import pickle
import os
from pathlib import Path
from collections import defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pandas as pd
from sklearn.model_selection import GroupKFold
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from behave_analysis.utils.creating_directories import make_directory
from behave_analysis.visualize.visualize_utils import open_tracking_data
from behave_analysis.process.session import get_experiment
from behave_analysis.utils.arena_plotting import Arena
from behave_analysis.database.Experiments.JAL003_ex import JAL3_25aug, JAL3_1sept, JAL3_4sept, JAL3_7sept
from behave_analysis.database.Experiments.JAL004_ex import JAL4_3rdSept, JAL4_19thSept, JAL4_28aug, JAL4_11thSept
from behave_analysis.database.Experiments.JAL005_ex import JAL005_8thSept, JAL005_21stSept
from behave_analysis.database.Experiments.JAL006_ex import JAL6_28mar, JAL6_flip4_21mar, JAL6_flip5_25mar, JAL6_flip3_18mar, JAL6_flip7_1apr
from behave_analysis.database.Experiments.JAL007_ex import (
    JAL7_sesh8_9apr,
    JAL7_sesh9_16apr,
    JAL7_flip5_22mar,
    JAL7_flip2_12mar,
    JAL7_23apr,
    JAL7_30apr,
)
from behave_analysis.database.Experiments.JAL008_ex import (
    JAL8_flip1_25apr,
    JAL8_flip2_29apr,
    JAL8_tiny_3may,
    JAL8_flip4_10may,
    JAL8_14may,
    JAL8_21may,
    JAL8_flip3_7may,
)

# The values must be in order of data for the code to work
mice_groups = {
    "JAL6": ["JAL6_flip3_18mar", "JAL6_flip4_21mar", "JAL6_flip5_25mar", "JAL6_28mar"],
    "JAL3": ["JAL3_25aug", "JAL3_1sept", "JAL3_4sept", "JAL3_7sept"],
    "JAL7": ["JAL7_flip2_12mar", "JAL7_flip5_22mar", "JAL7_sesh8_9apr", "JAL7_sesh9_16apr", "JAL7_23apr"],
    "JAL8": ["JAL8_flip1_25apr", "JAL8_flip2_29apr", "JAL8_flip4_10may", "JAL8_flip3_7may", "JAL8_14may"],
    "JAL4": ["JAL4_28aug", "JAL4_3rdSept", "JAL4_11thSept", "JAL4_19thSept"],
    "JAL5": ["JAL5_8thSept", "JAL5_21stSept"],
}

# The experiment objects index must match the session name index for the code to work
experiments_objects = [
    JAL6_flip3_18mar,
    JAL6_flip4_21mar,
    JAL6_flip5_25mar,
    JAL6_28mar,
    JAL3_25aug,
    JAL3_1sept,
    JAL3_4sept,
    JAL3_7sept,
    JAL005_8thSept,
    JAL005_21stSept,
    JAL7_sesh8_9apr,
    JAL7_sesh9_16apr,
    JAL7_flip5_22mar,
    JAL7_flip2_12mar,
    JAL7_23apr,
    JAL8_flip1_25apr,
    JAL8_flip2_29apr,
    JAL8_flip4_10may,
    JAL8_14may,
    JAL8_flip3_7may,
    JAL4_3rdSept,
    JAL4_19thSept,
    JAL4_28aug,
    JAL4_11thSept,
]

# The session name indexes must match the experiment object indexes
session_names = [
    "JAL6_flip3_18mar",
    "JAL6_flip4_21mar",
    "JAL6_flip5_25mar",
    "JAL6_28mar",
    "JAL3_25aug",
    "JAL3_1sept",
    "JAL3_4sept",
    "JAL3_7sept",
    "JAL5_8thSept",
    "JAL5_21stSept",
    "JAL7_sesh8_9apr",
    "JAL7_sesh9_16apr",
    "JAL7_flip5_22mar",
    "JAL7_flip2_12mar",
    "JAL7_23apr",
    "JAL8_flip1_25apr",
    "JAL8_flip2_29apr",
    "JAL8_flip4_10may",
    "JAL8_14may",
    "JAL8_flip3_7may",
    "JAL4_3rdSept",
    "JAL4_19thSept",
    "JAL4_28aug",
    "JAL4_11thSept",
]

# ------------------- GLOBALS -------------------

dir = make_directory(r"Z:\Jasmine_Laurence\single_trial_overview\decoding_spatial_efficiency")
dir = Path(dir)
SE_THRESHOLD = 0.90

# ------------------- FUNCTIONS -------------------


def load(dir, file_name):
    """Ensure you include the .pkl in the file_name"""
    dir = dir / file_name
    with open(dir, "rb") as f:
        file = pickle.load(f)
    return file


def load_video_data(experiment):
    """Loads the video data for the experiment"""
    loaded_session = get_experiment(experiment)
    try:
        video_df = pl.read_csv(os.path.join(loaded_session.base_path, loaded_session.processed_path) + "\\" "full_video_dataframe.csv")

    except FileNotFoundError:
        print("One of the files was not found")
        return None

    return video_df


def extract_behavioural_data_between_homing_onset_offset(homing_object: dict, video_df: pl.DataFrame) -> list:
    """Returns a list of behavioural dataframes for each homing event in the homing object."""
    homing_info = []
    for onset, offset in zip(homing_object.onset_frames, homing_object.offset_frames):
        homing = video_df[int(onset) - 1 : int(offset)]  # Frames -1 because of 0 indexing,
        homing = homing.select(
            [
                "frames",
                "mouse_x_position",
                "mouse_y_position",
                "hdir",
                "hsa",
                "h_preflipbar_a",
                "h_postflipbar_a",
            ]
        )
        homing_info.append(homing)
    return homing_info


def add_homing_id_to_homing_data(extracted_homing_info: list) -> list:
    """Adding the homing id (abitrary ascending interger) to the homing data.
    This is needed for the group cross validation object

    Args:
        extracted_homing_info (list): A list of homing dataframes for each homing period

    Returns:
        (list) of homing dataframes with the homing id added as a column
    """
    for idx, homing in enumerate(extracted_homing_info):
        updated_homing = homing.with_columns(pl.lit(idx).alias("homing_id"))
        extracted_homing_info[idx] = updated_homing
    return extracted_homing_info


def create_the_design_matrix(homing_data: pl.DataFrame, frame_by_cluster_matrix: np.ndarray, classes: list) -> np.ndarray:
    spike_data_per_homing = []
    classes_extended = []
    homing_ids = []
    for i, homing in enumerate(homing_data):
        first_frame = homing["frames"][0] - 1  # Frames -1 because of 0 indexing
        last_frame = homing["frames"][-1]  # Do not minus 1 for last frame as it is not inclusive in the slicing
        homing_id = homing["homing_id"].to_numpy().reshape(-1, 1)
        homing_ids.append(homing_id)
        spike_data = frame_by_cluster_matrix[first_frame:last_frame]
        spike_data_per_homing.append(spike_data)
        classes_extended.append([classes[i]] * (last_frame - first_frame))
        assert len(spike_data) == len(classes_extended[i])
    design_matrix = np.vstack(spike_data_per_homing)  # vertically stack the spike data for each homing into a design matrix
    return design_matrix, classes_extended, homing_ids


def create_the_past_design_matrix(homing_data: pl.DataFrame, frame_by_cluster_matrix: np.ndarray, classes: list, shift: int = 20) -> np.ndarray:
    """Pull neural data from before the homing onset by a certain amount of frames

    ARgs:
        shift (int) the number of frames to shift the neural data by to look before the homing"""
    spike_data_per_homing = []
    classes_extended = []
    homing_ids = []
    frame_numbers = []
    for i, homing in enumerate(homing_data):
        first_frame = max(homing["frames"][0] - 1 - shift, 0)  # Ensure non-negative index
        last_frame = homing["frames"][0] - 1  # The last frame is the first frame of the homing, -1 because of 0 indexing frames start at 1

        # Extract homing IDs and flatten appropriately
        homing_ids.extend(homing["homing_id"][: last_frame - first_frame])  # Extend with flattened values

        # Extract the spike data
        spike_data = frame_by_cluster_matrix[first_frame:last_frame]
        spike_data_per_homing.append(spike_data)

        classes_extended.append([classes[i]] * (last_frame - first_frame))
        assert len(spike_data) == len(classes_extended[i])
        frames = np.arange(first_frame, last_frame)
        frame_numbers.extend(frames)
    design_matrix = np.vstack(spike_data_per_homing)  # vertically stack the spike data for each homing into a design matrix

    # Vertically append the frame numbers to the design matrix
    frame_numbers = np.array(frame_numbers).reshape(-1, 1)
    design_matrix = np.hstack((design_matrix, frame_numbers))

    return design_matrix, classes_extended, homing_ids


def create_the_random_design_matrix(homing_data: pl.DataFrame, frame_by_cluster_matrix: np.ndarray, classes: list, period=200):
    """
    Homing_data (List): Of dataframes containing frames, mouse pos, hdir, hsa, bar angles, homing_id
    """
    spike_data_per_homing = []
    classes_extended = []
    homing_ids = []
    M, N = frame_by_cluster_matrix.shape  # M is number of frames in entire session
    frames = np.arange(0, M - period)  # Ensure range is valid

    for i, homing in enumerate(homing_data):
        first_frame = np.random.choice(frames)
        last_frame = first_frame + period

        # Validate slice bounds
        if last_frame > M:
            continue

        spike_data = frame_by_cluster_matrix[first_frame:last_frame]
        spike_data_per_homing.append(spike_data)

        # Append consistent homing IDs and classes
        id = homing["homing_id"][0]
        homing_ids.append([id] * period)
        classes_extended.append([classes[i]] * period)

    design_matrix = np.vstack(spike_data_per_homing)

    return design_matrix, np.array(classes_extended), np.array(homing_ids)


def create_histograms(homings_above_the_barrier, classes2):
    """Creates histograms for mouse_x_position, mouse_y_position, hdir, and hsa,
    comparing the two classes, with separate colors for each class."""

    metrics = ["mouse_x_position", "mouse_y_position", "hdir", "hsa"]
    colors = {0: "blue", 1: "red"}  # Define colors for classes

    for metric in metrics:
        class_0_data = []
        class_1_data = []

        for dataframe, cls in zip(homings_above_the_barrier, classes2):
            if cls == 0:
                class_0_data.extend(dataframe[metric].to_list())
            elif cls == 1:
                class_1_data.extend(dataframe[metric].to_list())

        plt.figure()
        plt.hist(class_0_data, bins=30, alpha=0.7, label="Class 0", color=colors[0])
        plt.hist(class_1_data, bins=30, alpha=0.7, label="Class 1", color=colors[1])
        plt.title(f"Histogram of {metric}")
        plt.xlabel(metric)
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()


def control_for_500ms_before_homing(experiments_objects, session_names, dir):

    # Initialize distributions for Class 0 and Class 1
    class_0_dist = []
    class_1_dist = []

    # First get homing onsets across all sessions
    for experiment, session_name in zip(experiments_objects, session_names):
        print(f"Running 500ms control before homing for session: {session_name}")
        loaded_session = get_experiment(experiment)

        # Set paths
        base_path = loaded_session.base_path
        processed_path = loaded_session.processed_path
        session_path = os.path.join(base_path, processed_path)
        homing_path = os.path.join(session_path, "homings", "homings_obj.pkl")

        # Load homing onset and video data
        try:
            video_df = pd.read_csv(os.path.join(loaded_session.base_path, loaded_session.processed_path, "full_video_dataframe.csv"))
            with open(homing_path, "rb") as hf:
                homings_object = pickle.load(hf)
        except FileNotFoundError:
            print(f"Skipping session {session_name} because homing or video data is missing")
            continue

        # Extract homing data
        onsets = homings_object.onset_frames  # list of onset frames
        homing_class = [1 if seff > SE_THRESHOLD else 0 for seff in homings_object.spatial_efficiency]  # List of classes

        # Extract behavioral data for each homing event
        for onset, cls in zip(onsets, homing_class):
            first_frame = onset - 20  # 500ms before homing onset
            behaviour = video_df.iloc[first_frame:onset]

            # Append X positions to the respective class distribution
            if cls == 1:
                class_1_dist.extend(behaviour["mouse_x_position"].values)
            else:
                class_0_dist.extend(behaviour["mouse_x_position"].values)

        # Plot the distributions
        plt.figure(figsize=(10, 6))
        plt.hist(class_0_dist, bins=30, alpha=0.7, color="blue", label="Class 0")
        plt.hist(class_1_dist, bins=30, alpha=0.7, color="red", label="Class 1")
        plt.title("Comparison of Mouse X Position Distributions Before Down Sampling for Class 0 and Class 1 500ms Before Homing")
        plt.suptitle(f"Session: {session_name}")
        plt.xlabel("Position")
        plt.ylabel("Frequency")
        plt.legend()

        # Dir
        save_path = dir / "500ms_before_control"
        make_directory(save_path)
        plt.savefig(save_path / f"{session_name}.png")
        plt.close()


def good_vs_bad_trajectories_plotted_to_arena(
    barrier_location, tracking_data, homings_above_the_barrier, classes, homing_conditions, dir, session_name
):
    fig, (ax_pre_flip, ax_post_flip) = plt.subplots(1, 2, figsize=(20, 16))
    Arena(ax=ax_pre_flip, shelter_coordinates=tracking_data["shelter_loc"], condition="barrier_pre_flip", barrier_coordinates=barrier_location)
    Arena(ax=ax_post_flip, shelter_coordinates=tracking_data["shelter_loc"], condition="barrier_post_flip", barrier_coordinates=barrier_location)
    for i, homing in enumerate(homings_above_the_barrier):
        if homing_conditions[i] == "barrier_pre_flip":
            if classes[i] == 0:
                ax_pre_flip.plot(homing["mouse_x_position"], homing["mouse_y_position"], color="r", label="Bad")
            if classes[i] == 1:
                ax_pre_flip.plot(homing["mouse_x_position"], homing["mouse_y_position"], color="g", label="Good")
        elif homing_conditions[i] == "barrier_post_flip":
            if classes[i] == 0:
                ax_post_flip.plot(homing["mouse_x_position"], homing["mouse_y_position"], color="r", label="Bad")
            if classes[i] == 1:
                ax_post_flip.plot(homing["mouse_x_position"], homing["mouse_y_position"], color="g", label="Good")

    handles, labels = plt.gca().get_legend_handles_labels()
    unique_handles_labels = dict(zip(labels, handles))
    ax_pre_flip.legend(unique_handles_labels.values(), unique_handles_labels.keys())
    ax_post_flip.legend(unique_handles_labels.values(), unique_handles_labels.keys())
    ax_pre_flip.set_title("Pre Flip Barrier")
    ax_post_flip.set_title("Post Flip Barrier")
    fig.suptitle(f"Good vs Bad Trajectories for {session_name}")

    # Dir
    save_path = dir / "good_vs_bad_trajectories"
    make_directory(save_path)
    plt.savefig(save_path / f"{session_name}.png")
    plt.close()


# ------------------- MAIN -------------------


def produce_data(experiments_objects, session_names, name_of_storage, create_design_matrix):
    storage = defaultdict(defaultdict)
    for experiment, session_name in zip(experiments_objects, session_names):
        print(f"Loading data for session: {session_name}")
        loaded_session = get_experiment(experiment)

        # Set paths
        base_path = loaded_session.base_path
        processed_path = loaded_session.processed_path
        session_path = os.path.join(base_path, processed_path)
        homing_path = os.path.join(session_path, "homings", "homings_obj.pkl")
        
        if session_name == "JAL6_flip5_25mar":
            continue

        # Load data
        try:
            video_df = pl.read_csv(os.path.join(loaded_session.base_path, loaded_session.processed_path) + "\\" "full_video_dataframe.csv")
            with open(homing_path, "rb") as hf:
                homings_object = pickle.load(hf)

            frame_by_cluster_matrix = np.load(
                os.path.join(loaded_session.base_path, loaded_session.processed_path) + "\\" + "frame_by_" + "good" + "_cluster_matrix.npy"
            )

            good_cluster_ids = np.load(os.path.join(session_path, "good_cluster_ids.npy"))

            assert len(good_cluster_ids) == frame_by_cluster_matrix.shape[1], "Cluster IDs do not match the number of clusters in the matrix"

            tracking_data = open_tracking_data(loaded_session)

        except FileNotFoundError:
            print("One of the files was not found")

        #     tracking_data["barrier_loc"] = [[224, 515], [797, 512], [510, 513]]

        # Extract logic data
        barrier_location = tracking_data["barrier_loc"]
        homing_info = extract_behavioural_data_between_homing_onset_offset(homings_object, video_df)
        homing_info = add_homing_id_to_homing_data(homing_info)
        homing_conditions = homings_object.homing_condition
        homing_class = [1 if seff > SE_THRESHOLD else 0 for seff in homings_object.spatial_efficiency]

        # Only keep the homings where the barrier is present
        barrier_homings = [i for i, homing in enumerate(homing_conditions) if homing in ["barrier_pre_flip", "barrier_post_flip"]]
        classes1 = [homing_class[i] for i in barrier_homings]
        homing_info1 = [homing_info[i] for i in barrier_homings]
        homing_conditions1 = [homing_conditions[i] for i in barrier_homings]

        # Only keep the homings above the barrier
        homings_above_the_barrier = [homing for i, homing in enumerate(homing_info1) if homing["mouse_y_position"][0] < barrier_location[0][1]]
        classes2 = [classes1[i] for i, homing in enumerate(homing_info1) if homing["mouse_y_position"][0] < barrier_location[0][1]]
        homing_conditions2 = [
            homing_conditions1[i] for i, homing in enumerate(homing_info1) if homing["mouse_y_position"][0] < barrier_location[0][1]
        ]

        # Plot the trajectories of good and bad homings
        good_vs_bad_trajectories_plotted_to_arena(
            barrier_location, tracking_data, homings_above_the_barrier, classes2, homing_conditions2, dir, session_name
        )

        # Create design matrix
        design_matrix, classes_extended, homing_ids = create_design_matrix(homings_above_the_barrier, frame_by_cluster_matrix, classes2)
        classes_extended = [item for sublist in classes_extended for item in sublist]  # Flatten the classes_extended list

        # Store the data
        storage[session_name]["design_matrix"] = design_matrix
        storage[session_name]["classes_extended"] = classes_extended
        storage[session_name]["homing_ids"] = homing_ids

    # Save the data as pickle
    with open(dir / f"{name_of_storage}.pkl", "wb") as f:
        pickle.dump(storage, f)

    return storage


if __name__ == "__main__":
    produce_data(experiments_objects, session_names, "compute_the_first_20_frames_before_homing", create_the_past_design_matrix)
#    produce_data(experiments_objects, session_names, "compute_entire_spatial_efficiency", create_the_design_matrix)
#    random_times_200_frames_5s = produce_data(experiments_objects, session_names, "random_times_200_frames_5s", create_the_random_design_matrix)
