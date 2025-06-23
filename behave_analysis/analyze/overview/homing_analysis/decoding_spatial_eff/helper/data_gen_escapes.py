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

dir = make_directory(r"Z:\Jasmine_Laurence\single_trial_overview\decoding_spatial_efficiency\escapes")
dir = Path(dir)
SE_THRESHOLD = 0.90  # Spatial efficiency threshold to classify escapes as good or bad

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


def extract_behavioural_data_between_escape_onset_offset(escapes_object, video_df):
    """Returns a list of behavioural dataframes for each escape event in the escapes object."""
    escape_info = []
    for onset, offset in zip(escapes_object.escape_onset_frames, escapes_object.escape_end_frames):
        # Check for NaN values
        if np.isnan(onset) or np.isnan(offset):
            print(f"Skipping escape with NaN onset ({onset}) or offset ({offset})")
            continue

        # Ensure indexes are valid integers
        onset_idx = max(0, int(onset) - 1)  # Ensure it's not negative
        offset_idx = min(int(offset), len(video_df) - 1)  # Ensure it's within bounds

        # Extract data
        escape = video_df[onset_idx:offset_idx]

        # Only process if we have data
        if len(escape) == 0:
            print(f"Skipping escape with no data between frames {onset_idx} and {offset_idx}")
            continue

        # Select relevant columns
        try:
            escape = escape.select(
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
            escape_info.append(escape)
        except Exception as e:
            print(f"Error selecting columns: {e}")
            continue

    return escape_info


def add_escape_id_to_escape_data(extracted_escape_info):
    """Adding the escape id (arbitrary ascending integer) to the escape data.
    This is needed for the group cross validation object

    Args:
        extracted_escape_info (list): A list of escape dataframes for each escape period

    Returns:
        (list) of escape dataframes with the escape id added as a column
    """
    for idx, escape in enumerate(extracted_escape_info):
        updated_escape = escape.with_columns(pl.lit(idx).alias("escape_id"))
        extracted_escape_info[idx] = updated_escape
    return extracted_escape_info


def create_the_design_matrix(escape_data, frame_by_cluster_matrix, classes):
    """Creates a design matrix from escape data and neural data

    Args:
        escape_data (list): List of escape dataframes
        frame_by_cluster_matrix (np.ndarray): Neural data
        classes (list): Class labels (0 for bad escapes, 1 for good escapes)

    Returns:
        design_matrix (np.ndarray): Design matrix
        classes_extended (list): Extended class labels
        escape_ids (list): Escape IDs
    """
    spike_data_per_escape = []
    classes_extended = []
    escape_ids = []
    for i, escape in enumerate(escape_data):
        first_frame = escape["frames"][0] - 1  # Frames -1 because of 0 indexing
        last_frame = escape["frames"][-1]  # Do not minus 1 for last frame as it is not inclusive in the slicing
        escape_id = escape["escape_id"].to_numpy().reshape(-1, 1)
        escape_ids.append(escape_id)
        spike_data = frame_by_cluster_matrix[first_frame:last_frame]
        spike_data_per_escape.append(spike_data)
        classes_extended.append([classes[i]] * (last_frame - first_frame))
        assert len(spike_data) == len(classes_extended[i])
    design_matrix = np.vstack(spike_data_per_escape)  # vertically stack the spike data for each escape into a design matrix
    return design_matrix, classes_extended, escape_ids


def create_the_past_design_matrix(escape_data, frame_by_cluster_matrix, classes, shift=20):
    """Pull neural data from before the escape onset by a certain amount of frames

    Args:
        shift (int): the number of frames to shift the neural data by to look before the escape
    """
    spike_data_per_escape = []
    classes_extended = []
    escape_ids = []
    frame_numbers = []

    for i, escape in enumerate(escape_data):
        try:
            # Check if we have valid frames
            if len(escape["frames"]) == 0:
                print(f"Skipping escape {i} - no frames available")
                continue

            # Get first frame with error handling
            try:
                first_frame_val = escape["frames"][0]
                if isinstance(first_frame_val, float) and np.isnan(first_frame_val):
                    print(f"Skipping escape {i} - first frame is NaN")
                    continue
            except:
                print(f"Skipping escape {i} - error accessing first frame")
                continue

            # Calculate frame indices
            first_frame = max(int(escape["frames"][0]) - 1 - shift, 0)  # Ensure non-negative index
            last_frame = int(escape["frames"][0]) - 1  # The last frame is the first frame of the escape

            # Skip if the window is invalid
            if last_frame <= first_frame:
                print(f"Skipping escape {i} - invalid frame window: {first_frame} to {last_frame}")
                continue

            # Check bounds against frame_by_cluster_matrix
            if last_frame >= frame_by_cluster_matrix.shape[0]:
                print(f"Skipping escape {i} - last frame {last_frame} exceeds matrix bounds {frame_by_cluster_matrix.shape[0]}")
                last_frame = frame_by_cluster_matrix.shape[0] - 1
                if last_frame <= first_frame:
                    continue

            # Extract escape IDs
            escape_id_values = escape["escape_id"].to_list() if len(escape["escape_id"]) > 0 else [i]
            escape_ids.extend(escape_id_values[: last_frame - first_frame])

            # Extract the spike data
            spike_data = frame_by_cluster_matrix[first_frame:last_frame]
            if spike_data.size == 0:
                print(f"Skipping escape {i} - no spike data for frames {first_frame} to {last_frame}")
                continue

            spike_data_per_escape.append(spike_data)

            # Add classes and verify dimensions
            classes_extended.append([classes[i]] * (last_frame - first_frame))
            if len(spike_data) != len(classes_extended[-1]):
                print(f"Warning: length mismatch in escape {i}: spike_data {len(spike_data)} vs classes {len(classes_extended[-1])}")
                # Fix the length mismatch
                min_len = min(len(spike_data), len(classes_extended[-1]))
                classes_extended[-1] = classes_extended[-1][:min_len]

            # Add frame numbers
            frames = np.arange(first_frame, last_frame)
            frame_numbers.extend(frames)

        except Exception as e:
            print(f"Error processing escape {i}: {e}")
            continue

    # Handle case where no valid data was found
    if not spike_data_per_escape:
        print("No valid escape data found to create design matrix")
        return np.array([]), [], []

    # Create design matrix
    design_matrix = np.vstack(spike_data_per_escape)  # vertically stack the spike data

    # Vertically append the frame numbers to the design matrix
    frame_numbers = np.array(frame_numbers).reshape(-1, 1)
    design_matrix = np.hstack((design_matrix, frame_numbers))

    # Flatten classes for consistency
    classes_extended = [item for sublist in classes_extended for item in sublist]

    return design_matrix, classes_extended, escape_ids


def create_the_random_design_matrix(escape_data, frame_by_cluster_matrix, classes, period=200):
    """Create a design matrix from random time periods

    Args:
        escape_data (List): Of dataframes containing frames, mouse pos, hdir, hsa, bar angles, escape_id
        frame_by_cluster_matrix (np.ndarray): Neural data
        classes (list): Class labels
        period (int): Number of frames to sample
    """
    spike_data_per_escape = []
    classes_extended = []
    escape_ids = []
    M, N = frame_by_cluster_matrix.shape  # M is number of frames in entire session
    frames = np.arange(0, M - period)  # Ensure range is valid

    for i, escape in enumerate(escape_data):
        first_frame = np.random.choice(frames)
        last_frame = first_frame + period

        # Validate slice bounds
        if last_frame > M:
            continue

        spike_data = frame_by_cluster_matrix[first_frame:last_frame]
        spike_data_per_escape.append(spike_data)

        # Append consistent escape IDs and classes
        id = escape["escape_id"][0]
        escape_ids.append([id] * period)
        classes_extended.append([classes[i]] * period)

    design_matrix = np.vstack(spike_data_per_escape)

    return design_matrix, np.array(classes_extended), np.array(escape_ids)


def create_histograms(escapes_above_the_barrier, classes2):
    """Creates histograms for mouse_x_position, mouse_y_position, hdir, and hsa,
    comparing the two classes, with separate colors for each class."""

    metrics = ["mouse_x_position", "mouse_y_position", "hdir", "hsa"]
    colors = {0: "blue", 1: "red"}  # Define colors for classes

    for metric in metrics:
        class_0_data = []
        class_1_data = []

        for dataframe, cls in zip(escapes_above_the_barrier, classes2):
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


def control_for_500ms_before_escape(experiments_objects, session_names, dir):
    """Controls for the 500ms before escape onset"""

    # Initialize distributions for Class 0 and Class 1
    class_0_dist = []
    class_1_dist = []

    # First get escape onsets across all sessions
    for experiment, session_name in zip(experiments_objects, session_names):
        print(f"Running 500ms control before escape for session: {session_name}")
        loaded_session = get_experiment(experiment)

        # Set paths
        base_path = loaded_session.base_path
        processed_path = loaded_session.processed_path
        session_path = os.path.join(base_path, processed_path)
        escape_path = os.path.join(session_path, "escapes", "escapes_obj.pkl")

        # Load escape onset and video data
        try:
            video_df = pd.read_csv(os.path.join(loaded_session.base_path, loaded_session.processed_path, "full_video_dataframe.csv"))
            with open(escape_path, "rb") as ef:
                escapes_object = pickle.load(ef)
        except FileNotFoundError:
            print(f"Skipping session {session_name} because escape or video data is missing")
            continue

        # Extract escape data
        onsets = escapes_object.escape_onset_frames  # list of onset frames
        escape_class = [1 if seff > SE_THRESHOLD else 0 for seff in escapes_object.spatial_efficiency]  # List of classes

        # Extract behavioral data for each escape event
        for onset, cls in zip(onsets, escape_class):
            first_frame = onset - 20  # 500ms before escape onset (assuming 40 fps)
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
        plt.title("Comparison of Mouse X Position Distributions Before Down Sampling for Class 0 and Class 1 500ms Before Escape")
        plt.suptitle(f"Session: {session_name}")
        plt.xlabel("Position")
        plt.ylabel("Frequency")
        plt.legend()

        # Save the figure
        save_path = dir / "500ms_before_control"
        make_directory(save_path)
        plt.savefig(save_path / f"{session_name}.png")
        plt.close()


def good_vs_bad_trajectories_plotted_to_arena(
    barrier_location, tracking_data, escapes_above_the_barrier, classes, escape_conditions, dir, session_name
):
    """Plots trajectories for good and bad escapes"""
    fig, (ax_pre_flip, ax_post_flip) = plt.subplots(1, 2, figsize=(20, 16))
    Arena(ax=ax_pre_flip, shelter_coordinates=tracking_data["shelter_loc"], condition="barrier_pre_flip", barrier_coordinates=barrier_location)
    Arena(ax=ax_post_flip, shelter_coordinates=tracking_data["shelter_loc"], condition="barrier_post_flip", barrier_coordinates=barrier_location)
    for i, escape in enumerate(escapes_above_the_barrier):
        if escape_conditions[i] == "barrier_pre_flip":
            if classes[i] == 0:
                ax_pre_flip.plot(escape["mouse_x_position"], escape["mouse_y_position"], color="r", label="Bad")
            if classes[i] == 1:
                ax_pre_flip.plot(escape["mouse_x_position"], escape["mouse_y_position"], color="g", label="Good")
        elif escape_conditions[i] == "barrier_post_flip":
            if classes[i] == 0:
                ax_post_flip.plot(escape["mouse_x_position"], escape["mouse_y_position"], color="r", label="Bad")
            if classes[i] == 1:
                ax_post_flip.plot(escape["mouse_x_position"], escape["mouse_y_position"], color="g", label="Good")

    handles, labels = plt.gca().get_legend_handles_labels()
    unique_handles_labels = dict(zip(labels, handles))
    ax_pre_flip.legend(unique_handles_labels.values(), unique_handles_labels.keys())
    ax_post_flip.legend(unique_handles_labels.values(), unique_handles_labels.keys())
    ax_pre_flip.set_title("Pre Flip Barrier")
    ax_post_flip.set_title("Post Flip Barrier")
    fig.suptitle(f"Good vs Bad Trajectories for {session_name}")

    # Save the figure
    save_path = dir / "good_vs_bad_trajectories"
    make_directory(save_path)
    plt.savefig(save_path / f"{session_name}.png")
    plt.close()


# ------------------- MAIN -------------------


def produce_data_for_escapes(experiments_objects, session_names, name_of_storage, create_design_matrix):
    """Produces and stores data for escape analysis

    Args:
        experiments_objects (list): List of experiment objects
        session_names (list): List of session names
        name_of_storage (str): Name of the storage file
        create_design_matrix (function): Function to create the design matrix
    """

    # Use a regular dictionary instead of defaultdict with lambda to avoid pickling issues
    storage = {}
    for experiment, session_name in zip(experiments_objects, session_names):
        print(f"Loading data for session: {session_name}")
        storage[session_name] = {}  # Initialize the session dictionary

        loaded_session = get_experiment(experiment)

        # Set paths
        base_path = loaded_session.base_path
        processed_path = loaded_session.processed_path
        session_path = os.path.join(base_path, processed_path)

        # Try both potential escape paths
        escape_paths = [
            os.path.join(session_path, "escapes", "escapes_obj.pkl"),
            os.path.join(session_path, "escape", "escapes_obj.pkl"),
            os.path.join(session_path, "escapes_obj.pkl"),
        ]

        escape_object_found = False
        escape_path = None
        for path in escape_paths:
            if os.path.exists(path):
                print(f"Found escape object at: {path}")
                escape_path = path
                escape_object_found = True
                break

        if not escape_object_found:
            print(f"Skipping session {session_name} - could not find escape object")
            continue

        if session_name == "JAL6_flip5_25mar":
            print(f"Skipping session {session_name} as specified")
            continue

        # Load data with error handling
        try:
            video_df_path = os.path.join(loaded_session.base_path, loaded_session.processed_path, "full_video_dataframe.csv")
            if not os.path.exists(video_df_path):
                print(f"Skipping session {session_name} - video dataframe not found at {video_df_path}")
                continue

            video_df = pl.read_csv(video_df_path)

            with open(escape_path, "rb") as ef:
                escapes_object = pickle.load(ef)

            # Verify escape object has required attributes
            required_attrs = ["escape_onset_frames", "escape_end_frames", "escape_condition", "spatial_efficiency"]
            if not all(hasattr(escapes_object, attr) for attr in required_attrs):
                print(f"Skipping session {session_name} - escape object missing required attributes")
                print(f"Available attributes: {dir(escapes_object)}")
                continue

            # Load neural data
            frame_by_cluster_path = os.path.join(loaded_session.base_path, loaded_session.processed_path, "frame_by_good_cluster_matrix.npy")
            if not os.path.exists(frame_by_cluster_path):
                print(f"Skipping session {session_name} - neural data not found at {frame_by_cluster_path}")
                continue

            frame_by_cluster_matrix = np.load(frame_by_cluster_path)

            # Load cluster IDs
            cluster_ids_path = os.path.join(session_path, "good_cluster_ids.npy")
            if not os.path.exists(cluster_ids_path):
                print(f"Warning: good_cluster_ids.npy not found for {session_name}, skipping verification")
            else:
                good_cluster_ids = np.load(cluster_ids_path)
                if len(good_cluster_ids) != frame_by_cluster_matrix.shape[1]:
                    print(f"Warning: Cluster IDs ({len(good_cluster_ids)}) do not match matrix columns ({frame_by_cluster_matrix.shape[1]})")

            # Load tracking data
            tracking_data = open_tracking_data(loaded_session)
            if not tracking_data or "barrier_loc" not in tracking_data:
                print(f"Skipping session {session_name} - tracking data missing or incomplete")
                continue

        except Exception as e:
            print(f"Error loading data for session {session_name}: {e}")
            continue

        try:
            # Extract logic data
            barrier_location = tracking_data["barrier_loc"]

            # Process escape data
            escape_info = extract_behavioural_data_between_escape_onset_offset(escapes_object, video_df)
            if not escape_info:
                print(f"No valid escape info for session {session_name}, skipping")
                continue

            escape_info = add_escape_id_to_escape_data(escape_info)

            # Get conditions and classification
            escape_conditions = list(escapes_object.escape_condition)
            escape_class = [1 if seff > SE_THRESHOLD else 0 for seff in escapes_object.spatial_efficiency]

            print(f"Session {session_name}: Found {len(escape_info)} escapes with {len(escape_conditions)} conditions")

            # Filter for barrier escapes
            barrier_escapes = []
            classes1 = []
            escape_info1 = []
            escape_conditions1 = []

            for i, condition in enumerate(escape_conditions):
                if isinstance(condition, str) and condition in ["barrier_pre_flip", "barrier_post_flip"]:
                    if i < len(escape_class) and i < len(escape_info):
                        barrier_escapes.append(i)
                        classes1.append(escape_class[i])
                        escape_info1.append(escape_info[i])
                        escape_conditions1.append(condition)

            print(f"Session {session_name}: Found {len(barrier_escapes)} barrier escapes")
            if not barrier_escapes:
                print(f"No barrier escapes for session {session_name}, skipping")
                continue

            # Filter for escapes above the barrier
            escapes_above_the_barrier = []
            classes2 = []
            escape_conditions2 = []

            for i, escape in enumerate(escape_info1):
                if len(escape["mouse_y_position"]) > 0:
                    y_pos = escape["mouse_y_position"][0]
                    if y_pos < barrier_location[0][1]:
                        escapes_above_the_barrier.append(escape)
                        classes2.append(classes1[i])
                        escape_conditions2.append(escape_conditions1[i])

            print(f"Session {session_name}: Found {len(escapes_above_the_barrier)} escapes above barrier")
            if not escapes_above_the_barrier:
                print(f"No escapes above barrier for session {session_name}, skipping")
                continue

            # Plot trajectories
            try:
                good_vs_bad_trajectories_plotted_to_arena(
                    barrier_location, tracking_data, escapes_above_the_barrier, classes2, escape_conditions2, dir, session_name
                )
            except Exception as e:
                print(f"Error plotting trajectories for session {session_name}: {e}")

            # Create design matrix
            try:
                design_matrix, classes_extended, escape_ids = create_design_matrix(escapes_above_the_barrier, frame_by_cluster_matrix, classes2)

                if len(design_matrix) == 0:
                    print(f"Empty design matrix for session {session_name}, skipping")
                    continue

                # Store the data - no defaultdict needed
                storage[session_name]["design_matrix"] = design_matrix
                storage[session_name]["classes_extended"] = classes_extended
                storage[session_name]["escape_ids"] = escape_ids

                print(f"Successfully processed session {session_name}")

            except Exception as e:
                print(f"Error creating design matrix for session {session_name}: {e}")
                continue

        except Exception as e:
            print(f"Error processing session {session_name}: {e}")
            continue

    # Save the data as pickle if we have data
    if storage:
        save_path = dir / f"{name_of_storage}.pkl"
        print(f"Saving data to {save_path}")
        try:
            with open(save_path, "wb") as f:
                pickle.dump(storage, f)
            print(f"Data saved successfully to {save_path}")
        except Exception as e:
            print(f"Error saving data: {e}")
    else:
        print("No data to save")

    return storage

if __name__ == "__main__":
    produce_data_for_escapes(experiments_objects, session_names, "escapes_first_20_frames", create_the_past_design_matrix)
