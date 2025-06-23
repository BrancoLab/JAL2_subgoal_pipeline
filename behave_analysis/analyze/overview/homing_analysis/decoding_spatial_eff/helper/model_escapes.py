import os
import pickle
from collections import defaultdict
from pathlib import Path

from loguru import logger
import pandas as pd
import scipy.stats as stats
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

from behave_analysis.utils.creating_directories import make_directory
from behave_analysis.analyze.overview.homing_analysis.decoding_spatial_eff.helper.data_gen import load_video_data


# ---------------------------- MAIN FUNCTION ------------------------------


def compute_accuracy_data_escapes(data_type, data_type_name, experiments_objects, random_labels=False):
    """Compute and analyze decoding accuracy data using logistic regression for a set of escape sessions.

    This function processes session data, performs logistic regression with group k-fold validation,
    and computes metrics such as accuracy, Kolmogorov-Smirnov statistics, and model coefficients.
    It also handles class imbalance and can optionally shuffle labels to evaluate performance on random labels.

    Args:
        data_type (dict): A dictionary where keys are session names and values are session-specific data,
                          including the design matrix and classes.
        data_type_name (str): A string representing the name of the data type, used for saving results.
        experiments_objects (list): A list of experiment objects corresponding to each session in `data_type`.
        random_labels (bool, optional): If True, shuffle class labels for randomization testing. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - whole_escape (dict): A dictionary mapping session names to their mean accuracy scores.
            - ks_decoding_comparison (defaultdict): Nested dictionaries containing Kolmogorov-Smirnov statistics
                                                   and accuracies for each fold of each session.
            - coefs (defaultdict): Nested dictionaries containing model coefficients for each fold of each session.
    """
    save_base = r"Z:\Jasmine_Laurence\single_trial_overview"
    save_path = os.path.join(save_base, "decoding_spatial_efficiency", data_type_name)
    dir = make_directory(save_path)
    whole_escape = {}
    ks_decoding_comparison = {}
    coefs = {}

    for session_name, data in data_type.items():
        if session_name not in [obj.__class__.__name__ for obj in experiments_objects]:
            logger.warning(f"Session {session_name} not found in experiments_objects, skipping")
            continue

        # Find the experiment object for this session
        experiment = None
        for obj in experiments_objects:
            if obj.__class__.__name__ == session_name:
                experiment = obj
                break

        if experiment is None:
            logger.warning(f"Could not find experiment object for session {session_name}, skipping")
            continue

        logger.info(f"Running logistic regression for session: {session_name}")

        try:
            video_df = load_video_data(experiment)
            if video_df is None:
                logger.warning(f"Failed to load video data for session {session_name}, skipping")
                continue

            design_matrix = data["design_matrix"]  # The X data
            classes_extended = np.asarray(data["classes_extended"])  # The y data

            # Check if escape_ids is present in the data
            group_data = data.get("escape_ids", data.get("homing_ids", None))
            if group_data is None:
                logger.warning(f"No group data (escape_ids) found for session {session_name}, skipping")
                continue

            if random_labels:
                np.random.shuffle(classes_extended)  # randomly shuffle the classes

            xBalanced, y_balanced, groups_balanced = handle_class_imbalance(classes_extended, session_name, design_matrix, {"escape_ids": group_data})

            if xBalanced is None:  # If the cutoff for the amount of data has not been met
                continue  # Skip this session

            # Initialize session data dictionaries if not already present
            if session_name not in ks_decoding_comparison:
                ks_decoding_comparison[session_name] = {}
            if session_name not in coefs:
                coefs[session_name] = {}

            group_kfold = GroupKFold(n_splits=2)  # k-fold iterator variation with non-overlapping groups
            fold_accuracies = []

            # Assign and remove the last column of the design matrix - Frame numbers are needed to access the behavior data
            frame_numbers = xBalanced[:, -1].astype(int)  # Access the last column of the design matrix
            xBalanced = xBalanced[:, :-1]  # Remove the last column of the design matrix

            for fold, (train_index, test_index) in enumerate(group_kfold.split(xBalanced, y_balanced, groups_balanced)):
                X_train, X_test = xBalanced[train_index], xBalanced[test_index]
                y_train, y_test = y_balanced[train_index], y_balanced[test_index]

                if not check_there_are_two_classes(y_train):
                    break

                down_count_ans = count_class_labels(y_train)
                logger.info(f"Class distribution for fold: {fold}. 0: {down_count_ans[0]}, 1: {down_count_ans[1]}")

                model = LogisticRegression(
                    class_weight="balanced",
                    random_state=1337,
                    max_iter=10000,
                ).fit(X_train, y_train)

                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                fold_accuracies.append(accuracy)

                # Plot behavioral data distributions for training set
                try:
                    x_ks, y_ks, s_ks, h_ks = plot_class_distributions_and_compute_kolmogorov_Smirnov(
                        frame_numbers, video_df, y_train, dir, session_name, fold, accuracy
                    )
                    ks_decoding_comparison[session_name][fold] = {"x": x_ks, "y": y_ks, "speed": s_ks, "hdir": h_ks, "accuracy": accuracy}
                except Exception as e:
                    logger.error(f"Error plotting class distributions for session {session_name}, fold {fold}: {e}")
                    ks_decoding_comparison[session_name][fold] = {"accuracy": accuracy}

                # Store model coefficients
                coefs[session_name][fold] = model.coef_[0]

            whole_escape[session_name] = np.mean(fold_accuracies)
            logger.info(f"Mean accuracy for session {session_name}: {whole_escape[session_name]}")

        except Exception as e:
            logger.error(f"Error processing session {session_name}: {e}")
            continue

    # Save all results
    save_data = {"accuracy_data": whole_escape, "ks_data": ks_decoding_comparison, "coefs": coefs}

    with open(dir / Path(f"{data_type_name}_accuracy_ks_coeffs_data.pkl"), "wb") as f:
        pickle.dump(save_data, f)

    return save_data


def compute_down_sampled_accuracy_data_escapes(data_type, experiments_objects, behaviour_to_subsample: str):
    """Compute accuracy data with behavioral subsampling for escape analysis.

    This version balances classes and then further subsamples based on behavioral variable distributions
    to ensure that any differences in accuracy aren't due to behavioral confounds.

    Args:
        data_type (dict): Dictionary of session data
        experiments_objects (list): List of experiment objects
        behaviour_to_subsample (str): Behavioral variable to subsample by (e.g., "mouse_x_position")

    Returns:
        dict: Dictionary of mean accuracies by session
    """
    accuracy_data = {}

    for session_name, data in data_type.items():
        if session_name not in [obj.__class__.__name__ for obj in experiments_objects]:
            logger.warning(f"Session {session_name} not found in experiments_objects, skipping")
            continue

        # Find the experiment object for this session
        experiment = None
        for obj in experiments_objects:
            if obj.__class__.__name__ == session_name:
                experiment = obj
                break

        if experiment is None:
            logger.warning(f"Could not find experiment object for session {session_name}, skipping")
            continue

        logger.info(f"Running subsampled logistic regression for session: {session_name}")

        try:
            design_matrix = data["design_matrix"]  # The X data
            classes_extended = np.asarray(data["classes_extended"])  # The y data

            # Check if escape_ids is present in the data
            group_data = data.get("escape_ids", data.get("homing_ids", None))
            if group_data is None:
                logger.warning(f"No group data (escape_ids) found for session {session_name}, skipping")
                continue

            xBalanced, y_balanced, groups_balanced = handle_class_imbalance(
                classes_extended, session_name, design_matrix, {"escape_ids": group_data}, cutoff=400  # 10s of data
            )

            video_df = load_video_data(experiment)
            if video_df is None or xBalanced is None:
                continue  # Skip this session

            group_kfold = GroupKFold(n_splits=2)
            fold_accuracies = []
            frame_numbers = xBalanced[:, -1].astype(int)
            xBalanced = xBalanced[:, :-1]  # Remove the last column of the design matrix

            for fold, (train_index, test_index) in enumerate(group_kfold.split(xBalanced, y_balanced, groups_balanced)):
                try:
                    # Find frames with good and bad labels in training set
                    good_indices = np.where(y_balanced[train_index] == 1)[0]
                    bad_indices = np.where(y_balanced[train_index] == 0)[0]
                    good_frames = frame_numbers[train_index][good_indices]
                    bad_frames = frame_numbers[train_index][bad_indices]

                    # Get behavioral values for the selected variable
                    if behaviour_to_subsample not in video_df.columns:
                        logger.warning(f"Behavior variable {behaviour_to_subsample} not found in video_df for {session_name}")
                        continue

                    good_behavior_values = video_df[behaviour_to_subsample][good_frames].values
                    bad_behavior_values = video_df[behaviour_to_subsample][bad_frames].values

                    # Create bins and find overlap between distributions
                    min_val = min(np.min(good_behavior_values), np.min(bad_behavior_values))
                    max_val = max(np.max(good_behavior_values), np.max(bad_behavior_values))
                    bins = np.linspace(min_val, max_val, 25)

                    good_hist, _ = np.histogram(good_behavior_values, bins=bins)
                    bad_hist, _ = np.histogram(bad_behavior_values, bins=bins)
                    overlap = np.minimum(good_hist, bad_hist)

                    # Find indices of frames in overlapping regions of the distributions
                    overlap_frames = []
                    for i in range(len(bins) - 1):
                        if overlap[i] > 0:
                            # Identify frames in this bin for both classes
                            bin_mask_good = (good_behavior_values >= bins[i]) & (good_behavior_values < bins[i + 1])
                            bin_mask_bad = (bad_behavior_values >= bins[i]) & (bad_behavior_values < bins[i + 1])

                            # Get frames in this bin
                            good_bin_frames = good_frames[bin_mask_good]
                            bad_bin_frames = bad_frames[bin_mask_bad]

                            # Add to our collection
                            overlap_frames.extend(good_bin_frames.tolist())
                            overlap_frames.extend(bad_bin_frames.tolist())

                    # Get indices in the original data matrix for the overlapping frames
                    overlap_indices = np.where(np.isin(frame_numbers[train_index], overlap_frames))[0]

                    # Skip if no overlap was found
                    if len(overlap_indices) == 0:
                        logger.warning(f"No behavioral overlap found for {session_name}, fold {fold}")
                        continue

                    # Extract balanced data
                    X_train_balanced = xBalanced[train_index][overlap_indices]
                    y_train_balanced = y_balanced[train_index][overlap_indices]
                    X_test = xBalanced[test_index]
                    y_test = y_balanced[test_index]

                    if not check_there_are_two_classes(y_train_balanced):
                        logger.warning(f"Not enough class diversity in fold {fold} for session {session_name}")
                        continue

                    # Check class balance after subsampling
                    down_count_ans = count_class_labels(y_train_balanced)
                    logger.info(f"Subsampled class distribution - fold {fold}: 0: {down_count_ans[0]}, 1: {down_count_ans[1]}")

                    # Train model and compute accuracy
                    model = LogisticRegression(
                        class_weight="balanced",
                        random_state=1337,
                        max_iter=10000,
                    ).fit(X_train_balanced, y_train_balanced)

                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    fold_accuracies.append(accuracy)

                except Exception as e:
                    logger.error(f"Error in fold {fold} for session {session_name}: {e}")
                    continue

            if fold_accuracies:
                accuracy_data[session_name] = np.mean(fold_accuracies)
                logger.info(f"Mean accuracy for session {session_name}: {accuracy_data[session_name]}")
            else:
                logger.warning(f"No valid folds for session {session_name}")

        except Exception as e:
            logger.error(f"Error processing session {session_name}: {e}")
            continue

    return accuracy_data


# - Helper functions for the model -


def count_class_labels(classes):
    """Count the number of instances for each class label."""
    dic_to_store = {}
    unique_classes = np.unique(classes)
    for cls in unique_classes:
        dic_to_store[cls] = np.sum(classes == cls)
    return dic_to_store


def downsample_larger_class(design_matrix, classes_extended, random_state, escape_ids):
    """Randomly downsample the larger class to match the smaller class to ensure balanced training.

    Args:
        design_matrix (np.ndarray): The feature matrix
        classes_extended (np.ndarray): The class labels
        random_state (int): Random seed for reproducibility
        escape_ids: Dict or list of escape identifiers for group-based cross-validation

    Returns:
        tuple: Balanced design matrix, balanced classes, and balanced group identifiers
    """
    X = design_matrix
    y = classes_extended
    class_0_indices = np.where(y == 0)[0]
    class_1_indices = np.where(y == 1)[0]

    # Ensure only downsampling of the larger class
    if len(class_0_indices) > len(class_1_indices):
        class_0_indices = resample(class_0_indices, replace=False, n_samples=len(class_1_indices), random_state=random_state)
    elif len(class_1_indices) > len(class_0_indices):
        class_1_indices = resample(class_1_indices, replace=False, n_samples=len(class_0_indices), random_state=random_state)

    # Combine indices from both classes
    balanced_indices = np.concatenate([class_0_indices, class_1_indices])

    # Extract balanced data
    xBalanced = X[balanced_indices]
    y_balanced = y[balanced_indices]

    # Handle group ids for cross-validation
    if isinstance(escape_ids, dict):
        groups_balanced = [escape_ids[i] for i in balanced_indices if i in escape_ids]
    else:
        # Assume escape_ids is a list or array that can be indexed directly
        groups_balanced = [escape_ids[i] for i in balanced_indices]

    return xBalanced, y_balanced, groups_balanced


def check_there_is_enough_data(count_ans, session_name, cutoff):
    """Returns True if there is enough data for both classes.

    Args:
        count_ans (dict): Dictionary with class counts
        session_name (str): Name of the session for logging
        cutoff (int): The minimum number of frames required for each class. 40 frames is 1 second of data
    """
    for cls in [0, 1]:
        if cls not in count_ans or count_ans[cls] < cutoff:
            logger.warning(f"For session {session_name}, class {cls} has less than {cutoff / 40} seconds of data, skipping session")
            return False
    return True


def check_there_are_two_classes(y_train):
    """Check if the training data contains examples from both classes."""
    if len(np.unique(y_train)) < 2:
        logger.warning("Skipping fold because there is only one class in the training data")
        return False
    return True


def handle_class_imbalance(classes_extended, session_name, design_matrix, data, cutoff=80):
    """Balance classes in the dataset by downsampling the larger class.

    Args:
        classes_extended (np.ndarray): Class labels
        session_name (str): Session name for logging
        design_matrix (np.ndarray): Feature matrix
        data (dict): Dictionary containing escape IDs or group identifiers
        cutoff (int): Minimum number of frames required per class (default 80 = 2 seconds at 40fps)

    Returns:
        tuple: Balanced design matrix, balanced classes, and balanced group identifiers,
               or (None, None, None) if there's not enough data
    """
    # Get group identifiers for cross-validation (escape_ids)
    group_data = data.get("escape_ids", data.get("homing_ids", None))
    if group_data is None:
        logger.warning(f"No group data found for session {session_name}")
        return None, None, None

    # Check the class distribution before down sampling
    count_ans = count_class_labels(classes_extended)
    logger.info(f"Class distribution before balancing - 0: {count_ans.get(0, 0)}, 1: {count_ans.get(1, 0)}")

    if not check_there_is_enough_data(count_ans, session_name, cutoff=cutoff):
        return None, None, None  # The cutoff has not been met

    # Down sample the larger class
    xBalanced, y_balanced, groups_balanced = downsample_larger_class(
        design_matrix=design_matrix, classes_extended=classes_extended, random_state=1337, escape_ids=group_data
    )

    count_of_downsampled_classes = count_class_labels(y_balanced)
    logger.info(f"Class distribution after balancing - 0: {count_of_downsampled_classes.get(0, 0)}, " f"1: {count_of_downsampled_classes.get(1, 0)}")

    return xBalanced, y_balanced, groups_balanced


def are_the_distributions_different(good_data, bad_data):
    """Conduct a Kolmogorov-Smirnov test to see if the distributions are different.

    Args:
        good_data: Data points from the 'good' class (high spatial efficiency)
        bad_data: Data points from the 'bad' class (low spatial efficiency)

    Returns:
        tuple: (is_different (bool), ks_statistic (float), p_value (float))
    """
    # Convert inputs to numpy arrays for safety
    good_data = np.array(good_data)
    bad_data = np.array(bad_data)

    # Handle empty data
    if len(good_data) == 0 or len(bad_data) == 0:
        return False, 0.0, 1.0

    # Remove NaN values
    good_data = good_data[~np.isnan(good_data)]
    bad_data = bad_data[~np.isnan(bad_data)]

    # Ensure we have enough data after filtering NaNs
    if len(good_data) < 2 or len(bad_data) < 2:
        return False, 0.0, 1.0

    # Perform KS test
    try:
        ks_statistic, p_value = stats.ks_2samp(good_data, bad_data)
        return p_value < 0.05, ks_statistic, p_value
    except Exception as e:
        logger.error(f"Error in KS test: {e}")
        return False, 0.0, 1.0


# --- Plotting functions for the model ---


def plot_class_distributions_and_compute_kolmogorov_Smirnov(frame_numbers, video_df, y_train, dir, session_name, fold, accuracy):
    """Plot behavioral distributions by class and compute KS statistics to check for differences.

    Args:
        frame_numbers: Frame indices for video data lookup
        video_df: DataFrame containing behavioral data
        y_train: Training set class labels
        dir: Directory to save plots
        session_name: Name of the session
        fold: Current fold number for cross-validation
        accuracy: Model accuracy for annotation

    Returns:
        tuple: KS statistics for x position, y position, speed, and head direction
    """
    # Create directory if it doesn't exist
    os.makedirs(dir, exist_ok=True)

    # Get the behavioral distributions of the classes for the training dataset
    good_indices = np.where(y_train == 1)[0]
    bad_indices = np.where(y_train == 0)[0]

    # Exit if either class is empty
    if len(good_indices) == 0 or len(bad_indices) == 0:
        logger.warning(f"Empty class in data for session {session_name}, fold {fold}")
        return 0.0, 0.0, 0.0, 0.0

    # Get frame numbers for both classes
    good_frames = frame_numbers[good_indices]
    bad_frames = frame_numbers[bad_indices]

    # Initialize arrays for behavioral variables
    good_x, bad_x = [], []
    good_y, bad_y = [], []
    good_speed, bad_speed = [], []
    good_hdir, bad_hdir = [], []

    # Extract behavioral data with error handling
    try:
        if "mouse_x_position" in video_df.columns:
            good_x = video_df["mouse_x_position"][good_frames].values
            bad_x = video_df["mouse_x_position"][bad_frames].values

        if "mouse_y_position" in video_df.columns:
            good_y = video_df["mouse_y_position"][good_frames].values
            bad_y = video_df["mouse_y_position"][bad_frames].values

        if "speed" in video_df.columns:
            good_speed = video_df["speed"][good_frames].values
            bad_speed = video_df["speed"][bad_frames].values

        if "hdir" in video_df.columns:
            good_hdir = video_df["hdir"][good_frames].values
            bad_hdir = video_df["hdir"][bad_frames].values
    except Exception as e:
        logger.error(f"Error extracting behavioral data: {e}")
        return 0.0, 0.0, 0.0, 0.0

    # Compute the Kolmogorov-Smirnov statistic
    x_stat, x_ks, _ = are_the_distributions_different(good_x, bad_x)
    y_stat, y_ks, _ = are_the_distributions_different(good_y, bad_y)
    speed_stat, s_ks, _ = are_the_distributions_different(good_speed, bad_speed)
    hdir_stat, h_ks, _ = are_the_distributions_different(good_hdir, bad_hdir)

    # Plot the distributions
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        # X position plot
        if len(good_x) > 0 and len(bad_x) > 0:
            axes[0].hist(good_x, bins=30, alpha=0.5, label="Good")
            axes[0].hist(bad_x, bins=30, alpha=0.5, label="Bad")
            axes[0].set_title(f"X Position Distribution - Diff: {x_stat}")
            axes[0].set_ylabel("Frequency")
            axes[0].legend()

        # Y position plot
        if len(good_y) > 0 and len(bad_y) > 0:
            axes[1].hist(good_y, bins=30, alpha=0.5, label="Good")
            axes[1].hist(bad_y, bins=30, alpha=0.5, label="Bad")
            axes[1].set_title(f"Y Position Distribution - Diff: {y_stat}")
            axes[1].set_ylabel("Frequency")
            axes[1].legend()

        # Speed plot
        if len(good_speed) > 0 and len(bad_speed) > 0:
            axes[2].hist(good_speed, bins=30, alpha=0.5, label="Good")
            axes[2].hist(bad_speed, bins=30, alpha=0.5, label="Bad")
            axes[2].set_title(f"Speed Distribution - Diff: {speed_stat}")
            axes[2].set_ylabel("Frequency")
            axes[2].legend()

        # Head direction plot
        if len(good_hdir) > 0 and len(bad_hdir) > 0:
            axes[3].hist(good_hdir, bins=30, alpha=0.5, label="Good")
            axes[3].hist(bad_hdir, bins=30, alpha=0.5, label="Bad")
            axes[3].set_title(f"Hdir Distribution - Diff: {hdir_stat}")
            axes[3].set_ylabel("Frequency")
            axes[3].legend()

        plt.suptitle(f"Behavioral Class Distributions - {session_name}, Fold {fold}, Acc: {accuracy:.2f}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(dir, f"class_distributions_behaviour_{session_name}_fold_{fold}.png"))
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting distributions: {e}")

    return x_ks, y_ks, s_ks, h_ks


def plot_accuracy_across_sessions(data_type, mice_groups, plot_title="Escape Decoding Accuracy"):
    """Plot accuracy results across sessions and mice to visualize trends.

    Args:
        data_type (dict): Dictionary mapping session names to accuracy values
        mice_groups (dict): Dictionary mapping mouse IDs to lists of session names in order
        plot_title (str): Title for the plot

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Process data into a DataFrame
    data = list(data_type.items())
    df = pd.DataFrame(data, columns=["session", "accuracy"])

    # Extract mouse ID from session name
    df["mouse"] = df["session"].str.extract(r"(?P<mouse>JAL\d+)")

    # Sort sessions based on their predefined order in mice_groups
    def sort_sessions(mouse, session):
        if mouse in mice_groups and session in mice_groups[mouse]:
            return mice_groups[mouse].index(session)
        return float("nan")

    df["session_order"] = df.apply(lambda x: sort_sessions(x["mouse"], x["session"]), axis=1)

    # Remove sessions not in the specified order
    sorted_df = df.dropna(subset=["session_order"]).sort_values(by=["mouse", "session_order"])

    # Assign session numbers within each mouse group
    sorted_df["session_num"] = sorted_df.groupby("mouse").cumcount() + 1

    # Calculate average accuracy by session number
    df_avg = sorted_df.groupby("session_num")["accuracy"].agg(["mean", "std"]).reset_index()

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot average line across all mice
    ax.errorbar(
        df_avg["session_num"], df_avg["mean"], yerr=df_avg["std"], marker="o", linestyle="-", color="black", ecolor="gray", capsize=5, label="Average"
    )

    # Plot individual mice lines
    for mouse in sorted_df["mouse"].unique():
        mouse_data = sorted_df[sorted_df["mouse"] == mouse]
        ax.plot(mouse_data["session_num"], mouse_data["accuracy"], marker="o", linestyle="--", alpha=0.5, label=mouse)

    # Add 50% chance level line
    ax.axhline(y=0.5, color="r", linestyle=":", label="Chance")

    # Set axis properties
    ax.set_xticks(range(1, sorted_df["session_num"].max() + 1))
    ax.set_xticklabels(range(1, sorted_df["session_num"].max() + 1))
    ax.set_title(f"Decoding Accuracy Across Mice and Sessions - {plot_title}")
    ax.set_xlabel("Session Number")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.4, 1.0)  # Set y-axis from 0.4 to 1.0 for better visualization
    ax.legend(title="Mouse ID")

    plt.tight_layout()
    return fig


# --- Main usage functions ---


def run_escape_analysis(escape_data_path, experiments_objects, mice_groups, random_comparison=True):
    """Run the full escape analysis pipeline.

    Args:
        escape_data_path (str): Path to the pickle file containing escape data
        experiments_objects (list): List of experiment objects for the sessions
        mice_groups (dict): Dictionary mapping mouse IDs to ordered lists of session names
        random_comparison (bool): Whether to run a random label comparison

    Returns:
        dict: Dictionary containing all analysis results
    """
    # Load the escape data
    try:
        with open(escape_data_path, "rb") as f:
            escape_data = pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading escape data: {e}")
        return None

    # Get the data type name from the file path
    data_type_name = Path(escape_data_path).stem

    # Print basic info about the data
    logger.info(f"Loaded {len(escape_data)} sessions from {escape_data_path}")
    if escape_data:
        first_session = next(iter(escape_data))
        logger.info(f"Sample session {first_session} has keys: {escape_data[first_session].keys()}")

    # Run the main accuracy computation
    logger.info(f"Computing accuracy for {data_type_name}...")
    results = compute_accuracy_data_escapes(escape_data, data_type_name, experiments_objects)

    # Run random label comparison if requested
    if random_comparison:
        logger.info(f"Computing accuracy with random labels for {data_type_name}...")
        random_results = compute_accuracy_data_escapes(escape_data, f"{data_type_name}_random", experiments_objects, random_labels=True)

        # Compare real vs. random accuracy
        real_accs = list(results["accuracy_data"].values())
        random_accs = list(random_results["accuracy_data"].values())

        if real_accs and random_accs:
            logger.info(f"Average real accuracy: {np.mean(real_accs):.3f} ± {np.std(real_accs):.3f}")
            logger.info(f"Average random accuracy: {np.mean(random_accs):.3f} ± {np.std(random_accs):.3f}")

            # Statistical test if we have enough data
            if len(real_accs) >= 5 and len(random_accs) >= 5:
                t_stat, p_val = stats.ttest_ind(real_accs, random_accs)
                logger.info(f"T-test: t={t_stat:.3f}, p={p_val:.3f}")

        # Plot real vs. random comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot([real_accs, random_accs], labels=["Real Labels", "Random Labels"])
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Escape Decoding Accuracy: Real vs. Random ({data_type_name})")
        ax.axhline(y=0.5, color="r", linestyle=":", label="Chance")
        fig.savefig(f"escape_accuracy_comparison_{data_type_name}.png")
        plt.close(fig)

    # Plot accuracy across sessions
    if results["accuracy_data"]:
        try:
            fig = plot_accuracy_across_sessions(results["accuracy_data"], mice_groups, plot_title=f"Escape Decoding - {data_type_name}")
            fig.savefig(f"escape_accuracy_across_sessions_{data_type_name}.png")
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error plotting accuracy across sessions: {e}")

    return results


def analyze_behavioral_contribution(escape_data_path, experiments_objects, behaviors=None):
    """Analyze how specific behavioral variables contribute to decoding performance.

    This function compares normal decoding accuracy to accuracy achieved when
    subsampling to control for specific behavioral variables.

    Args:
        escape_data_path (str): Path to the pickle file containing escape data
        experiments_objects (list): List of experiment objects for the sessions
        behaviors (list): List of behavioral variables to analyze. If None, defaults
                         to ["mouse_x_position", "mouse_y_position", "speed", "hdir"]

    Returns:
        dict: Dictionary mapping behavior variables to accuracy results
    """
    # Set default behaviors if none provided
    if behaviors is None:
        behaviors = ["mouse_x_position", "mouse_y_position", "speed", "hdir"]

    # Load the escape data
    try:
        with open(escape_data_path, "rb") as f:
            escape_data = pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading escape data: {e}")
        return None

    # Get normal accuracy as baseline
    data_type_name = Path(escape_data_path).stem
    baseline_results = compute_accuracy_data_escapes(escape_data, f"{data_type_name}_baseline", experiments_objects)

    # Store all results
    all_results = {"baseline": baseline_results["accuracy_data"]}

    # Analyze each behavior variable
    for behavior in behaviors:
        logger.info(f"Analyzing contribution of {behavior}...")
        behavior_results = compute_down_sampled_accuracy_data_escapes(escape_data, experiments_objects, behavior)
        all_results[behavior] = behavior_results

        # Compare with baseline if we have results
        if behavior_results and baseline_results["accuracy_data"]:
            # Get sessions that appear in both results
            common_sessions = set(behavior_results.keys()) & set(baseline_results["accuracy_data"].keys())

            if common_sessions:
                baseline_accs = [baseline_results["accuracy_data"][s] for s in common_sessions]
                behavior_accs = [behavior_results[s] for s in common_sessions]

                # Calculate differences
                mean_baseline = np.mean(baseline_accs)
                mean_behavior = np.mean(behavior_accs)

                logger.info(f"Baseline accuracy: {mean_baseline:.3f}")
                logger.info(f"{behavior} controlled accuracy: {mean_behavior:.3f}")
                logger.info(f"Difference: {mean_baseline - mean_behavior:.3f}")

                # Statistical test if enough data
                if len(common_sessions) >= 5:
                    t_stat, p_val = stats.ttest_rel(baseline_accs, behavior_accs)
                    logger.info(f"Paired t-test: t={t_stat:.3f}, p={p_val:.3f}")

    # Plot comparison of all behaviors
    plot_behavioral_contributions(all_results, data_type_name)

    return all_results


def plot_behavioral_contributions(all_results, data_type_name):
    """Plot comparison of decoding accuracy with different behavioral controls.

    Args:
        all_results (dict): Dictionary mapping behavior names to accuracy results
        data_type_name (str): Name of the data type for the plot title
    """
    # Get common sessions across all comparisons
    all_sessions = set()
    for behavior, results in all_results.items():
        all_sessions.update(results.keys())

    common_sessions = all_sessions.copy()
    for behavior, results in all_results.items():
        common_sessions &= set(results.keys())

    if not common_sessions:
        logger.warning("No common sessions across all behavioral controls")
        return

    # Extract accuracies for each behavior control
    data_for_plot = []
    behavior_names = []

    for behavior, results in all_results.items():
        behavior_names.append(behavior)
        data_for_plot.append([results[s] for s in common_sessions])

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data_for_plot, labels=behavior_names)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Effect of Controlling for Behavioral Variables ({data_type_name})")
    ax.axhline(y=0.5, color="r", linestyle=":", label="Chance")

    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(f"behavioral_contribution_{data_type_name}.png")
    plt.close(fig)
