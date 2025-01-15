import os
import pickle
from collections import defaultdict
from pathlib import Path

import pandas as pd
import scipy.stats as stats
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

from behave_analysis.utils.creating_directories import make_directory
from behave_analysis.analyze.overview.homing_analysis.decoding_spatial_eff.data_gen import load_video_data


# - Helper functions for the model -


def count_class_labels(classes):
    dic_to_store = defaultdict(int)
    for idx, val in enumerate(classes):
        dic_to_store[val] += 1
    return dic_to_store


def downsample_larger_class(design_matrix, classes_extended, random_state, homing_ids):
    "Randomly downsamples the larger class to match the smaller class"
    X = design_matrix
    y = classes_extended
    class_0_indices = np.where(y == 0)[0]
    class_1_indices = np.where(y == 1)[0]

    # Ensure only downsampling of the larger class
    if len(class_0_indices) > len(class_1_indices):
        class_0_indices = resample(class_0_indices, replace=False, n_samples=len(class_1_indices), random_state=random_state)
    elif len(class_1_indices) > len(class_0_indices):
        class_1_indices = resample(class_1_indices, replace=False, n_samples=len(class_0_indices), random_state=random_state)

    balanced_indices = np.concatenate([class_0_indices, class_1_indices])
    X_balanced = X[balanced_indices]
    y_balanced = y[balanced_indices]
    print("The length of balanced indcies", len(balanced_indices))
    print("The length of homing_ids", len(homing_ids))
    # groups_balanced = homing_ids[balanced_indices]
    groups_balanced = [homing_ids[i] for i in balanced_indices]

    return X_balanced, y_balanced, groups_balanced


def check_there_is_enough_data(count_ans, session_name, cutoff):
    """Returns True if there is enough data

    Args:
        cutoff (int): The minimum number of frames required for each class. 40 frames is 1 second of data"""
    if count_ans[0] < cutoff or count_ans[1] < cutoff:
        print(f"For session {session_name}, there is less than {cutoff / 40} seconds of data for one class, skipping session")
        return False
    return True


def check_there_are_two_classes(y_train):
    if len(np.unique(y_train)) == 1:
        print("Skipping fold because there is only one class")
        return False
    return True


def handle_class_imbalance(classes_extended, session_name, design_matrix, data):
    # Check the class distribution before down sampling
    count_ans = count_class_labels(classes_extended)  # How many frames have 0 or 1
    print(f"Class distribution before down sampling. 0: {count_ans[0]}, 1: {count_ans[1]}")
    if not check_there_is_enough_data(
        count_ans, session_name, cutoff=80
    ):  # If there are less than x seconds of data in either class, skip this session
        return None, None, None  # The cuttoff has not been met

    # Down sample the larger class
    X_balanced, y_balanced, groups_balanced = downsample_larger_class(
        design_matrix=design_matrix, classes_extended=classes_extended, random_state=1337, homing_ids=data["homing_ids"]
    )
    count_of_downsampled_classes = count_class_labels(y_balanced)
    print(
        f"Class distribution for the entire session after down sampling. 0: {count_of_downsampled_classes[0]}, 1: {count_of_downsampled_classes[1]}"
    )

    return X_balanced, y_balanced, groups_balanced


def are_the_distributions_different(good_data, bad_data):
    """Conduct a ks_statistic test to see if the distributions are different. If p < 0.05, then the distributions are different"""
    ks_statistic, p_value = stats.ks_2samp(good_data, bad_data)
    if p_value < 0.05:
        return True, ks_statistic, p_value
    return False, ks_statistic, p_value


# --- Plotting functions for the model ---


def plot_class_distributions_and_compute_kolmogorov_Smirnov(frame_numbers, video_df, y_train, dir, session_name, fold, accuracy):

    # Get the behavioural distributions of the classes for the training dataset to plot histograms to check they overlap
    good_indicies = np.where(y_train == 1)[0]
    bad_indicies = np.where(y_train == 0)[0]
    good_frames = frame_numbers[good_indicies]
    bad_frames = frame_numbers[bad_indicies]
    good_x_positions = video_df["mouse_x_position"][good_frames]
    bad_x_positions = video_df["mouse_x_position"][bad_frames]
    good_y_positions = video_df["mouse_y_position"][good_frames]
    bad_y_positions = video_df["mouse_y_position"][bad_frames]
    good_speed = video_df["speed"][good_frames]
    bad_speed = video_df["speed"][bad_frames]
    good_hdir = video_df["hdir"][good_frames]
    bad_hdir = video_df["hdir"][bad_frames]

    # Compute the Kolmogorov-Smirnov statistic to see if the distributions are different
    x_stat, x_ks, _ = are_the_distributions_different(good_x_positions, bad_x_positions)
    y_stat, y_ks, _ = are_the_distributions_different(good_y_positions, bad_y_positions)
    speed_stat, s_ks, _ = are_the_distributions_different(good_speed, bad_speed)
    hdir_stat, h_ks, _ = are_the_distributions_different(good_hdir, bad_hdir)

    # Plot the train class distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    axes[0].hist(good_x_positions, bins=30, alpha=0.5, label="Good")
    axes[0].hist(bad_x_positions, bins=30, alpha=0.5, label="Bad")
    axes[0].set_title(f"X Position Distribution - Distributions Different: {x_stat}")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    axes[1].hist(good_y_positions, bins=30, alpha=0.5, label="Good")
    axes[1].hist(bad_y_positions, bins=30, alpha=0.5, label="Bad")
    axes[1].set_title(f"Y Position Distribution - Distributions Different: {y_stat}")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    axes[2].hist(good_speed, bins=30, alpha=0.5, label="Good")
    axes[2].hist(bad_speed, bins=30, alpha=0.5, label="Bad")
    axes[2].set_title(f"Speed Distribution - Are Distributions Different: {speed_stat}")
    axes[2].set_ylabel("Frequency")
    axes[2].legend()

    axes[3].hist(good_hdir, bins=30, alpha=0.5, label="Good")
    axes[3].hist(bad_hdir, bins=30, alpha=0.5, label="Bad")
    axes[3].set_title(f"Hdir Distribution - Are Distributions Different: {hdir_stat}")
    axes[3].set_ylabel("Frequency")
    axes[3].legend()
    plt.suptitle(f"Behavioural Data class break down training distributions for Session {session_name}, Fold {fold}, Accuracy: {accuracy}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(dir, f"class_distributions_behaviour_{session_name}_fold_{fold}.png"))
    plt.close()

    return x_ks, y_ks, s_ks, h_ks


def plot_the_train_test_behavioural_split(video_df, frame_numbers, train_index, test_index, save_dir, session_name, fold, accuracy):

    # Extract the frame numbers for the training and testing dataset
    train_indices = frame_numbers[train_index]
    test_indices = frame_numbers[test_index]

    # Extract the behaviour data for the training and testing dataset so we can plot historgrams of the train and test data
    # to make sure they overlap within the folds
    x_position_train, x_position_test = video_df["mouse_x_position"][train_indices], video_df["mouse_x_position"][test_indices]
    y_position_train, y_position_test = video_df["mouse_y_position"][train_indices], video_df["mouse_y_position"][test_indices]
    speed_train, speed_test = video_df["speed"][train_indices], video_df["speed"][test_indices]
    hdir_train, hdir_test = video_df["hdir"][train_indices], video_df["hdir"][test_indices]

    # Plot the test-train distribution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    axes[0].hist(x_position_train, bins=30, alpha=0.5, label="Train")
    axes[0].hist(x_position_test, bins=30, alpha=0.5, label="Test")
    x_stat, _, _ = are_the_distributions_different(x_position_train, x_position_test)
    axes[0].set_title(f"X Position Distribution - Are Distributions Different: {x_stat}")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    axes[1].hist(y_position_train, bins=30, alpha=0.5, label="Train")
    axes[1].hist(y_position_test, bins=30, alpha=0.5, label="Test")
    y_stat, _, _ = are_the_distributions_different(y_position_train, y_position_test)
    axes[1].set_title(f"Y Position Distribution - Are Distributions Different: {y_stat}")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    axes[2].hist(speed_train, bins=30, alpha=0.5, label="Train")
    axes[2].hist(speed_test, bins=30, alpha=0.5, label="Test")
    speed_stat, _, _ = are_the_distributions_different(speed_train, speed_test)
    axes[2].set_title(f"Speed Distribution - Are Distributions Different: {speed_stat}")
    axes[2].set_ylabel("Frequency")
    axes[2].legend()

    axes[3].hist(hdir_train, bins=30, alpha=0.5, label="Train")
    axes[3].hist(hdir_test, bins=30, alpha=0.5, label="Test")
    hdir_stat, _, _ = are_the_distributions_different(hdir_train, hdir_test)
    axes[3].set_title(f"Hdir Distribution - Are Distributions Different: {hdir_stat}")
    axes[3].set_ylabel("Frequency")
    axes[3].legend()

    plt.suptitle(f"Behavioural Data Distribution for Session {session_name}, Fold {fold}, Accuracy: {accuracy}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_dir, f"test_train_distributions_behaviour{session_name}_fold_{fold}.png"))
    plt.close()


def plot_accuracy_across_sessions(data_type, mice_groups, plot_title: str):

    # Process data into a DataFrame for easier manipulation
    data = list(data_type.items())
    df = pd.DataFrame(data, columns=["session", "accuracy"])

    # Extract mouse ID
    df["mouse"] = df["session"].str.extract(r"(?P<mouse>JAL\d+)")

    # Sort sessions for each mouse based on their predefined order in mice_groups
    def sort_sessions(mouse, session):
        if mouse in mice_groups:
            return mice_groups[mouse].index(session)
        return np.nan

    df["session_order"] = df.apply(lambda x: sort_sessions(x["mouse"], x["session"]), axis=1)

    # Drop any sessions that don't align with the mouse groups
    sorted_df = df.dropna(subset=["session_order"]).sort_values(by=["mouse", "session_order"])

    # Assign session numbers within each mouse group
    sorted_df["session_num"] = sorted_df.groupby("mouse").cumcount() + 1

    # Average accuracy across all sessions
    df_avg = sorted_df.groupby("session_num", as_index=False)["accuracy"].mean()

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot average line across all mice
    ax.plot(df_avg["session_num"], df_avg["accuracy"], marker="o", linestyle="-", color="black", label="Average Across Mice")

    # Plot individual mice
    for mouse in sorted_df["mouse"].unique():
        mouse_data = sorted_df[sorted_df["mouse"] == mouse]
        ax.plot(mouse_data["session_num"], mouse_data["accuracy"], marker="o", linestyle="--", alpha=0.5, label=mouse)

    ax.set_xticks(range(1, sorted_df["session_num"].max() + 1))
    ax.set_xticklabels(range(1, sorted_df["session_num"].max() + 1))
    ax.set_title(f"Decoding Accuracy Across Mice and Sessions - {plot_title}")
    ax.set_xlabel("Session Number")
    ax.set_ylabel("Accuracy")
    ax.legend(title="Mouse ID")
    plt.tight_layout()
    plt.show()


# - Main function for the model -


def compute_accuracy_data(data_type, data_type_name, experiments_objects, random_labels=False):

    # Save files
    save_base = r"Z:\Jasmine_Laurence\single_trial_overview"
    save_path = os.path.join(save_base, "decoding_spatial_efficiency", data_type_name)
    dir = make_directory(save_path)

    # Create a dictionary to store the accuracy for each session
    whole_homing = defaultdict(list)
    ks_decoding_comparison = defaultdict(defaultdict)
    coefs = defaultdict(defaultdict)

    # Loop through each session and run logistic regression
    for (session_name, data), experiment in zip(data_type.items(), experiments_objects):
        print(f"Running logistic regression for session: {session_name}")

        # Load required data
        video_df = load_video_data(experiment)
        design_matrix = data["design_matrix"]  # The X data
        classes_extended = np.asarray(data["classes_extended"])  # The y data
        if random_labels:
            np.random.shuffle(classes_extended)  # randomly shuffle the classes

        X_balanced, y_balanced, groups_balanced = handle_class_imbalance(classes_extended, session_name, design_matrix, data)
        if X_balanced is None:  # If the cuttoff for the amount of data has not been met
            continue  # Skip this session

        group_kfold = GroupKFold(n_splits=2)  # k-fold interator variation with non-overlapping groups
        fold_accuracies = []

        # Assign and remove the last column of the design matrix - Frame numbers are needed to access the behaviour data
        frame_numbers = X_balanced[:, -1].astype(int)  # Access the last column of the design matrix
        X_balanced = X_balanced[:, :-1]  # Remove the last column of the design matrix

        for fold, (train_index, test_index) in enumerate(group_kfold.split(X_balanced, y_balanced, groups_balanced)):
            X_train, X_test = X_balanced[train_index], X_balanced[test_index]
            y_train, y_test = y_balanced[train_index], y_balanced[test_index]
            if not check_there_are_two_classes(y_train):
                break
            down_count_ans = count_class_labels(y_train)
            print(f"Class distribution for fold: {fold}. 0: {down_count_ans[0]}, 1: {down_count_ans[1]}")
            model = LogisticRegression(
                penalty="l2",
                dual=False,
                tol=0.0001,
                C=1.0,
                fit_intercept=True,
                intercept_scaling=1,
                class_weight="balanced",
                random_state=1337,
                solver="lbfgs",
                max_iter=10000,
                verbose=0,
                warm_start=False,
                n_jobs=None,
                l1_ratio=None,
            ).fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            fold_accuracies.append(accuracy)
            plot_the_train_test_behavioural_split(video_df, frame_numbers, train_index, test_index, dir, session_name, fold, accuracy)
            x_ks, y_ks, s_ks, h_ks = plot_class_distributions_and_compute_kolmogorov_Smirnov(
                frame_numbers, video_df, y_train, dir, session_name, fold, accuracy
            )
            ks_decoding_comparison[session_name][fold] = {"x": x_ks, "y": y_ks, "speed": s_ks, "hdir": h_ks, "accuracy": accuracy}
            coefs[session_name][fold] = model.coef_[0]

        if len(fold_accuracies) == 0:
            print(f"Skipping session {session_name} because there are no folds to compute accuracy")
            continue

        # Compute the mean accuracy for the session
        whole_homing[session_name] = np.mean(fold_accuracies)
        print(f"Mean accuracy for session {session_name}: {whole_homing[session_name]}")

        # save the data to a dictionary
        save_data = {"accuracy_data": whole_homing, "ks_data": ks_decoding_comparison, "coefs": coefs}

    # Save the data
    with open(dir / Path(f"{data_type_name}_accuracy_ks_coeffs_data.pkl"), "wb") as f:
        pickle.dump(save_data, f)

    return whole_homing, ks_decoding_comparison, coefs
