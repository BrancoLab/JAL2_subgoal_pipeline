import os

import numpy as np
import dill as pickle
from sklearn.model_selection import train_test_split


def gen_random_pred_array(frame_by_cluster_matrix):
    """Create a random array of the same length as the input matrix
    to use as a dummy prediction which should fail"""

    random_y = np.random.rand(len(frame_by_cluster_matrix), 1) * 2 * np.pi - np.pi
    return np.asarray(random_y).reshape(
        len(random_y),
    )


def split_data(x, y, test_size=0.2):
    """Split the data into training and testing sets"""
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
    return x_train, x_test, y_train, y_test


def remove_hdir_cells(session, cluster_matrix, cluster_ids):
    path = os.path.join(session.base_path, session.processed_path, "cells", "hdir_cells.pkl")

    # Load the hdir cell ids
    with open(path, "rb") as f:
        hdir_cells = pickle.load(f)

    # Using the hdir cell ids, find the index of the position within the cluster_ids
    hdir_cells = np.asarray(hdir_cells).astype(np.int64)
    index = np.where(np.isin(cluster_ids, hdir_cells))[0]

    cluster_matrix = np.delete(cluster_matrix, index - 1, 1)

    return cluster_matrix
