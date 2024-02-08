"""All the scripts that process the LDA inputs
- process the angles by subselecting the frames to use
- process the neural data
 """

import numpy as np
from sklearn.decomposition import PCA

from behave_analysis.analyze.filtering_data.filtering_functions import (
    filter_video_dataframe,
    generate_bin_angles,
    filter_video_df_mouse_behaviour,
    filter_video_df_homing_number,
)
from behave_analysis.analyze.LDA.LDA_utils import EqualBins_matrix, data_chunker

## --------------- PROCESS LDA INPUTS


def select_relevant_frames(self):
    # subselect relevant times based on condition types ( experimentally or behaviorally defined)
    if self.condition_types == "experimental_conditions":
        filtered_video_df = filter_video_dataframe(self.video_df, self.condition)
    else:
        filtered_video_df = filter_video_dataframe(self.video_df, self.condition, exclude_escape=False)
        if self.condition_types == "good_behavioral_conditions":
            filtered_video_df = filter_video_df_mouse_behaviour(filtered_video_df, self.condition, self.session, good_homie=True)
        elif self.condition_types == "bad_behavioral_conditions":
            filtered_video_df = filter_video_df_mouse_behaviour(filtered_video_df, self.condition, self.session, good_homie=False)
        elif self.condition_types == 'after_'+str(self.number_of_homings)+'good_homings':
            filtered_video_df = filter_video_df_homing_number(filtered_video_df, self.condition, self.session, good_homie=True, number_of_homings = self.number_of_homings)
        elif self.condition_types == 'before_'+str(self.number_of_homings)+'good_homings':
            filtered_video_df = filter_video_df_homing_number(filtered_video_df, self.condition, self.session, good_homie=False, number_of_homings = self.number_of_homings)

    # subselect relevant frames based on compartment
    if self.compartment == "threat_zone":
        filtered_video_df = filtered_video_df.filter((filtered_video_df["mouse_y_position"].to_numpy() > 512))
    elif self.compartment == "shelter_compartment":
        filtered_video_df = filtered_video_df.filter((filtered_video_df["mouse_y_position"].to_numpy() < 512))

    return filtered_video_df


def BinDfbyAngle(self, variable, settings):
    """
    A function that processes dataframe for discriminant analysis
    variable: what we're trying to predict (e.g. head_shelter_angle), it needs to be one of the columns of video_df
    """
    # edges for binning firing rate at different angles
    bin_angles, _ = generate_bin_angles(settings.number_of_bins)

    title = str(variable + "_" + self.condition)
    filtered_video_df = self.filtered_video_df.select(["frames", variable])
    frames = filtered_video_df["frames"].unique().to_numpy() - 1

    # bin angles
    binned_angles = np.array(filtered_video_df[variable].to_numpy())

    # median filter! no longer used
    # binned_angles = np.arctan2(sp.medfilt(np.sin(binned_angles),41),sp.medfilt(np.cos(binned_angles),41))

    binned_angles = np.digitize(binned_angles, bin_angles)

    return binned_angles, frames, title


def BinDfbyPos(self):
    """
    A function that processes dataframe for discriminant analysis
    variable: what we're trying to predict (e.g. head_shelter_angle), it needs to be one of the columns of video_df
    """
    mouse_x = self.filtered_video_df["mouse_x_position"].to_numpy()
    mouse_y = self.filtered_video_df["mouse_y_position"].to_numpy()

    # bin into quadrants
    mouse_x = mouse_x > (self.session.video.height / 2)
    mouse_y = mouse_y > (self.session.video.width / 2)

    _, binned_pos = np.unique(np.vstack((mouse_x, mouse_y)), axis=1, return_inverse=True)

    return binned_pos


def binDfbyEpoch(matrix, matriy, binned_pos, epoch_num):
    _, unique_pos_ang = np.unique(np.vstack((binned_pos, matriy)), axis=1, return_inverse=True)

    # make angle + position bins equally populated
    matrix, matriy, unique_pos_ang = EqualBins_matrix(matrix, matriy, unique_pos_ang)  # this step randomly subsamples!!

    # chunk data into training and test data for each angle bin!!
    epochs = np.empty_like(matriy)
    bins = np.unique(unique_pos_ang)
    for i in bins:
        x_filt = matrix[unique_pos_ang == i, :]
        binned_frames = data_chunker(np.shape(x_filt)[0], epoch_num)
        epochs[unique_pos_ang == i] = binned_frames

    epochs = epochs[np.argsort(matrix[:, 0])]

    return matrix, matriy, epochs


def ProcessPredictors(self, frames, settings):
    # select frames that have been filtered
    X = self.frame_by_cluster_matrix
    X = X[frames, :]

    # remove NaN columns (empty clusters)
    nancolumns = np.where(np.sum(X == 0, axis=0) == np.shape(X)[0])[0]
    if len(nancolumns) > 0:
        X = np.delete(X, nancolumns, axis=1)

    # normalize firing rates
    # X = X/np.amax(X,axis=0)

    # z-score firing rates
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # optional: run PCA
    if settings.PCA_process:
        pca = PCA(n_components=15)
        X = pca.fit_transform(X)

    # first column of X is frame num
    X = np.c_[frames, X]

    return X
