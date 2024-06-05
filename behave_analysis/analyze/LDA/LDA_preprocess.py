"""All the scripts that process the LDA inputs
- process the angles by subselecting the frames to use
- process the neural data
 """

import os
import dill as pickle
import numpy as np
from sklearn.decomposition import PCA
import polars as pl

from behave_analysis.utils.creating_directories import make_directory
from behave_analysis.analyze.filtering_data.filtering_functions import (
    filter_video_dataframe,
    generate_bins,
    filter_video_df_mouse_behaviour,
    filter_video_df_homing_number,
)
from behave_analysis.analyze.LDA.LDA_utils import EqualBins_matrix, data_chunker, correct_variable_name

## --------------- PROCESS LDA INPUTS


def select_relevant_frames(self):
    """
    Based on the condition types (experimentally or behaviorally defined),
    and on the compartments the user wants to look at (e.g. 'all','threat_zone')
    this function subselects relevant frames

    RETURNS: filtered_video_df - a subset of video_df with only the relevant frames
    """

    if self.condition_types == "experimental_conditions":
        filtered_video_df = filter_video_dataframe(self.video_df, self.condition)
    else:
        filtered_video_df = filter_video_dataframe(self.video_df, self.condition, exclude_escape=False)
        if self.condition_types == "good_behavioral_conditions":
            filtered_video_df = filter_video_df_mouse_behaviour(filtered_video_df, self.condition, self.session, good_homie=True)
        elif self.condition_types == "bad_behavioral_conditions":
            filtered_video_df = filter_video_df_mouse_behaviour(filtered_video_df, self.condition, self.session, good_homie=False)
        elif self.condition_types == "after_" + str(self.number_of_homings) + "good_homings":
            filtered_video_df = filter_video_df_homing_number(
                filtered_video_df, self.condition, self.session, good_homie=True, number_of_homings=self.number_of_homings
            )
        elif self.condition_types == "before_" + str(self.number_of_homings) + "good_homings":
            filtered_video_df = filter_video_df_homing_number(
                filtered_video_df, self.condition, self.session, good_homie=False, number_of_homings=self.number_of_homings
            )

    # subselect relevant frames based on compartment
    if self.compartment == "threat_zone":
        filtered_video_df = filtered_video_df.filter((filtered_video_df["mouse_y_position"].to_numpy() < 512))
    elif self.compartment == "shelter_compartment":
        filtered_video_df = filtered_video_df.filter((filtered_video_df["mouse_y_position"].to_numpy() > 512))
    if self.compartment == "left_arena":
        filtered_video_df = filtered_video_df.filter((filtered_video_df["mouse_x_position"].to_numpy() < 512))
    elif self.compartment == "right_arena":
        filtered_video_df = filtered_video_df.filter((filtered_video_df["mouse_x_position"].to_numpy() > 512))

    return filtered_video_df


def exclude_proximal_frames(video_df, variable, tracking, dist_thresh):
    """This function takes a video_df and a point as inputs.
    It computes the distance of the mouse to that point at every frame.
    It then reduces the video_df to only include points where the mouse was > dist_thresh away from that point"""

    dist = distance_mouse_point(video_df, variable, tracking)

    if np.logical_or("dist" in variable, "vect" in variable):
        video_df = video_df.hstack([pl.Series(variable, dist)])

    # filter video_df based on distance
    video_df = video_df.filter(dist > dist_thresh)

    return video_df


def distance_mouse_point(video_df, variable, tracking, dist_to_centre=False, centre=[]):
    """It computes the distance of the mouse to that point at every frame"""
    # find coordinates of the relevant point
    if np.logical_or(variable == "hsa", "shelt" in variable):
        point = [
            int(np.mean([tracking["shelter_loc"][0][0], tracking["shelter_loc"][1][0]])),
            int(np.mean([tracking["shelter_loc"][0][1], tracking["shelter_loc"][1][1]])),
        ]
    elif "bar_north" in variable:
        point = tracking["barrier_loc"][0]
    elif "bar_south" in variable:
        point = tracking["barrier_loc"][1]
    elif "bar_centre" in variable:
        point = tracking["barrier_loc"][2]
    elif "randP" in variable:
        var = "randP"
        if "head" in variable:
            var = "head_" + var + "_"
        if "dist" in variable:
            var = var + "_dist"
        if "vect" in variable:
            var = var + "_vect"
        num = int(variable[len(var) :])
        point = tracking["randP_loc"][num, :]

    if dist_to_centre:
        # what is the distance of the point to the centre of the arena
        centre_dist = np.sqrt(((centre[0] - point[0]) ** 2) + ((centre[1] - point[1]) ** 2))
        return centre_dist

    # measure the distance of the mouse from that point
    X = video_df["mouse_x_position"].to_numpy()
    Y = video_df["mouse_y_position"].to_numpy()
    dist = np.sqrt(((X - point[0]) ** 2) + ((Y - point[1]) ** 2))

    return dist


def BinDfbyAngle(self, variable, n_bins):
    """
    A function that bins the angles of interest extracting them from the behavioral dataframe

    INPUTS: variable - what we're trying to predict (e.g. head_shelter_angle), it needs to be one of the columns of video_df

    RETURNS: binned_angles - a vector of binned angles we're trying to decode
    """
    # edges for binning firing rate at different angles
    bins, bin_centre = generate_bins(n_bins, -np.pi, np.pi)

    # bin angles
    binned_angles = np.array(self.filtered_video_df[variable].to_numpy())

    binned_angles = np.digitize(binned_angles, bins)

    return binned_angles, bins, bin_centre


def BinDfbyDistance(self, variable, n_bins):
    """
    A function that bins the distances of interest extracting them from the behavioral dataframe

    INPUTS: variable - what we're trying to predict (e.g. shelt_dist) - if this variable is not in video_df we will need to compute it

    RETURNS: binned_distance - a vector of binned distances we're trying to decode
    """
    if variable not in self.filtered_video_df.columns:
        distance = distance_mouse_point(self.filtered_video_df, variable, self.tracking_data)
        self.filtered_video_df = self.filtered_video_df.hstack([pl.Series(variable, distance)])
    else:
        distance = np.array(self.filtered_video_df[variable].to_numpy())

    distance = distance / self.session.video.pixels_per_cm
    self.bins, self.bin_centre = generate_bins(n_bins, np.amin(distance), 95)  # in cm 95 is the diameter of the arena

    # figure out what the max distance is for this point in the arena, reduce bins to max
    if hasattr(self.session.video, "radius"):  # in the future radius should always be there and we can delete this line
        radius = self.session.video.radius
    else:
        radius = 460
    centre_dist = distance_mouse_point(
        self.filtered_video_df,
        variable,
        self.tracking_data,
        dist_to_centre=True,
        centre=[self.session.video.height / 2, self.session.video.width / 2],
    )
    # this actually uses all available distances for a given point, but that means the max distance is different for different points, so maybe not valid
    # max_dst = (radius + centre_dist) / self.session.video.pixels_per_cm
    max_dst = (radius) / self.session.video.pixels_per_cm

    # which bin edge is closest to the max_dst? that is our new biggest allowed big
    max_bin = np.argmin(np.abs(self.bins - max_dst))
    self.bins = self.bins[: max_bin + 1]
    self.bin_centre = self.bin_centre[:max_bin]

    binned_dist = np.digitize(distance, self.bins)

    return binned_dist

def BinDfbyVector(self, variable, dist_n_bins,angle_n_bins):

    dist_n_bins = 11 # this will be chunked down to 5 bins of 10 cm anyway
    angle_n_bins = 9 # more than this definitely doesn't give us enough data
    binned_dist = BinDfbyDistance(self, variable, dist_n_bins)
    head_var = correct_variable_name(variable)
    binned_angles,_,bin_centre_angle = BinDfbyAngle(self, head_var, angle_n_bins)

    vect = np.vstack((binned_angles,binned_dist))
    uni_vect_key, uni_vect = np.unique(vect,axis=1,return_inverse=True) 
    uni_vect = uni_vect + 1 # don't want 0 indexing!
    ref_bins = np.unique(uni_vect)
    # uni_vect is a vector len(frames) in which each value indicates a unique vector bin
    # uni_vect_key tells us how to interpret the vector bins in terms of distance and angle
    # now make uni_vect an actually legible value
    bin_centre_dist = np.hstack((self.bin_centre,95))
    bin_key = np.vstack((bin_centre_angle[uni_vect_key[0,:]-1],bin_centre_dist[uni_vect_key[1,:]-1]))

    # all frames that have binned_dist > len(self.bin_centre) should be set to zero so they can be eliminated later
    bad = binned_dist > len(self.bin_centre)
    uni_vect[bad] = 0
    self.bin_centre = ref_bins

    return uni_vect, bin_key

def BinDfbyPos(filtered_video_df, video_height, video_width, numpoints = 3, return_bin_centre = False):
    """
    A function that bins the x-y position of the mouse extracting them from the behavioral dataframe
    """
    mouse_x = filtered_video_df["mouse_x_position"].to_numpy()
    mouse_y = filtered_video_df["mouse_y_position"].to_numpy()

    bins, bin_x_centre = generate_bins(numpoints,(video_height/2) - 460,(video_height/2) + 460) # 3 points gives us two spatial bins and four quadrants
    mouse_x = np.digitize(mouse_x, bins)
    mouse_x[mouse_x > len(bin_x_centre)] = 0
    bin_mouse_x = bin_x_centre[mouse_x - 1]

    bins, bin_y_centre = generate_bins(numpoints,(video_width/2) - 460,(video_width/2) + 460) # 3 points gives us two spatial bins and four quadrants
    mouse_y = np.digitize(mouse_y, bins)
    mouse_y[mouse_y > len(bin_y_centre)] = 0
    bin_mouse_y = bin_y_centre[mouse_y - 1]

    # the bad frames
    bad_frames = np.logical_or(mouse_x == 0, mouse_y == 0)
    bin_mouse_x[bad_frames] = 0
    bin_mouse_y[bad_frames] = 0

    dist = np.sqrt(((bin_mouse_x - video_height/2)**2) + ((bin_mouse_y - video_width/2)**2))
    outside_arena = np.where(dist>460)[0]

    if len(outside_arena) > 0:
        bin_mouse_x[outside_arena] = 0
        bin_mouse_y[outside_arena] = 0

    bin_centre, binned_pos = np.unique(np.vstack((bin_mouse_x, bin_mouse_y)), axis=1, return_inverse=True)
    # binned_pos should be zero where bin_mouse_x and bin_mouse_y are zero
    assert np.unique(binned_pos[bin_mouse_x == 0])[0] == 0

    if return_bin_centre:
        bin_centre = bin_centre[:,np.sum(bin_centre == 0, axis = 0) == 0]
        return binned_pos, bin_centre
    else:
        return binned_pos


def binDfbyEpoch(matrix, pos_ang, epoch_num):
    """
    A function that splits the data into n epochs for crossvalidation.
    It also subsamples the data so that each epoch is populated by uniformly distributed data of angles and positions

    INPUTS: matrix - the frames x cluster matrix
    matriy - vector of angles to decode
    binned_pos - vector of same length as matriy with binned position of the mouse
    epoch_num - number of epochs to divide data in

    RETURNS: matrix - the subsampled frames x cluster matrix
    matriy - subsampled vector of angles to decode
    epochs - a vector of the same length as matriy with the epochs that each frame is assigned to
    """
    _, unique_pos_ang = np.unique(pos_ang, axis=1, return_inverse=True)
    matriy = pos_ang[0, :]

    # make angle + position bins equally populated
    matrix, matriy, unique_pos_ang = EqualBins_matrix(matrix, matriy, unique_pos_ang)  # this step randomly subsamples!!

    # chunk data into training and test data for each angle bin!!
    epochs = np.empty_like(matriy)
    bins = np.unique(unique_pos_ang)
    for i in bins:
        y_filt = matriy[unique_pos_ang == i]
        binned_frames = data_chunker(np.shape(y_filt)[0], epoch_num)
        epochs[unique_pos_ang == i] = binned_frames

    epochs = epochs[np.argsort(matrix[:, 0])]

    return matrix, matriy, epochs


def ProcessPredictors(self, frames, settings):
    """
    This is a function for processing the frames x cluster matrix to get it ready for decodr analysis
    """
    # select frames that have been filtered
    X = self.frame_by_cluster_matrix
    X = X[frames, :]

    # remove NaN columns (empty clusters)
    nancolumns = np.where(np.sum(X == 0, axis=0) == np.shape(X)[0])[0]
    if len(nancolumns) > 0:
        X = np.delete(X, nancolumns, axis=1)

    # z-score firing rates
    # X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # optional: run PCA
    if settings.PCA_process:
        pca = PCA(n_components=15)
        X = pca.fit_transform(X)

    if settings.exclude_hdir:
        path = make_directory(os.path.join(self.session.base_path, self.session.processed_path, "cells"))
        file_name = os.path.join(path, "hdir_cells.pkl")
        # TODO: write conditional that if there are no classified cells you need to classify
        with open(file_name, "rb") as dill_file:
            hdir_cells = pickle.load(dill_file)
        # match columns to cluster_Ids
        boolean_cluster = np.isin(self.cluster_Ids, hdir_cells)
        # delete those columns
        X = X[:, boolean_cluster == False]

    # add a first column to X to be frame num
    X = np.c_[frames, X]

    return X


def zscore_predictors(X):
    """This function z-scores an input matrix
    if a cluster (column) has all zero values than the output will be a column of zeros
    the z-scoring will not be computed because the std(0) is 0 and we can't divide by 0"""
    ZscoredX = np.zeros_like(X)
    nonzero_clu = np.where(np.sum(X == 0, axis=0) < np.shape(X)[0])
    # z-score firing rates
    ZscoredX[:, nonzero_clu] = (X[:, nonzero_clu] - np.mean(X[:, nonzero_clu], axis=0)) / np.std(X[:, nonzero_clu], axis=0)

    # nanclusters should be zero clusters
    # nanclusters = np.where(np.sum(np.isnan(X),axis=0) == np.shape(X)[0])[0]
    # if len(nanclusters) > 0:
    #     X[:,nanclusters] = np.zeros((np.shape(X)[0],1))
    return ZscoredX


def prep_target_and_predictors(self, variable, settings):

    # create a unique identifier name for this LDA iteration
    savename = str(variable + "_" + self.condition)
    # extract frame numbers
    frames = self.filtered_video_df["frames"].unique().to_numpy() - 1

    # bin the target values into classes
    if "dist" in variable:
        hdir, _, _ = BinDfbyAngle(self, "hdir", 5)  #  this is kind of dumb, but the order matters here because you need to overwrite self.bins
        binned_target = BinDfbyDistance(self, variable, self.number_of_bins)
        target = np.vstack((binned_target.T, hdir))  # at each distance make sure we're sampling a somewhat even set of
    elif 'vect' in variable:
        binned_target, self.target_key = BinDfbyVector(self, variable, dist_n_bins = 6,angle_n_bins = 13)
        target = np.vstack((binned_target.T, np.ones_like(binned_target.T)))  # we're taking all bins, not further equalizing them
    else:
        binned_target, self.bins, self.bin_centre = BinDfbyAngle(self, variable, self.number_of_bins)
        target = np.vstack((binned_target.T, self.filtered_video_df["binned_position"].to_numpy()))

    # prep the predictor matrix
    X = ProcessPredictors(self, frames, settings)

    return savename, target, X
