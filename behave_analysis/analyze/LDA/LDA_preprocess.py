"""All the scripts that process the LDA inputs
- process the angles by subselecting the frames to use
- process the neural data
 """
import os
import dill as pickle
import numpy as np
from sklearn.decomposition import PCA

from behave_analysis.utils.creating_directories import make_directory
from behave_analysis.analyze.filtering_data.filtering_functions import (
    filter_video_dataframe,
    generate_bin_angles,
    filter_video_df_mouse_behaviour,
    filter_video_df_homing_number,
)
from behave_analysis.analyze.LDA.LDA_utils import EqualBins_matrix, data_chunker

## --------------- PROCESS LDA INPUTS


def select_relevant_frames(self):
    '''
    Based on the condition types (experimentally or behaviorally defined),
    and on the compartments the user wants to look at (e.g. 'all','threat_zone')
    this function subselects relevant frames

    RETURNS: filtered_video_df - a subset of video_df with only the relevant frames
    '''

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
        filtered_video_df = filtered_video_df.filter((filtered_video_df["mouse_y_position"].to_numpy() < 512))
    elif self.compartment == "shelter_compartment":
        filtered_video_df = filtered_video_df.filter((filtered_video_df["mouse_y_position"].to_numpy() > 512))
    if self.compartment == "left_arena":
        filtered_video_df = filtered_video_df.filter((filtered_video_df["mouse_x_position"].to_numpy() < 512))
    elif self.compartment == "right_arena":
        filtered_video_df = filtered_video_df.filter((filtered_video_df["mouse_x_position"].to_numpy() > 512))

    return filtered_video_df

def exclude_proximal_frames(video_df, variable, tracking, dist_thresh):
    '''This function takes a video_df and a point as inputs. 
    It computes the distance of the mouse to that point at every frame. 
    It then reduces the video_df to only include points where the mouse was > dist_thresh away from that point'''
    # find coordinates of the relevant point
    if variable == 'hsa':
        point = [int(np.mean([tracking['shelter_loc'][0][0],tracking['shelter_loc'][1][0]])),
                 int(np.mean([tracking['shelter_loc'][0][1],tracking['shelter_loc'][1][1]]))]
    elif variable == 'h_bar_north_a':
        point = tracking['barrier_loc'][0]
    elif variable == 'h_bar_south_a':
        point = tracking['barrier_loc'][1]
    elif variable == 'h_bar_centre_a':
        point = tracking['barrier_loc'][2]
    elif 'randP' in variable:
        num = int(variable[len('randP'):])
        point = tracking["randP_loc"][num, :]
    
    # measure the distance of the mouse from that point
    X = video_df['mouse_x_position'].to_numpy()
    Y = video_df['mouse_y_position'].to_numpy()
    dist = np.sqrt(((X-point[0])**2)+((Y-point[1])**2))

    # filter video_df
    video_df = video_df.filter(dist > dist_thresh)

    return video_df

def BinDfbyAngle(self, variable, settings):
    """
    A function that bins the angles of interest extracting them from the behavioral dataframe

    INPUTS: variable - what we're trying to predict (e.g. head_shelter_angle), it needs to be one of the columns of video_df

    RETURNS: binned_angles - a vector of binned angles we're trying to decode
    frames - a vector of the same length as binned_angles with the frame number of the frames utilized in this condition
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


def BinDfbyPos(filtered_video_df,video_height,video_width):
    """
    A function that bins the x-y position of the mouse extracting them from the behavioral dataframe
    """
    mouse_x = filtered_video_df["mouse_x_position"].to_numpy()
    mouse_y = filtered_video_df["mouse_y_position"].to_numpy()

    # bin into quadrants
    mouse_x = mouse_x > (video_height / 2)
    mouse_y = mouse_y > (video_width / 2)

    _, binned_pos = np.unique(np.vstack((mouse_x, mouse_y)), axis=1, return_inverse=True)

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
    matriy = pos_ang[0,:]

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
    '''
    This is a function for processing the frames x cluster matrix to get it ready for decodr analysis
    '''
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
        X = X[:,boolean_cluster == False]

    # add a first column to X to be frame num
    X = np.c_[frames, X]

    return X

def zscore_predictors(X):
    '''This function z-scores an input matrix
    if a cluster (column) has all zero values than the output will be a column of zeros
    the z-scoring will not be computed because the std(0) is 0 and we can't divide by 0'''
    ZscoredX = np.zeros_like(X)
    nonzero_clu = np.where(np.sum(X == 0, axis = 0) < np.shape(X)[0])
    # z-score firing rates
    ZscoredX[:,nonzero_clu] = (X[:,nonzero_clu] - np.mean(X[:,nonzero_clu], axis=0)) / np.std(X[:,nonzero_clu], axis=0)
    
    # nanclusters should be zero clusters
    # nanclusters = np.where(np.sum(np.isnan(X),axis=0) == np.shape(X)[0])[0]
    # if len(nanclusters) > 0:
    #     X[:,nanclusters] = np.zeros((np.shape(X)[0],1))
    return ZscoredX