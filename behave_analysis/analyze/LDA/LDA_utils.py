"""A script for utility functions associated with LDA"""

import os
import numpy as np
from sklearn.metrics import confusion_matrix
from behave_analysis.analyze.filtering_data.filtering_functions import identify_angles, identify_dist

## --------------- UTILITY FUNCTIONS

def choose_predictors(settings, session, include_rand_points = True):
    '''This function looks at the settings for LDA and creates a list of the variables we want to predict.
    The names in this list are either fields in video_df or will be calculated ad hoc'''
    if np.logical_or(settings.run_LDA == "all_angles", np.logical_and(type(settings.run_LDA) is list, settings.run_LDA[0] == "all_angles")):
        target = identify_angles(session, include_rand_points = include_rand_points)
    elif np.logical_or(settings.run_LDA == "all_distance", np.logical_and(type(settings.run_LDA) is list, settings.run_LDA[0] == "all_distance")):
        target = identify_dist(session,'dist')
    elif np.logical_or(settings.run_LDA == "all_vectors", np.logical_and(type(settings.run_LDA) is list, settings.run_LDA[0] == "all_vectors")):
        target = identify_dist(session,'vect')
    else:
        # this ould be a list of angles
        target = settings.run_LDA
    return target

def list_conditions(settings):
    '''
    This function looks at settings and creates a list of the types of conditions that we will be looking at'''
    number_of_homings = []
    if settings.condition_types == "experimental_conditions":
        condition_types = ["experimental_conditions"]
    elif settings.condition_types == "btwn_escapes":
        condition_types = ["btwn_escapes"] # LDA is run on segments between each escape for each condition
    elif settings.condition_types == "time_conditions":
        condition_types = ["first_half","second_half"]
    elif settings.condition_types == "behavioral_conditions":
        condition_types = ["good_behavioral_conditions", "bad_behavioral_conditions"]
        # good means the times when mousie is doing correct homies,
        # bad is when mouse is doing incorrect homies
    elif "homing_number" in settings.condition_types:
        number_of_homings = int(settings.condition_types.replace("homing_number_", ""))
        condition_types = ["before_" + str(number_of_homings) + "good_homings", "after_" + str(number_of_homings) + "good_homings"]
    return number_of_homings, condition_types

def BuildSavingFolder(basepath, settings, cluster_type, condition_types, condition=[], compartment=[]):
    """
    This function builds a folder structure for where the decoder results and plots will be saved
    under processed_data > models
    - a folder named after the model type with addition of whether PCa cleaning and/or firing rates were used
    - a folder for the cluster type (e.g. 'good', 'synthetic')
    - a folder for the condition types (e.g. 'experimental_conditions', 'behavioral_condition')
    - a folder for each compartment (e.g. 'all', 'threat_zone)
    - a folder for each condition (e.g. 'shelter_only','barrier_pre_flip')
    """
    # folder name
    if settings.discriminant_type == "linear":
        pathh = str(basepath) + "/" + "LDA" + "/"
    elif settings.discriminant_type == "quadratic":
        pathh = str(basepath) + "/" + "QDA" + "/"
    elif settings.discriminant_type == "LSTM":
        pathh = str(basepath) + "/" + "LSTM" + "/"

    if isinstance(settings.run_LDA, list):
        pathh = str(pathh) + 'angle_list'
    else:
        pathh = str(pathh) + settings.run_LDA

    # if subsampling to equalize the bins
    if settings.subsampling:
        pathh = str(pathh) + "_subsampled"

    # if PCA, add to folder name
    if len(settings.PCA_process) > 0:
        pathh = str(pathh) + "_PCA"

    # if using fr, add to folder name
    if settings.use_firing_rate:
        pathh = str(pathh) + "_fr"

    # if excluding proximal points for head angle decoding, add to folder name
    if settings.exclude_proximal > 0:
        pathh = str(pathh) + "_excl_prox_" + str(settings.exclude_proximal) + 'cm' 

    # if excluding hdir
    if settings.exclude_hdir:
        pathh = str(pathh) + "_excl_hdir"

    if settings.exclude_stationary:
        pathh = str(pathh) + "_excl_stationary"

    # add subfolder for cluster type
    pathh = str(pathh) + "/" + str(cluster_type)

    # add subfolder for condition type (experimental or behavioral)
    pathh = str(pathh) + "/" + str(condition_types)

    # if splitting by compartment
    if len(compartment) > 0:
        pathh = str(pathh) + "/" + str(compartment)

    # add subfolder for each condition
    if len(condition) > 0:
        pathh = str(pathh) + "/" + str(condition)

    # if path doesn't exist, make it
    if not (os.path.exists(pathh)):
        os.makedirs(pathh)

    return pathh

def check_if_we_do_LDA(self):
    """
    This function checks whether the decoder and the linear shift has already be run for this set of conditions, clusters, etc.
    It also checks if the user asked to redo compute

    returns: two booleans for whether to run decoder and linear shift
    """
    # if LDA has already been run and saved, don't redo
    self.LDA_out = str(self.savepath) + "/" + str(self.cluster_type) + "_" + str(self.condition) + "_LDA_pa" + ".pkl"
    self.dropout_out = str(self.savepath) + "/" + str(self.cluster_type) + "_" + str(self.condition) + "_LDA_dropout_pa" + ".pkl"
    self.LS_out = str(self.savepath) + "/" + str(self.cluster_type) + "_" + str(self.condition) + "_LDA_LS_pa" + ".pkl"
    self.do_LDA = True
    self.do_dropout = self.settings.dropout
    self.do_LS = self.settings.linear_shift
    if not self.settings.redo_compute:
        if os.path.exists(self.LDA_out):
            self.do_LDA = False
        if os.path.exists(self.LS_out):
            self.do_LS = False
        if os.path.exists(self.dropout_out):
            self.do_dropout = False

def plotConfusionMatrix(y, x, title, axy):
    """
    This function makes a confusion matrix and plots it on the given axes

    INPUTS:
    y - real data
    x - predicted data
    title - tiel of the plot
    axy - axes for plotting

    RETURNS:
    conf - confusion matrix
    """
    conf = confusion_matrix(y, x)
    conf = conf.astype("float64")
    conf = conf / np.sum(conf, axis=1)

    axy.imshow(conf, cmap="Blues", vmin=0, vmax=1)
    axy.set_ylabel("real")
    axy.set_xlabel("predicted")
    axy.set_title(title)
    return conf

def EqualBins_matrix(x, y, unique_fr):
    """
    This function subsamples the input vectors and matrix such that they are composed of an equal number of samples for each unique angle and position bin

    INPUT: x - a matrix of frames x clusters
    y - a vector of length frames of bnned angles
    unique_fr - a vector of length frames of integers that specify which frames belong to the same bin of unique angles and xy position

    RETURNS: x,y,unique_fr subsampled
    """
    angbins, counts = np.unique(unique_fr, return_counts=True)
    samples = np.amin(counts)

    unique_new = []
    for c, i in enumerate(angbins):
        x_filt = x[unique_fr == i, :]
        y_filt = y[unique_fr == i]
        samplingidx = np.random.randint(0, len(x_filt), samples)
        if c == 0:
            x_new = x_filt[samplingidx, :]
            y_new = y_filt[samplingidx]
        else:
            x_new = np.append(x_new, x_filt[samplingidx, :], axis=0)
            y_new = np.append(y_new, y_filt[samplingidx], axis=0)
        unique_new = np.append(unique_new, np.ones(samples) * i)

    y_new = y_new[np.argsort(x_new[:, 0])]
    unique_new = unique_new[np.argsort(x_new[:, 0])]
    x_new = x_new[np.argsort(x_new[:, 0]), :]
    return x_new, y_new, unique_new

def data_chunker(frame_num, epoch_num):
    """
    This function creates a vector of length = number of frames in our dataset
    and randomly divides it into equally populated bins.

    INPUTS:
    epoch_num = the number of bins (the number of epochs for crossvalidaiton)
    frame_num = the length of our dataset

    RETURNS:
    binned_frames
    """

    min_length = frame_num // epoch_num
    num_longer_bins = frame_num % epoch_num
    rng = np.random.default_rng()
    binned_frames = np.hstack((np.repeat(np.arange(epoch_num), min_length), rng.integers(epoch_num, size=num_longer_bins)))
    # rows = np.arange(frame_num)
    # epoch_edge = np.round(np.linspace(np.amin(rows) - 1, np.amax(rows) + 1, epoch_num + 1))
    # binned_frames = np.digitize(rows, epoch_edge)
    return binned_frames

def compute_prediction_accuracy(matrixx):
    """
    This function computes the prediction accuracy given a confusion matrix.
    It takes the mean prediction accuracy of all the values on the diagonal and the two bins adjecent to the diagonal.

    INPUT: confusion matrix

    RETURNS: mean prediction accuracy
    """
    pos = np.floor(np.shape(matrixx)[1] / 2).astype(int)
    pred_acc = np.zeros(np.shape(matrixx)[0])
    for i in np.arange(np.shape(matrixx)[0]):
        x = np.roll(matrixx[i.astype(int), :], pos - i)
        pred_acc[i] = np.sum(x[pos - 1 : pos + 2])
    return np.mean(pred_acc)

def compute_prediction_accuracy_vect(matrixx, key):
    """
    This function computes the prediction accuracy given a confusion matrix.
    It takes the mean prediction accuracy of all the values on the diagonal and the two bins adjecent to the diagonal.

    INPUT: confusion matrix

    RETURNS: mean prediction accuracy
    """
    pos = np.floor(np.shape(matrixx)[1] / 2).astype(int)
    pred_acc = np.zeros(np.shape(matrixx)[0])
    for i in np.arange(np.shape(matrixx)[0]):
        x = np.roll(matrixx[i.astype(int), :], pos - i)
        idx = [pos]
        # append the distance bins near it
        if key[1,0] == np.amin(key[1,:]):
            idx.append(pos+1) # the distance bin after it
        elif key[1,0] == np.amax(key[1,:]):
            idx.append(pos-1)
        else:
            idx.append(pos-1)
            idx.append(pos+1)
        # append the angle bins near it
        d_bins = len(np.unique(key[1,:]))
        idx.append(pos+d_bins)
        idx.append(pos-d_bins)
        pred_acc[i] = np.sum(x[idx])
    return np.mean(pred_acc)

def fill_dict_with_zeros(self,prediction_coef,prediction_accuracy,LDA_y_output,dropout_pa,LS_compiled,variable):
    '''If no frames meet the criteria (the video_df is blank for this condition), make this condition blank'''
    pa = 0
    LS_out = 0

    if self.do_LDA:
        prediction_accuracy.update({variable: pa})
        prediction_coef.update({variable: None})
        LDA_y_output.update({variable: None})
    if self.do_dropout:
        dropout_pa.update({variable:pa})
    if self.do_LS:
        LS_compiled.update({variable: LS_out})

    return prediction_coef,prediction_accuracy,LDA_y_output,LS_compiled,dropout_pa

def correct_variable_name(variable):
    '''Take the variable name for distance and transform it into a head_angle column name in video_df'''
    if "shelt" in variable:
        head_variable = 'hsa'
    elif "bar_preflip" in variable:
        head_variable = 'h_preflipbar_a'
    elif "bar_postflip" in variable:
        head_variable = 'h_postflipbar_a'
    elif "bar_centre" in variable:
        head_variable = 'h_bar_centre_a'
    elif "randP" in variable:
        var = 'randP_vect'
        head_variable = "head_randP_" + (variable[len(var) :])
    return head_variable