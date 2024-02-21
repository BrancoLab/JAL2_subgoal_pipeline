"""A script for utility functions associated with LDA"""

import os
import numpy as np
from sklearn.metrics import confusion_matrix

## --------------- UTILITY FUNCTIONS


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
        pathh = str(basepath) + "/" + "LDA"
    elif settings.discriminant_type == "quadratic":
        pathh = str(basepath) + "/" + "QDA"
    elif settings.discriminant_type == "LSTM":
        pathh = str(basepath) + "/" + "LSTM"

    # if PCA, add to folder name
    if len(settings.PCA_process) > 0:
        pathh = str(pathh) + "_PCA"

    # if using fr, add to folder name
    if settings.use_firing_rate:
        pathh = str(pathh) + "_fr"

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


def check_if_we_do_LDA(self, settings):
    """
    This function checks whether the decoder and the linear shift has already be run for this set of conditions, clusters, etc.
    It also checks if the user asked to redo compute

    returns: two booleans for whether to run decoder and linear shift
    """
    # if LDA has already been run and saved, don't redo
    LDA_out = str(self.savepath) + "/" + str(self.cluster_type) + "_" + str(self.condition) + "_LDA_prediction_accuracy" + ".pkl"
    LS_out = str(self.savepath) + "/" + str(self.cluster_type) + "_" + str(self.condition) + "_LDA_LS_prediction_accuracy" + ".pkl"
    do_LDA = True
    do_LS = settings.linear_shift
    if not settings.redo_compute:
        if os.path.exists(LDA_out):
            do_LDA = False
        if os.path.exists(LS_out):
            do_LS = False

    return LDA_out, LS_out, do_LDA, do_LS


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
