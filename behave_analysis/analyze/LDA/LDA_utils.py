"""A script for utility functions associated with LDA"""

import os
import numpy as np
from sklearn.metrics import confusion_matrix

## --------------- UTILITY FUNCTIONS

def BuildSavingFolder(basepath, settings, cluster_type, condition_types, condition=[], compartment=[]):
    # folder name
    if settings.discriminant_type == "linear":
        pathh = str(basepath) + "/" + "LDA"
    elif settings.discriminant_type == "quadratic":
        pathh = str(basepath) + "/" + "QDA"

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
    conf = confusion_matrix(y, x)
    conf = conf.astype("float64")
    conf = conf / np.sum(conf, axis=1)

    axy.imshow(conf, cmap="Blues", vmin=0, vmax=1)
    axy.set_ylabel("real")
    axy.set_xlabel("predicted")
    axy.set_title(title)
    return conf


def EqualBins_matrix(x, y, unique_fr):
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
    rows = np.arange(frame_num)
    epoch_edge = np.round(np.linspace(np.amin(rows) - 1, np.amax(rows) + 1, epoch_num + 1))
    binned_frames = np.digitize(rows, epoch_edge)
    return binned_frames


def compute_prediction_accuracy(matrixx):
    pos = np.floor(np.shape(matrixx)[1] / 2).astype(int)
    pred_acc = np.zeros(np.shape(matrixx)[0])
    for i in np.arange(np.shape(matrixx)[0]):
        x = np.roll(matrixx[i.astype(int), :], pos - i)
        pred_acc[i] = np.sum(x[pos - 1 : pos + 2])
    return np.mean(pred_acc)
