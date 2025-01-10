import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from JR_test_scripts.escape.escape_utils import firing_by_bin 

def neuron_tuning_by_var(esc_var, escape_matrix, cond, h_start = [], epoch_method = 'trial', xval_method = 'cosinesim', n_epochs = 3, xval_thresh = .7):
    """What is the tuning curve and peak firing bin of each neuron per condition?
    INPUTS:
        esc_var: is in time, the variable we want to compute the tuning for
        escape_matrix: is neurons x time
        cond: a vector of length time indicating what experimental condition the homing/escape was in
        h_start: the start time of the homings or escapes, duration of homing/escape is cropped to when the mouse reaches shelter
    RETURNS:
        peak_firing_condition is a matrix of neurons x condition, where each entry is the bin with the peak firing for that nruon in that condition to the behavioral variable
        tuning_by_cond: is a list of len(conditions), each entry is a matrix of tuning curves of shape neurons x bins
        xval: vector of length neurons x conditions, indicating if the neuron's tuning passed xval"""
    peak_firing_condition = np.zeros((np.shape(escape_matrix)[0], len(np.unique(cond))))
    tuning_by_cond = []
    xval = np.zeros((np.shape(escape_matrix)[0], len(np.unique(cond))))
    cond_start = []
    for i in np.unique(cond):
        # start by condition
        if len(h_start) > 0:
            start = [x for x in h_start if cond[x] == i]
            cond_start = [x-start[0] for x in start]
        tuning_matrix, xval[:,int(i)] = create_xval_tuning_curve(esc_var[cond == i], 
                                                                 escape_matrix[:,cond == i], 
                                                                 start = cond_start, # resetting the timestamp of homing starts to align to the start of the condition
                                                                 bins = int(np.amax(esc_var)+1),
                                                                 epoch_method = epoch_method, 
                                                                 xval_method = xval_method,
                                                                 xval_thresh = xval_thresh, 
                                                                 n_epochs = n_epochs)
        # bins = esc_var[cond == i]
        peak_firing = np.argmax(tuning_matrix, axis = 1)
        peak_firing_condition[:,int(i)] = peak_firing # bins[peak_firing]
        tuning_by_cond.append(tuning_matrix)
    return peak_firing_condition, tuning_by_cond, xval.astype(int)

def create_xval_tuning_curve(esc_var, escape_matrix, bins, start = [], xval_thresh = .7, epoch_method = 'trial', xval_method = 'cosinesim', n_epochs = 3, normalize_tuning_curve = False, plot = False):
    """Function that computes the tuning of each neuron for a given variable, cross vlaidates it and checks if it's a reliable cell
    several methods can be implemented for xval: separate by trial, even and odd time point, time
    NB: you can call this by condition or for all time!
    
    INPUTS:
        esc_var: is in time, the variable we want to compute the tuning for
        escape_matrix: is neurons x time
        start: if using 'trial' method for xval, this is the start time of the trials (if dividing by condition this will have been adjusted to the start of the condition)
        bins: number of bins for the tuning curve
        epoch_method: how to chunk the epochs, options are 'time', 'trial', 'alt_time'
        xval_method: how to compare the tuning curves, options are 'corr', 'mse', 'cosinesim', 'euclid'
        n_epochs: how many epochs we want to chunk into
        normalize_tuning_curve: if True normalize the tuning curve to 1, if False keep the raw values
    
    RETURNS:
        full_tuning: is neurons x bins, where each line is the tuning of that neuron for the variable
        xval_pass: is len(neurons) and is True if the neuron passed the xval test
    """
    # divide into epochs
    epochs = create_xval_epochs(len(esc_var), start, n_epochs, epoch_method)

    result = np.zeros((np.shape(escape_matrix)[0],len(np.unique(epochs))))
    for i in np.unique(epochs):

        # compute tuning curve for each epoch
        test_var = esc_var[epochs == i]
        train_var = esc_var[epochs != i]
        test_mat = escape_matrix[:,epochs == i]
        train_mat = escape_matrix[:,epochs != i]
        test_tuning = creat_tuning_curve(test_var, test_mat, bins)
        train_tuning = creat_tuning_curve(train_var, train_mat, bins)

        # compare tuning curves, iterate over neurons and compute the similarity
        for it in np.arange(len(result)):
            curve1 = test_tuning[it,:]
            curve2 = train_tuning[it,:]
            if normalize_tuning_curve:
                curve1 = test_tuning[it,:]/np.amax(test_tuning[it,:])
                curve2 = train_tuning[it,:]/np.amax(train_tuning[it,:])
                curve1[np.logical_or(test_tuning[it,:] == 0, np.isinf(curve1))] = 0
                curve2[np.logical_or(train_tuning[it,:] == 0, np.isinf(curve2))] = 0
            if xval_method == 'corr':
                result[it,int(i)], _ = pearsonr(curve1, curve2)
            # if xval_method == 'mse': # don't like it, values not spread well
            #     result[it,int(i)] = np.mean((curve1 - curve2)** 2)
            if xval_method == 'cosinesim':
                result[it,int(i)] = cosine_similarity(curve1.reshape(1, -1),curve2.reshape(1, -1))[0,0]
            if xval_method == 'euclid':
                result[it,int(i)] = np.linalg.norm(curve1 - curve2) / np.linalg.norm(curve1 + curve2)

    # average results across epochs
    avg_result = np.mean(result, axis = 1)
    if np.logical_or(xval_method == 'corr', xval_method == 'cosinesim'):
        xval_pass = avg_result > xval_thresh
    if xval_method == 'euclid':
        xval_pass = avg_result < xval_thresh # recommend .45

    if plot:
        _, axs = plt.subplots(2,2,figsize = (3,6))
        for i in [0,1]:
            t = test_tuning[xval_pass == i,:]
            idx = np.argmax(t, axis = 1)
            isort = np.argsort(idx)
            axs[i,0].set_ylabel('neurons sorted by test tuning \nxval pass = ' + str(i), fontsize = 8)
            axs[i,0].imshow(t[isort,:], cmap="gray_r", vmin = 0, vmax = 1.2, aspect="auto", interpolation = "none")
            t2 = train_tuning[xval_pass == i,:]
            axs[i,1].imshow(t2[isort,:], cmap="gray_r", vmin = 0, vmax = 1.2, aspect="auto", interpolation = "none")
        axs[0,0].set_title('test tuning')
        axs[0,1].set_title('train tuning')
        plt.tight_layout()
        plt.show()

    # compute tuning curve for all time
    full_tuning = creat_tuning_curve(esc_var, escape_matrix, bins)

    return full_tuning, xval_pass

def creat_tuning_curve(esc_var, escape_matrix, nbins):
    """This function creates a matrix of neurons x bins, where each line is the tuning of that neuron for the variable
    INPUT: 
        escape_matrix: is neurons x time
        esc_var: is in time
    
    RETURNS:
        tuning_matrix: is neurons x bins, where each line is the tuning of that neuron for the variable
    """
    tuning_matrix = np.empty((np.shape(escape_matrix)[0],nbins))
    for i, n in enumerate(escape_matrix):
        tuning_matrix[i,:] = firing_by_bin(esc_var.astype(int), n, nbins, remove_empty = False)
    return tuning_matrix

def create_xval_epochs(time, start, n_epochs, epoch_method):
    """This function creates a vector of integers indicating the epochs for cross validation
    INPUTS:
        time: in frames usually (it's the length of time we want to chunk into epochs), gives us the length of the epochs
        n_epochs: how many epochs we want to chunk into
        epoch_method: how to chunk the epochs, options are 'time', 'trial', 'alt_time'
        start: if using 'trial' method, this is the start time of the trials
    
    RETURNS:
        epochs: a vector of integers indicating the epochs for cross validation, of length var
    """
    epochs = np.zeros(time)
    if epoch_method == 'time':
        transitions = [int(time/4),int(time/2),int((time/4)*3)]
        for i in transitions:
            epochs[i:] += 1
    if epoch_method == 'alt_time':
        values = np.arange(n_epochs)
        block_size = 40*3 # 40 frames per second, 3 seconds per epoch
        epochs = np.tile(np.repeat(values, block_size), time // (len(values) * block_size) + 1)
        epochs = epochs[:time]
    if epoch_method == 'trial':
        for st in start[1:-1]:
            epochs[st:] += 1
            epochs = np.mod(epochs, n_epochs) # these epochs are not equal in size, because the trials are all of different lengths!!
    return epochs

def single_trial_tuning(escape_matrix, var, cond, h_start):
    """Make a matrix for each neuron of activity per bin on each trial
    INPUT: 
        escape_matrix: is neurons x time (firing rates)
        esc_var: is in time, the behavioral variable of interest (discretized)
        cond: a vector of length time indicating what experimental condition the homing/escape was in
        h_start: the start time of the homings or escapes, duration of homing/escape is cropped to when the mouse reaches shelter
        """
    mat_by_cond = []
    # iterate through conditions
    for i in np.unique(cond):
        # start by condition
        cond_start = [x for x in h_start if cond[x] == i]
        cond_start.append(np.sum(cond == i))
        # initialize variable, neurons x trials x bins
        bins = int(np.amax(var)+1)
        mat = np.zeros((np.shape(escape_matrix)[0],len(cond_start),bins))
        # iterate through neurons
        for j, n in enumerate(escape_matrix):
            # iterate through trials, pull out firing by bin
            for tr, _ in enumerate(cond_start[:-1]):
                neur = n[cond_start[tr]:cond_start[tr+1]]
                mat[j, tr,:] = firing_by_bin(v.astype(int), neur, bins, remove_empty = False)
        mat_by_cond.append(mat)
    return mat_by_cond

def compute_r_squared(y_observed, y_predicted):
    ss_res = np.sum((y_observed - y_predicted) ** 2)
    ss_tot = np.sum((y_observed - np.mean(y_observed)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared