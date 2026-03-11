import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

from JR_test_scripts.escape.functions.escape_tuning_funcs import fit_gaussian, fit_double_gaussian, gaussian_fitting
from JR_test_scripts.escape.functions.escape_utils import smooth_firing_by_bin_by_trial
from behave_analysis.utils.creating_directories import make_directory

###------------------------PLOTTING FUNCTIONS----------------------

def tuning_curve_by_condition(tuning, xval, nickname, peak_firing_condition = [], vmax = 1.2, dump_path = "Z:/Jasmine_Laurence/homing/peak_firing_condition"):
    """A plot of tuning curves, by condition sorted for that condition and the sorting applied to the other two conditions
    INPUTS:
        tuning: is a list of len(conditions), each entry is a matrix of tuning curves of shape neurons x bins
        xval: vector of length neurons x conditions, indicating if the neuron's tuning passed xval
        nickname: a string of the session information (mouse name and date) and the behavioral variable
        peak_firing_condition: is a matrix of neurons x condition, where each entry is the bin that neuron is tuned to in that condition to the behavioral variable
    """
    condy = ['shelter only', 'barrier','flipped barrier']
    fig, axs = plt.subplots(3,3,figsize = (9,9))
    fig.suptitle(nickname)

    for j, cc in enumerate(condy):
        if len(peak_firing_condition) == 0:
            t = tuning[j][xval[:,j] == 1,:]
            idx = np.argmax(t, axis = 1)
        else:
            xval_peak_firing = peak_firing_condition[xval[:,j] == 1,:]
            idx = xval_peak_firing[:,j]
        isort = np.argsort(idx)
        axs[j,0].set_ylabel('neurons sorted by ' + cc)
        for i, c in enumerate(condy):
            plot_t = tuning[i][xval[:,j] == 1,:]
            if len(plot_t) == 0:
                continue
            axs[j,i].imshow(plot_t[isort,:], cmap="gray_r", vmin = 0, vmax = vmax, aspect="auto", interpolation = "none")
            axs[j,i].set_title(c)

    plt.tight_layout()
    dump_path = make_directory(dump_path)
    fig.savefig(dump_path + "/" + nickname + ".png")
    plt.close()

def plot_pref_firing_condition(peak_firing_condition, xval, nickname, dump_path = "Z:/Jasmine_Laurence/homing/peak_firing_condition"):
    """Make a figure with 9 subplots, each with a scatter plot comparing the bin with the peak firing for each neuron in each condition
    It only shows neurons that passed the xval test
    
    INPUTS:
        peak_firing_condition: is a matrix of neurons x condition, where each entry is the bin that neuron is tuned to in that condition to the behavioral variable
        xval: vector of length neurons x conditions, indicating if the neuron's tuning passed xval 
        nickname: a string of the session information (mouse name and date) and the behavioral variable
    """
    
    c = ['shelter only', 'barrier','flipped barrier']
    fig, axs = plt.subplots(3,3,figsize = (9,9))
    fig.suptitle(nickname)

    axlim = [0,np.amax(peak_firing_condition)]

    for j, cc in enumerate(c):
        pfc = peak_firing_condition[xval[:,j] == 1,:]
        if len(pfc) == 0:
            continue
        for i, (x, y) in enumerate(zip([0,0,1],[1,2,2])):
            axs[j,i].scatter(pfc[:,x],pfc[:,y], s = 3)
            axs[j,i].plot(axlim,axlim,'--k')
            axs[j,i].set_xlabel(c[x])
            axs[j,i].set_ylabel(c[y])
            axs[j,i].set_xlim(axlim)
            axs[j,i].set_ylim(axlim)

    plt.tight_layout()
    dump_path = make_directory(dump_path)
    fig.savefig(dump_path + "/" + nickname + ".png")
    plt.close()

def plot_dist_pref_tuning_diff(peak_firing_condition, xval, nickname, dump_path = "Z:/Jasmine_Laurence/homing/peak_firing_condition"):
    """This function compares the pref tuning bin for each neuron in each condition and plots the distribution of the difference.
    
    INPUTS:
        peak_firing_condition: is a matrix of neurons x condition, where each entry is the bin that neuron is tuned to in that condition to the behavioral variable
        xval: vector of length neurons x conditions, indicating if the neuron's tuning passed xval 
        nickname: a string of the session information (mouse name and date) and the behavioral variable

    TODO: what is the null?
    """

    c = ['shelter only', 'barrier','flipped barrier']
    fig, axs = plt.subplots(1,3,figsize = (9,9))
    fig.suptitle(nickname)

    axlim = [- np.amax(peak_firing_condition),np.amax(peak_firing_condition)]

    xval_any = np.sum(xval, axis = 1) > 0
    pfc = peak_firing_condition[xval_any == True,:]

    if len(pfc) > 0:
        for i, (x, y) in enumerate(zip([0,0,1],[1,2,2])):
            axs[i].hist(pfc[:,x] - pfc[:,y])
            axs[i].set_xlabel('diff in pref tuning\n (' + c[x] + ' - ' + c[y] + ')')
            axs[i].set_ylabel('fraction of neurons')
            axs[i].set_xlim(axlim)
            ylim = axs[i].get_ylim()
            axs[i].plot([0,0],ylim,'--k')

    plt.tight_layout()
    dump_path = make_directory(dump_path)
    fig.savefig(dump_path + "/" + nickname + ".png")
    plt.close()

def plot_dist_pref_tuning_diff_compare(peak_firing_condition1, peak_firing_condition2, xval1, xval2, name1, name2, nickname, dump_path = "Z:/Jasmine_Laurence/homing/peak_firing_condition"):
    """This function compares the pref tuning bin for each neuron in each condition across two datasets and plots the distribution of the difference.
    
    INPUTS:
        peak_firing_condition1, peak_firing_condition2: are two matrices of neurons x condition, where each entry is the bin that neuron is tuned to in that condition to the behavioral variable
        xval1, xval2: vector of length neurons x conditions, indicating if the neuron's tuning passed xval 
        nickname: a string of the session information (mouse name and date) and the behavioral variable

    TODO: what is the null?
    """

    c = ['shelter only', 'barrier','flipped barrier']
    fig, axs = plt.subplots(2,3,figsize = (9,9))
    fig.suptitle(nickname)

    combined_pfc = [peak_firing_condition1, peak_firing_condition2]
    combined_xval = [xval1, xval2]

    for row in [0,1]:
        axlim = [- np.amax(combined_pfc[row]),np.amax(combined_pfc[row])]

        xval_any = np.sum(combined_xval[row], axis = 1) > 0
        pfc1 = peak_firing_condition1[xval_any == True,:]
        pfc2 = peak_firing_condition2[xval_any == True,:]

        if len(np.where(xval_any)[0]) > 0:
            for i in np.arange(np.shape(peak_firing_condition1)[1]):
                axs[row,i].hist(pfc1[:,i] - pfc2[:,i])
                axs[row,i].set_xlabel('diff in pref tuning\n (' + name1 + ' - ' + name2 + ')')
                axs[row,i].set_ylabel('fraction of neurons')
                axs[row,i].set_title(c[i])
                axs[row,i].set_xlim(axlim)
                ylim = axs[row,i].get_ylim()
                axs[row,i].plot([0,0],ylim,'--k')

    plt.tight_layout()
    dump_path = make_directory(dump_path)
    fig.savefig(dump_path + "/" + nickname + ".png")
    plt.close()

def plot_tuning_matrix(tuning_matrix, cond, compression_var, escape_matrix, var, esc_start, h_start, xval, nickname, peak_firing_condition = [], dump_path = "Z:/Jasmine_Laurence/homing/tuning", show = False):
    """This creates a figure with three rows of subplots, one for each condition.
    In each row the left plot is the tuning curve (neurons x bins) and the right plot is the complete neural data for that condition (neurons x time),
    with the start of escape periods in red and the start of homings in blue. 
    Both are sorted by the bin with peak firing for each neuron.
    Above the heatmap of complete neural data is a plot of the behavioral variable of interest

    NB: unless xval is all ones, it only shows neurons whose tuning curves passed xval 
    
    INPUTS:
        tuning_matrix: is a list of len(conditions), each entry is a matrix of tuning curves of shape neurons x bins
        cond: a vector of length time indicating what experimental condition the homing/escape was in
        compression_var: the name of the behavioral variable of interest, e.g. 'y_pos', 'distance_shelter', 'escape','speed'
        escape_matrix: a matrix of neural data, neurons x time
        var: the behavioral variable of interest, discretized into bins
        h_start: the start time of the homings or escapes, duration of homing/escape is cropped to when the mouse reaches shelter
        esc_start: the start time of the escapes only, duration of homing/escape is cropped to when the mouse reaches shelter
        xval: vector of length neurons x conditions, indicating if the neuron's tuning passed xval 
        nickname: a string of the session information (mouse name and date) and the behavioral variable
        peak_firing_condition: is a matrix of neurons x condition, where each entry is the bin that neuron is tuned to in that condition to the behavioral variable
    """

    condy = ["shelter only", "barrier", "flipped barrier"]

    # set up the figure
    fig = plt.figure(figsize=(40, 22), dpi=200)
    grid = plt.GridSpec(26, 20, figure=fig, wspace=0.05, hspace=0.3)

    # create a vector of len(var) which is nan for all homing periods but has the behavioral data for the escape periods
    # this allows us to plot the behavior during escapes a different color
    esc_var = np.full_like(var, np.nan)
    for it, st in enumerate(h_start):
        if st in esc_start:
            if it < len(h_start) - 1:
                esc_var[st : h_start[it + 1]] = var[st : h_start[it + 1]]
            else:
                esc_var[st : ] = var[st : ] # this is the case that the last h_start is an escape!

    # iterate over the three conditions, given by where we want their plots to be in the figure grid
    for i, lim in enumerate([[1, 8], [10, 17], [19, 26]]):
        
        # only show neurons that passed xval
        mat = tuning_matrix[i][xval[:,i] == 1,:]
        if len(mat) == 0:
            continue
        full_mat = escape_matrix[xval[:,i] == 1,:]
        
        # define the sorting index of neurons based on their peak firing in the tuning curve
        if len(peak_firing_condition) == 0:
            idx = np.argmax(mat, axis=1)
        else:
            xval_peak_firing = peak_firing_condition[xval[:,i] == 1,:]
            idx = xval_peak_firing[:,i]
        isort = np.argsort(idx)
        
        # tuning curve, by condition
        ax = plt.subplot(grid[lim[0] : lim[1], :2])
        ax.imshow(
            mat[isort, :],
            cmap="gray_r",
            vmin=0,
            vmax=1.2,
            aspect="auto",
            interpolation="none",
        )
        ax.set_ylabel("neurons")
        ax.set_ylim([0, len(isort)])
        ax.set_xlabel(compression_var)
        ax.set_title("tuning curve")
        ax.set_ylabel("sorted by tuning in " + condy[i])

        # behavioral var
        ax = plt.subplot(grid[lim[0] - 1, 3:])
        ax.plot(var[cond == i], "b")
        ax.plot(esc_var[cond == i], "r")
        ax.set_xlim([0, len(var[cond == i])])
        ax.axis("off")
        ax.set_ylabel(compression_var)

        # neural data
        ax = plt.subplot(grid[lim[0] : lim[1], 3:])
        e = full_mat[:, cond == i]
        ax.imshow(
            e[isort, :],
            cmap="gray_r",
            vmin=0,
            vmax=1.2,
            aspect="auto",
            interpolation="none",
        )
        
        # add red and blue dashed lines to indicate the start of escape periods and homings
        for st in h_start:
            if cond[st] == i:
                st_adjusted = st - np.where(cond == i)[0][0]
                if st in esc_start:
                    ax.plot([st_adjusted, st_adjusted], [len(isort), 0], "--r", linewidth=0.7)
                else:
                    ax.plot([st_adjusted, st_adjusted], [len(isort), 0], "--b", linewidth=0.7)
        ax.set_xlabel("time")
        ax.set_yticks([])
        ax.set_ylim([0, len(isort)])
        ax.set_xlim([0, len(var[cond == i])])

    if show:
        plt.show()
    else:
        dump_path = make_directory(dump_path)
        fig.savefig(dump_path + "/" + nickname + ".png")
        plt.close()

def tuning_curve_compare(tuning, xval1, name, nickname, peak_firing_condition = [], dump_path = "Z:/Jasmine_Laurence/homing/peak_firing_condition"):
    """This function compares tuning curves for two sets of time (e.g. explore vs homing/escape, or homing vs escape, or homing in different time periods)
    
    INPUTS:
        tuning: is a list of length n (sets to compare), each list is a list of len(conditions), each entry is a matrix of tuning curves of shape neurons x bins
        xval1: matrix of length neurons x conditions, indicating if the neuron's tuning passed xval (this is assumed to be based on tuning[0] curves)
        name: is a list of strings indicating the n sets of time
        peak_firing_condition: is a matrix of neurons x condition, where each entry is the bin that neuron is tuned to in that condition to the behavioral variable, if given it is based on tuning[0]
        nickname: string to name the figure for saving
        dump_path: where to save the figure
    """
    condy = ['shelter only', 'barrier','flipped barrier']
    fig, axs = plt.subplots(3,len(tuning),figsize = (9,9))
    fig.suptitle(nickname)

    for j, cc in enumerate(condy):
        for i in np.arange(len(tuning)):
            # reference tuning curves used for sorting
            if len(peak_firing_condition) == 0:
                t = tuning[0][j][xval1[:,j] == 1,:]
                if len(t) == 0:
                    continue
                idx = np.argmax(t, axis = 1)
            else:
                xval_peak_firing = peak_firing_condition[xval1[:,j] == 1,:]
                idx = xval_peak_firing[:,j]
            isort = np.argsort(idx)
            # actually picking out the xval neurons from the tuning curves we're plotting
            t = tuning[i][j][xval1[:,j] == 1,:]
            axs[j,i].set_ylabel('neurons sorted by ' + cc)
            axs[j,i].imshow(t[isort,:], cmap="gray_r", vmin = 0, vmax = 1.2, aspect="auto", interpolation = "none")

    for i in np.arange(len(tuning)):
        axs[0,i].set_title(name[i])

    plt.tight_layout()
    dump_path = make_directory(dump_path)
    fig.savefig(dump_path + "/" + nickname + ".png")
    plt.close()

def tuning_curve_sorter(sorter, sortee, xval_sorter, name_sorter, name_sorted, nickname, peak_firing_condition_sorter = [], dump_path = "Z:/Jasmine_Laurence/homing/peak_firing_condition"):
    """This function compares tuning curves for two sets of time (e.g. explore vs homing/escape, or homing vs escape, or homing in different time periods)
    
    INPUTS:
        sorter: is a list of matrices of tuning curves of shape neurons x bins
        sortee: is a list of matrices of tuning curves of shape neurons x bins to which the sorting will be applied
        xval: is a list (!) of boolean vectors of length neurons, indicating if the neuron's tuning passed xval (this is assumed to be based on sorter curves)
        name_sorter: is a list of strings indicating the name of the sorter
        name_sorted: is a list of strings indicating the name of the sorted matrices
        peak_firing_condition: is a list(!) of vectors of neurons, where each entry is the bin that neuron is tuned to in that condition to the behavioral variable, if given it is based on tuning[0]
        nickname: string to name the figure for saving
        dump_path: where to save the figure

    RETURNS:
        a figure with len(sorter) rows and len(sorteee)+1 columns of subplots 
        in each row the first plot is the heatmap of sorter, sorted based on peak_firing_condition (if that is not passed the max of each neuron is used)
        the next plots are the sortee matrices, all sorted the same way
    """
    fig, axs = plt.subplots(len(sorter),len(sortee)+1,figsize = ((len(sortee)+1)*5,(len(sorter)+1)*5))
    if len(sorter) == 1:
        axs = np.atleast_2d(axs)
    fig.suptitle(nickname)

    for j in np.arange(len(sorter)):
        # reference tuning curves used for sorting
        t = sorter[j][xval_sorter[j],:]
        if len(peak_firing_condition_sorter) == 0:
            if len(t) == 0:
                continue
            idx = np.argmax(t, axis = 1)
        else:
            idx = peak_firing_condition_sorter[j][xval_sorter[j]]
        isort = np.argsort(idx)
        axs[j,0].set_ylabel('neurons sorted by ' + name_sorter[j])
        axs[j,0].set_title(name_sorter[j])
        axs[j,0].imshow(t[isort,:], cmap="gray_r", vmin = 0, vmax = 1.2, aspect="auto", interpolation = "none")
        for i in np.arange(len(sortee)):
            # actually picking out the xval neurons from the tuning curves we're plotting
            t = sortee[i][xval_sorter[j],:]
            axs[j,i+1].set_title(name_sorted[i])
            axs[j,i+1].imshow(t[isort,:], cmap="gray_r", vmin = 0, vmax = 1.2, aspect="auto", interpolation = "none")

    plt.tight_layout()
    dump_path = make_directory(dump_path)
    fig.savefig(dump_path + "/" + nickname + ".png")
    plt.close()

def pref_tuning_comparer_hist(reference, comparisons, xval_reference, name_reference, name_comparisons, nickname, dump_path = "Z:/Jasmine_Laurence/homing/peak_firing_condition"):
    """This function compares tuning curves for two sets of time (e.g. explore vs homing/escape, or homing vs escape, or homing in different time periods)
    
    INPUTS:
        reference: is a list of vectors of len(neurons), where each entry is the bin that neuron is tuned to in that condition to the behavioral variable, in eah subplot comparisons will be subtracted from each reference
        comparisons: is a list (len(reference)) of lists of of vectors of len(neurons), where each entry is the bin that neuron is tuned to in that condition to the behavioral variable
        xval_reference: is a list (!) of boolean vectors of len(neurons), indicating if the neuron's tuning passed xval (this is assumed to be based on sorter curves)
        name_reference: is a list of strings indicating the name of the sorter
        name_comparisons: is a list (len(reference)) of lists of strings indicating the name of the sorted matrices
        nickname: string to name the figure for saving
        dump_path: where to save the figure

    RETURNS:
        a figure with len(reference) subplots 
        in each subplot is a set of histograms of the difference in pref tuning between the reference and the comparisons using peak_firing_condition (if that is not passed the max of each neuron is used)
    """
    fig, axs = plt.subplots(1, len(reference),figsize = ((len(reference))*5,5))
    fig.suptitle(nickname)

    # find bin range

    for j in np.arange(len(reference)):

        axs[j].set_title('reference ' + name_reference[j])
        for i in np.arange(len(comparisons[j])):
            axs[j].hist(reference[j][xval_reference[j]] - comparisons[j][i][xval_reference[j]], alpha = .5, label = name_comparisons[j][i])
        axs[j].legend()

    plt.tight_layout()
    dump_path = make_directory(dump_path)
    fig.savefig(dump_path + "/" + nickname + ".png")
    plt.close()

def xval_compare(xval1, xval2, name1, name2, nickname, dump_path):
    """This function makes a bar plot to show how many neurons have xval'd tuning curves in two sets of time (e.g. homing/escape vs explore) and in different conditions (e.g. 'shelter_only', 'barrier','flipped_barrier')
    It produces a figure with n subplots one for each condition. Each subplot has 4 bars showing how many neurons passed xval in neither set of time, in only one of the two sets of time or in both.
    """
    
    condy = ['shelter only', 'barrier','flipped barrier']
    fig, axs = plt.subplots(1,3,figsize = (12,4))
    fig.suptitle(nickname)

    axs[0].set_ylabel('number of neurons')

    for j, cc in enumerate(condy):
        axs[j].bar([0],np.sum(np.logical_and(xval1[:,j] == 0, xval2[:,j] == 0)), label = 'neither')
        axs[j].bar([1],np.sum(np.logical_and(xval1[:,j] == 1, xval2[:,j] == 0)), label = name1 + ' only')
        axs[j].bar([2],np.sum(np.logical_and(xval1[:,j] == 0, xval2[:,j] == 1)), label = name2 + ' only')
        axs[j].bar([3],np.sum(np.logical_and(xval1[:,j] == 1, xval2[:,j] == 1)), label = name1 + '\n and ' + name2)
        axs[j].set_title(cc)
        axs[j].set_xticks([])
    axs[j].legend()

    plt.tight_layout()
    dump_path = make_directory(dump_path)
    fig.savefig(dump_path + "/" + nickname + ".png")
    plt.close()

def plot_gaussian_fit_tuning(tuning, xval, dump_path, mat_by_cond, comp, xval_true = True, verbose = False, method = 'across_trials'):
    """This models the tuning of each neuron in each condition using either a single or double gaussian and returns the parameters of the best fit.
    It makes a plot with in one subplot the average firing by bin and the fitted firing by bin. 
    The other subplot has the activity per bin on each trial (if you have trial based activity like in homing/escape)

    INPUTS:
        tuning: is a list of len(conditions), each entry is a matrix of tuning curves of shape neurons x bins
        xval: matrix of length neurons x conditions, indicating if the neuron's tuning passed xval
        dump_path: where to save the figures (usually a folder named after the session + the behavioral vairable being studied)
        mat_by_cond: is a list of len(conditions), each entry is a matrix of firing rates neurons x trials x bins 
        comp: a string of the name of the behavioral variable of interest (e.g. 'speed', 'escape', 'distance_shelter')
        xval_true: it will only do the fitting for neurons that have passed xval
    RETURNS:
        y_fitted_all: is a list of len(conditions), each entry is a matrix (neurons x bins) of the predicted firing rate by bin based on the fit
        R_all: matrix of length neurons x conditions of the R squared of the fit
        param_all: matrix (params x neurons x conditions) for single gauss [amplitude, mu, sigma] for double gaussian [A1, mu1, sigma1, A2, mu2, sigma2]
        shift_constant_all: matrix of length neurons x conditions of how much did we shift the firing rates to ensure they were all positive before fitting. This will have to be subtracted from the amplitude param!
        double_wins_all: boolean matrix of length neurons x conditions of whether the double gaussian was the better fit
    """

    colors = ['#228B22','#FF8C00','#008B8B']
    custom_cmap = mcolors.ListedColormap(colors)
    c_names = ['shelter_only', 'barrier', 'barrier_flipped']

    # find the cells that have xval'd tuning curves in at least one condition
    xval_any = np.ones(len(xval))
    xval_id = np.where(xval_any == xval_true)[0] # we're going to use this to name the figure so that we can find neurons across conditions

    # initialize variables for output
    y_fitted_all = np.zeros((len(tuning), tuning[0].shape[0], tuning[0].shape[1])) # conditions x neurons x n_bins
    R_all = np.zeros((len(xval_id), np.shape(xval)[1])) # neurons x conditions
    param_all = np.zeros((6, len(xval_id), np.shape(xval)[1])) # params x neurons x conditions
    shift_constant_all = np.zeros((len(xval_id), np.shape(xval)[1])) # neurons x conditions
    double_wins_all = np.zeros((len(xval_id), np.shape(xval)[1])).astype(bool) # neurons x conditions
    total_trials = np.sum([np.shape(a)[1] for a in mat_by_cond])

    for neuron in np.arange(len(xval_id)):
        matrix_this_neuron = np.full((total_trials, tuning[0].shape[1]), np.nan)
        trials_by_cond = np.zeros(total_trials)
        c = 0
        # initialize plot for this neuron
        if len(mat_by_cond) > 0:
            fig = plt.figure(figsize=(12, 8))
            gs = gridspec.GridSpec(2, 3, width_ratios=[10, 10, .5], height_ratios=[1, 1])
            # Add plots
            ax1 = fig.add_subplot(gs[0, 0])  # Top-left
            ax2 = fig.add_subplot(gs[1, 0])  # Bottom left
            ax3 = fig.add_subplot(gs[:, 1])  # right column
            ax3b = fig.add_subplot(gs[:, 2])  # Spanning both rows
        else:
            fig = plt.figure(figsize=(4, 4))
            gs = gridspec.GridSpec(1, 1, figure=fig)  # 2 rows, 2 columns
            # Add plots
            ax1 = fig.add_subplot(gs[0, 0])  # Top-left

        for condition in np.arange(len(tuning)):

            if len(mat_by_cond) > 0:
                # process the firing by bin for each trial for this neuron in this 
                xval_mat = mat_by_cond[condition][xval_any == xval_true,:,:]
                matrix = xval_mat[neuron,:,:]
            all_time = tuning[condition][xval_any == xval_true,:]
            # obtain firing rates by taking the median in each bin for all time
            flatline = np.where(np.diff((np.mean(all_time, axis = 0) == 0).astype(int)) == 1)[0]
            if len(flatline) > 0:
                all_time = all_time[:,:flatline[0]]
            fr_all_time  = all_time[neuron,:]

            distances, smoothed_firing_rates, shift_constant, smooth_test = smooth_firing_by_bin_by_trial(matrix, fr_all_time, method)

            R, y, params, double_wins = gaussian_fitting(smoothed_firing_rates[~np.isnan(smoothed_firing_rates)], np.arange(np.sum(~np.isnan(smoothed_firing_rates))), verbose = verbose)
            y_fitted = np.full_like(smoothed_firing_rates, np.nan)
            y_fitted[~np.isnan(smoothed_firing_rates)] = y

            # dump parameters into the variables to return
            R_all[neuron, condition] = R
            shift_constant_all[neuron, condition] = shift_constant
            double_wins_all[neuron, condition] = double_wins
            param_all[:len(params),neuron, condition] = params
            y_fitted_all[condition, neuron, :len(y_fitted)] = y_fitted

            # Plot data and gaussian fit
            # top left plot: original data and gaussian fit for each condition (diff colour for each)
            ax1.plot(distances, smoothed_firing_rates - shift_constant, label="Smoothed Data " + c_names[condition], linestyle = '--', color=colors[condition])
            ax1.set_ylabel('z-scored firing rates')
            if double_wins:
                ax1.plot(distances, y_fitted - shift_constant, label=f"Double Gaussian R^2 = {R:.2f}", linestyle = '-', color=colors[condition])
            else:
                ax1.plot(distances, y_fitted - shift_constant, label=f"Gaussian R^2 = {R:.2f}", linestyle = '-', color=colors[condition])
            
            if len(mat_by_cond) > 0:
                # botom left plot: plot of activity on each trial (diff colour for each condition)
                ax2.plot(smooth_test.T, color=colors[condition], linewidth = .5, alpha = .5)
                ax2.plot(smoothed_firing_rates - shift_constant, color=colors[condition], linewidth = 2)
                ax2.set_xlabel(comp)
                ax2.set_ylabel('z-scored firing rates')
                matrix_this_neuron[c:c+smooth_test.shape[0],:] = smooth_test
                trials_by_cond[c:c+smooth_test.shape[0]] = np.ones(np.shape(smooth_test)[0])*condition
                c += smooth_test.shape[0]

        # right plot: heatmap of firing rate by bin on each trial, side bar indicating conditions
        if len(mat_by_cond) > 0:
            all_nan_rows = np.all(np.isnan(matrix_this_neuron), axis=1)
            trials_by_cond = trials_by_cond[~all_nan_rows]
            ax3.imshow(matrix_this_neuron[~all_nan_rows,:], cmap="gray_r", vmin = -1, vmax = 1, aspect="auto", interpolation = "none")
            ax3.set_ylabel("Trials")
            ax3.set_xlabel(comp)
            ax3.set_title('Median activity per trial')
            ax3b.imshow(trials_by_cond[:, np.newaxis], cmap=custom_cmap, aspect="auto")
            ax3b.set_xticks([])  # Hide x-axis ticks
            ax3b.set_yticks([])  # Hide y-axis ticks
            # Fine-tune the indicator's position to move it closer to the main heatmap
            pos_main = ax3.get_position()  # Get position of the main heatmap
            pos_indicator = ax3b.get_position()  # Get position of the indicator heatmap

            # Adjust the indicator's position
            new_x0 = pos_main.x1 + 0  # Position slightly to the right of the main heatmap
            new_x1 = new_x0 + (pos_indicator.x1 - pos_indicator.x0)  # Maintain the same width
            ax3b.set_position([new_x0, pos_indicator.y0, new_x1 - new_x0, pos_indicator.y1 - pos_indicator.y0])

            ax2.set_xlim((0, np.shape(tuning[condition])[1]))

        ax1.set_xlim((0, np.shape(tuning[condition])[1]))
        if len(mat_by_cond) == 0:
            ax1.set_xlabel(comp)
        ax1.legend(fontsize="small", markerscale=0.5)

        if xval_true:
            fig.savefig(dump_path + "/xval_neuron" + str(xval_id[neuron]) + "_allcond.png")
        else:
            fig.savefig(dump_path + "/neuron" + str(xval_id[neuron]) + "_allcond.png")
        plt.close()

    return y_fitted_all, R_all, param_all, shift_constant_all, double_wins_all

def plot_reliability(mat_full_cond, fr_full, full_reliability, comp, colors, c_names, n_cond, n_neur, dump_path):   
    """Plotting reliability!
    Makes a figure for each neuron showing the trial by trial response for each condition, with the average firing rate overlaid."""

    # compute min/max for this neuron across all conditions
    vmin, vmax = np.nanmin(mat_full_cond, axis=(0, 2, 3)), np.nanmax(mat_full_cond, axis=(0, 2, 3))  # Ignore NaNs

    # compute min/max for the average also
    ymin, ymax = [np.nanmin(fr_full, axis=(0, 2)), np.nanmax(fr_full, axis=(0, 2))]

    for neur in range(n_neur):
        fig, axs = plt.subplots(1,3, figsize = (12,4), constrained_layout=True)

        ylim = [ymin[neur], ymax[neur]]
        if ylim == [0,0]: ylim = [0,1]

        for c in range(n_cond):
            nan_rows = np.all(np.isnan(mat_full_cond[c,neur,:,:]), axis=1)
            im = axs[c].imshow(mat_full_cond[c,neur,~nan_rows,:], cmap="gray_r", vmin = vmin[neur], vmax = vmax[neur], aspect="auto", interpolation = "none")
            axs[c].set_title(c_names[c] + f'\n Reliability = {full_reliability[c,neur]:.2f}')
            axs[c]. set_xlabel(comp)
            if c == 0:
                axs[c].set_ylabel('trials')

            ax2 = axs[c].twinx()
            ax2.plot(fr_full[c,neur,:], linewidth = 2, color = colors[c])
            ax2.spines["right"].set_color(colors[c])
            ax2.tick_params(axis="y", colors=colors[c])  # Change tick color
            ax2.yaxis.label.set_color(colors[c])  # Change axis label color
            ax2.set_ylim(ylim)
            
            if c == 2:
                ax2.set_ylabel('Firing rate')
                cbar = fig.colorbar(im, ax=axs[c], location="right", pad=0.1)
                cbar.set_label('Firing rate')

            fig.savefig(dump_path + "/neuron" + str(neur) + "_loo_reliability.png")
            plt.close()

def plot_linear_shift(y_fitted_shift, y_fitted_real, params_shifts, params_real, R_shift, R_real, comp, n_neur, n_cond, colors, c_names, dump_path = [], name = []):
    """Plot linear shift and real stats"""

    # compute min/max for the average also
    nancells = (np.sum(np.isnan(y_fitted_shift), axis = (0,1,3)) == (y_fitted_shift.shape[0]*y_fitted_shift.shape[1]*y_fitted_shift.shape[3])) | (np.sum(np.isnan(y_fitted_real), axis = (0,2)) == (y_fitted_real.shape[0]*y_fitted_real.shape[2]))
    ymin = np.full(y_fitted_shift.shape[2], np.nan)
    ymax = np.full(y_fitted_shift.shape[2], np.nan)
    ymin[~nancells] = np.nanmin((np.nanmin(y_fitted_shift[:,:,~nancells,:], axis = (0,1,3)), np.nanmin(y_fitted_real[:,~nancells,:], axis = (0,2))), axis = 0)
    ymax[~nancells] = np.nanmax((np.nanmax(y_fitted_shift[:,:,~nancells,:], axis = (0,1,3)), np.nanmax(y_fitted_real[:,~nancells,:], axis = (0,2))), axis = 0)

    for neuron in range(n_neur):
        fig, axs = plt.subplots(3,3,figsize = (12,12))
        
        ylim = [ymin[neuron], ymax[neuron]]
        if ylim == [0,0]: ylim = [0,1]
        if np.sum(np.isnan(ylim)) == 2:
            continue

        for c in range(n_cond):
            # fit in real vs shifted trials
            axs[c,0].plot(y_fitted_shift[:,c,neuron,:].T,'k', alpha = .3)
            axs[c,0].plot(y_fitted_real[c,neuron,:],color = colors[c], linewidth = 3)
            # axs[c,0].plot(fr_shift[:,c,neuron,:].T,'k', alpha = .3)
            # axs[c,0].plot(fr_real[c,neuron,:],color = colors[c], linewidth = 3)
            axs[c,0].set_title(c_names[c])
            axs[c,0].set_ylabel('Firing rate')
            axs[c,0].set_ylim(ylim)
            axs[c,0].set_xlabel(comp)

            # fit amplitude
            axs[c,1].hist(params_shifts[:,neuron,c,0], edgecolor = None, facecolor = 'k', alpha = .3)
            yl = axs[c,1].get_ylim()
            axs[c,1].plot([params_real[neuron, c, 0],params_real[neuron, c, 0]],yl,color = colors[c])
            axs[c,1].set_xlabel('Firing rate')
            axs[c,1].set_xlim(ylim)
            axs[c,1].set_ylabel('Linear shifts')
            sig = params_real[neuron, c, 0] > np.percentile(params_shifts[:,neuron,c,0],95)
            axs[c,1].set_title('Fit amp. 95th perc sig: ' + str(sig))
            
            # goodness of fit
            axs[c,2].hist(R_shift[:,neuron,c], edgecolor = None, facecolor = 'k', alpha = .3)
            yl = axs[c,2].get_ylim()
            axs[c,2].plot([R_real[neuron, c],R_real[neuron, c]],yl,color = colors[c])
            if R_real[neuron,c] > np.percentile(R_shift[:,neuron,c],95):
                axs[c,2].set_title('Fit goodness 95th perc.: sig')
            else:
                axs[c,2].set_title('Fit goodness 95th perc.: not sig')
            axs[c,2].set_xlabel('R^2')
            axs[c,2].set_ylabel('Linear shifts')

        plt.tight_layout()
        if len(dump_path) == 0:
            plt.show()
        else:
            fig.savefig(dump_path + "/neuron" + str(neuron) + "_linshit" + name + ".png")
            plt.close()

def new_plot_linear_shift(y_fitted_shift, y_fitted_real, y_fitted_full, params_shifts, params_real, fr_full, fr_shift, params_full, comp, n_neur, n_cond, colors, c_names, dump_path = [], name = []):
    """Plot linear shift and real stats"""

    # compute min/max for the average also
    nancells = (np.sum(np.isnan(y_fitted_shift), axis = (0,1,3)) == (y_fitted_shift.shape[0]*y_fitted_shift.shape[1]*y_fitted_shift.shape[3])) | (np.sum(np.isnan(y_fitted_real), axis = (0,2)) == (y_fitted_real.shape[0]*y_fitted_real.shape[2]))
    ymin = np.full(y_fitted_shift.shape[2], np.nan)
    ymax = np.full(y_fitted_shift.shape[2], np.nan)
    ymin[~nancells] = np.nanmin((np.nanmin(y_fitted_shift[:,:,~nancells,:], axis = (0,1,3)), 
                                 np.nanmin(y_fitted_real[:,~nancells,:], axis = (0,2)),
                                 np.nanmin(y_fitted_full[:,~nancells,:], axis = (0,2)),
                                 np.nanmin(fr_full[:,~nancells,:], axis = (0,2))), axis = 0)
    ymax[~nancells] = np.nanmax((np.nanmax(y_fitted_shift[:,:,~nancells,:], axis = (0,1,3)), 
                                 np.nanmax(y_fitted_real[:,~nancells,:], axis = (0,2)),
                                 np.nanmax(y_fitted_full[:,~nancells,:], axis = (0,2)),
                                 np.nanmax(fr_full[:,~nancells,:], axis = (0,2))), axis = 0)

    for neuron in range(n_neur):
        fig, axs = plt.subplots(3,3,figsize = (12,12))
        
        ylim = [ymin[neuron], ymax[neuron]]
        if ylim == [0,0]: ylim = [0,1]
        if np.sum(np.isnan(ylim)) == 2:
            continue

        for c in range(n_cond):
            # fit in real vs shifted trials
            axs[c,0].plot(y_fitted_full[c,neuron,:], c = 'C1', label = 'full_smoothed')
            axs[c,0].plot(y_fitted_real[c,neuron,:], c = 'C0', label = 'real_smoothed')
            axs[c,0].plot(fr_full[c,neuron,:], c = 'k', label = 'full_firing_rate')
            axs[c,0].legend()
            axs[c,0].scatter(params_full[neuron,c,1], params_full[neuron,c,0], s=15, c = 'C1')
            axs[c,0].scatter(params_real[neuron,c,1], params_real[neuron,c,0], s=15, c = 'C0')
            axs[c,0].set_title(c_names[c])
            axs[c,0].set_ylabel('Firing rate')
            axs[c,0].set_ylim(ylim)
            axs[c,0].set_xlabel(comp)

            # plot the linear shifts
            axs[c,1].plot(fr_shift[1:,c,neuron,:].T, c = 'k',alpha = .25, linewidth = .5)
            axs[c,1].plot(fr_shift[0,c,neuron,:].T, c = 'k',alpha = .25, linewidth = .5, label = 'shifted_firing_rate')
            axs[c,1].plot(y_fitted_shift[1:,c,neuron,:].T, c = 'r',alpha = .25, linewidth = .5)
            axs[c,1].plot(y_fitted_shift[0,c,neuron,:].T, c = 'r',alpha = .25, linewidth = .5, label = 'shifted_smoothed')
            axs[c,1].scatter(params_shifts[:,neuron,c,1], params_shifts[:,neuron,c,0], s=5, c = 'r')
            axs[c,1].legend()
            axs[c,1].set_ylim(ylim)
            axs[c,1].set_xlabel(comp)

            # fit amplitude
            axs[c,2].hist(params_shifts[:,neuron,c,0], edgecolor = None, facecolor = 'k', alpha = .3)
            yl = axs[c,2].get_ylim()
            axs[c,2].plot([params_real[neuron, c, 0],params_real[neuron, c, 0]],yl,color = colors[c])
            axs[c,2].set_xlabel('Firing rate')
            axs[c,2].set_xlim(ylim)
            axs[c,2].set_ylabel('Linear shifts')
            sig = params_real[neuron, c, 0] > np.percentile(params_shifts[:,neuron,c,0],95)
            axs[c,2].set_title('Fit amp. 95th perc sig: ' + str(sig))

        plt.tight_layout()
        if len(dump_path) == 0:
            plt.show()
        else:
            fig.savefig(dump_path + "/neuron" + str(neuron) + "_linshit" + name + ".png")
            plt.close()

##-------------DEPRECATED----------

# def tuning_curve_compare(tuning1, tuning2, xval1, name1, name2, nickname, dump_path = "Z:/Jasmine_Laurence/homing/peak_firing_condition"):
#     """This function compares tuning curves for two sets of time (e.g. explore vs homing/escape, or homing vs escape, or homing in different time periods)
    
#     INPUTS:
#         tuning1, tuning2: are lists for the two sets of time of len(conditions), each entry is a matrix of tuning curves of shape neurons x bins
#         xval1: matrix of length neurons x conditions, indicating if the neuron's tuning passed xval (this is assumed to be based on tuning1 curves)
#         name1, name2: strings indicating the two sets of time
#         nickname: string to name the figure for saving
#         dump_path: where to save the figure
#     """
#     condy = ['shelter only', 'barrier','flipped barrier']
#     fig, axs = plt.subplots(3,2,figsize = (9,9))
#     fig.suptitle(nickname)

#     for j, cc in enumerate(condy):
#         t = tuning1[j][xval1[:,j] == 1,:]
#         if len(t) == 0:
#             continue
#         idx = np.argmax(t, axis = 1)
#         isort = np.argsort(idx)
#         axs[j,0].set_ylabel('neurons sorted by ' + cc)
#         axs[j,0].imshow(t[isort,:], cmap="gray_r", vmin = 0, vmax = 1.2, aspect="auto", interpolation = "none")
#         t2 = tuning2[j][xval1[:,j] == 1,:]
#         axs[j,1].imshow(t2[isort,:], cmap="gray_r", vmin = 0, vmax = 1.2, aspect="auto", interpolation = "none")

#     axs[0,0].set_title(name1)
#     axs[0,1].set_title(name2)

#     plt.tight_layout()
#     dump_path = make_directory(dump_path)
#     fig.savefig(dump_path + "/" + nickname + ".png")
#     plt.close()

# def plot_gaussian_fit_tuning(tuning, xval, dump_path, mat_by_cond, comp, xval_true = True, verbose = False):
#     """This models the tuning of each neuron in each condition using either a single or double gaussian and returns the parameters of the best fit.
#     It makes a plot with in one subplot the average firing by bin and the fitted firing by bin. 
#     The other subplot has the activity per bin on each trial (if you have trial based activity like in homing/escape)

#     INPUTS:
#         tuning: is a list of len(conditions), each entry is a matrix of tuning curves of shape neurons x bins
#         xval: matrix of length neurons x conditions, indicating if the neuron's tuning passed xval
#         dump_path: where to save the figures (usually a folder named after the session + the behavioral vairable being studied)
#         mat_by_cond: is a list of len(conditions), each entry is a matrix of firing rates neurons x trials x bins 
#         comp: a string of the name of the behavioral variable of interest (e.g. 'speed', 'escape', 'distance_shelter')
#         xval_true: it will only do the fitting for neurons that have passed xval
#     RETURNS:
#         y_fitted_all: is a list of len(conditions), each entry is a matrix (neurons x bins) of the predicted firing rate by bin based on the fit
#         R_all: matrix of length neurons x conditions of the R squared of the fit
#         param_all: matrix (params x neurons x conditions) for single gauss [amplitude, mu, sigma] for double gaussian [A1, mu1, sigma1, A2, mu2, sigma2]
#         shift_constant_all: matrix of length neurons x conditions of how much did we shift the firing rates to ensure they were all positive before fitting. This will have to be subtracted from the amplitude param!
#         double_wins_all: boolean matrix of length neurons x conditions of whether the double gaussian was the better fit
#     """

    
#     # find the cells that have xval'd tuning curves in at least one condition
#     xval_any = np.sum(xval, axis = 1) > 0
#     xval_id = np.where(xval_any == xval_true)[0] # we're going to use this to name the figure so that we can find neurons across conditions

#     # initialize variables for output
#     y_fitted_all = []
#     R_all = np.zeros((len(xval_id), np.shape(xval)[1])) # neurons x conditions
#     param_all = np.zeros((6, len(xval_id), np.shape(xval)[1])) # params x neurons x conditions
#     shift_constant_all = np.zeros((len(xval_id), np.shape(xval)[1])) # neurons x conditions
#     double_wins_all = np.zeros((len(xval_id), np.shape(xval)[1])).astype(bool) # neurons x conditions

#     for condition in np.arange(len(tuning)):
#         test = tuning[condition][xval_any == xval_true,:]
#         y_fit_this_condition = np.zeros_like(test)
#         for neuron in np.arange(test.shape[0]):
#             flatline = np.where(np.diff((np.mean(test, axis = 0) == 0).astype(int)) == 1)[0]
#             if len(flatline) > 0:
#                 test = test[:,:flatline[0]]
#             firing_rates  = test[neuron,:]
#             # from scipy.ndimage import gaussian_filter1d
#             sigma = 3.0  # Standard deviation of the Gaussian kernel
#             smoothed_firing_rates = gaussian_filter1d(firing_rates, sigma)

#             # make firing rates positive
#             shift_constant = abs(np.amin(smoothed_firing_rates))+ 1e-6 # Add a small epsilon to avoid exact zero
#             smoothed_firing_rates = smoothed_firing_rates + shift_constant
#             distances = np.arange(len(smoothed_firing_rates))

#             R, y_fitted, params, double_wins = gaussian_fitting(smoothed_firing_rates, distances, verbose = False)

#             # Plot original data and fitted Gaussian
#             fig, axs = plt.subplots(1,2,figsize = (12,4))
#             axs[0].scatter(distances, firing_rates - shift_constant, label="Original Data", s=3, color="blue")
#             axs[0].plot(distances, y_fitted - shift_constant, label="Fitted Gaussian", color="red")
#             if double_wins:
#                 axs[0].scatter([params[1], params[4]],[params[0] - shift_constant, params[3] - shift_constant], label = "Gaussian mu", s = 10, color = "green")
#             else:
#                 axs[0].scatter(params[1],params[0] - shift_constant, label = "Gaussian mu", s = 10, color = "green")
#             axs[0].set_ylabel("Firing Rate")
#             axs[0].set_xlabel(comp)
#             axs[0].set_xlim((0, np.shape(tuning[condition])[1]))
#             axs[0].legend()
#             if double_wins:
#                 A1 = params[0]
#                 A2 = params[3]
#                 # Prominence as relative amplitude
#                 A1_relative = A1 / (A1 + A2)
#                 A2_relative = A2 / (A1 + A2)
#                 axs[0].set_title((f"Double Gaussian R^2 = {R:.2f}\n" 
#                                 f"relative amplitude = {A1_relative:.2f}, {A2_relative:.2f}"))
#             else:
#                 axs[0].set_title(f"Gaussian R^2 = {R:.2f}")

#             if len(mat_by_cond) > 0:
#                 xval_mat = mat_by_cond[condition][xval_any == xval_true,:,:]
#                 axs[1].imshow(xval_mat[neuron,:,:], cmap="gray_r", vmin = 0, vmax = 1.2, aspect="auto", interpolation = "none")
#                 axs[1].set_ylabel("Trials")
#             else:
#                 fig.delaxes(axs[1])

#             c = ['shelter_only', 'barrier', 'barrier_flipped']
#             if xval_true:
#                 fig.savefig(dump_path + "/xval_neuron" + str(xval_id[neuron]) + '_' + c[condition] + ".png")
#             else:
#                 fig.savefig(dump_path + "/neuron" + str(xval_id[neuron]) + '_' + c[condition] + ".png")
#             plt.close()

#             # dump parameters into the variables to return
#             R_all[neuron, condition] = R
#             shift_constant_all[neuron, condition] = shift_constant
#             double_wins_all[neuron, condition] = double_wins
#             param_all[:len(params),neuron, condition] = params
#             y_fit_this_condition[neuron, :len(y_fitted)] = y_fitted

#         y_fitted_all.append(y_fit_this_condition)

#     return y_fitted_all, R_all, param_all, shift_constant_all, double_wins_all