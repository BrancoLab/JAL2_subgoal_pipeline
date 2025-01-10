import matplotlib.pyplot as plt
import numpy as np

from behave_analysis.utils.creating_directories import make_directory

###------------------------PLOTTING FUNCTIONS----------------------

def tuning_curve_by_condition(tuning, xval, nickname, vmax = 1.2, dump_path = "Z:/Jasmine_Laurence/homing/peak_firing_condition"):
    """A plot of tuning curves, by condition sorted for that condition and the sorting applied to the other two conditions
    INPUTS:
        tuning: is a list of len(conditions), each entry is a matrix of tuning curves of shape neurons x bins
        xval: vector of length neurons x conditions, indicating if the neuron's tuning passed xval
        nickname: a string of the session information (mouse name and date) and the behavioral variable
    """
    condy = ['shelter only', 'barrier','flipped barrier']
    fig, axs = plt.subplots(3,3,figsize = (9,9))
    fig.suptitle(nickname)

    for j, cc in enumerate(condy):
        t = tuning[j][xval[:,j] == 1,:]
        idx = np.argmax(t, axis = 1)
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
        peak_firing_condition: is a matrix of neurons x condition, where each entry is the bin with the peak firing for that neuron in that condition to the behavioral variable
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

def plot_tuning_matrix(tuning_matrix, cond, compression_var, escape_matrix, var, esc_start, h_start, xval, nickname, dump_path = "Z:/Jasmine_Laurence/homing/tuning"):
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
    """

    condy = ["shelter only", "barrier", "flipped barrier"]

    # set up the figure
    fig = plt.figure(figsize=(40, 22), dpi=200)
    grid = plt.GridSpec(26, 20, figure=fig, wspace=0.05, hspace=0.3)

    # create a vector of len(var) which is nan for all homing periods but has the behavioral data for the escape periods
    # this allows us to plot the behavior during escape sa different color
    esc_var = np.zeros_like(var)
    for it, st in enumerate(h_start):
        if st in esc_start:
            if it < len(h_start) - 1:
                esc_var[st : h_start[it + 1]] = var[st : h_start[it + 1]]
            else:
                esc_var[st : ] = var[st : ] # this is the case that the last h_start is an escape!
    esc_var[esc_var == 0] = np.nan

    # iterate over the three conditions, given by where we want their plots to be in the figure grid
    for i, lim in enumerate([[1, 8], [10, 17], [19, 26]]):
        
        # only show neurons that passed xval
        mat = tuning_matrix[i][xval[:,i] == 1,:]
        if len(mat) == 0:
            continue
        full_mat = escape_matrix[xval[:,i] == 1,:]
        
        # define the sorting index of neurons based on their peak firing in the tuning curve
        idx = np.argmax(mat, axis=1)
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

    dump_path = make_directory(dump_path)
    fig.savefig(dump_path + "/" + nickname + ".png")
    plt.close()

def tuning_curve_compare(tuning1, tuning2, xval1, name1, name2, nickname, dump_path = "Z:/Jasmine_Laurence/homing/peak_firing_condition"):
    """This function compares tuning curves for two sets of time (e.g. explore vs homing/escape, or homing vs escape, or homing in different time periods)
    
    INPUTS:
        tuning1, tuning2: are lists for the two sets of time of len(conditions), each entry is a matrix of tuning curves of shape neurons x bins
        xval1: matrix of length neurons x conditions, indicating if the neuron's tuning passed xval (this is assumed to be based on tuning1 curves)
        name1, name2: strings indicating the two sets of time
        nickname: string to name the figure for saving
        dump_path: where to save the figure
    """
    condy = ['shelter only', 'barrier','flipped barrier']
    fig, axs = plt.subplots(3,2,figsize = (9,9))
    fig.suptitle(nickname)

    for j, cc in enumerate(condy):
        t = tuning1[j][xval1[:,j] == 1,:]
        if len(t) == 0:
            continue
        idx = np.argmax(t, axis = 1)
        isort = np.argsort(idx)
        axs[j,0].set_ylabel('neurons sorted by ' + cc)
        axs[j,0].imshow(t[isort,:], cmap="gray_r", vmin = 0, vmax = 1.2, aspect="auto", interpolation = "none")
        t2 = tuning2[j][xval1[:,j] == 1,:]
        axs[j,1].imshow(t2[isort,:], cmap="gray_r", vmin = 0, vmax = 1.2, aspect="auto", interpolation = "none")

    axs[0,0].set_title(name1)
    axs[0,1].set_title(name2)

    plt.tight_layout()
    dump_path = make_directory(dump_path)
    fig.savefig(dump_path + "/" + nickname + ".png")
    plt.close()

def xval_compare(xval1, xval2, name1, name2, nickname, dump_path):
    """This function makes a bar plot to show how many neurons have xval tuning curves in two sets of time and in different conditions"""
    
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