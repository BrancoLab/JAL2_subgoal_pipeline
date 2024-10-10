"""All the functions for plotting LDA results"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Wedge
import seaborn as sns
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale
import numpy as np
import re
import pickle
from loguru import logger
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from behave_analysis.analyze.LDA.LDA_utils import BuildSavingFolder
# from behave_analysis.analyze.behaviour.spatial_efficiency import base_plotting
from behave_analysis.analyze.behaviour.utils import base_plotting
from behave_analysis.utils.arena_plotting import Arena
from behave_analysis.utils.heatplot_utils import add_features, add_features_binned

## --------------- PLOTTING FUNCTIONS


def plot_LDA_model(self):
    """A function to call all plotting functions"""
    # make a plot of prediction accuracy across variables
    with open(self.LDA_out, "rb") as dill_file:
        prediction_accuracy = pickle.load(dill_file)
    title = [key for key, val in prediction_accuracy.items() if not re.search("time", key)]

    PlotPredictionAccuracy(self, prediction_accuracy, title)

    # map random points on arena:
    if len(list(filter(lambda x: "randP" in x, prediction_accuracy.keys()))) > 10:
        pa = [val for key, val in prediction_accuracy.items() if re.search("randP", key)]
        fr = [val / (self.session.video.fps * 60) for key, val in prediction_accuracy.items() if re.search("time_rP", key)]
        PredictionAccuracyMapped(self, pa, fr=fr)

    # make a plot of prediction accuracy across variables with linear shift stats
    if self.settings.linear_shift:
        with open(self.LS_out, "rb") as dill_file:
            LS_compiled = pickle.load(dill_file)
        PlotLSPredictionAccuracy(self, LS_compiled, title)

        # map random points on arena:
        if len(list(filter(lambda x: "randP" in x, LS_compiled.keys()))) > 10:
            LS_mean = [np.mean(val.pseudo_stats) for key, val in LS_compiled.items() if re.search("randP", key)]
            PredictionAccuracyMapped(self, LS_mean, "LS")

            # now figure out which ones are significant
            n_randP = np.shape(self.tracking_data["randP_loc"])[0]
            alpha = 5 / 2  # two-sided .05 significance thresh
            alpha_perc = 100 - (alpha / n_randP)  # bonferroni corrected percentile
            LS_thresh = [np.percentile(val.pseudo_stats, alpha_perc) for key, val in LS_compiled.items() if re.search("randP", key)]
            LS_real = [val.real_stat for key, val in LS_compiled.items() if re.search("randP", key)]
            PredictionAccuracyMapped(self, LS_real, "LS_sig", LS_thresh)


def plot_LDA_by_position(self, target):
    with open(self.LDA_out, "rb") as dill_file:
        prediction_accuracy = pickle.load(dill_file)

    for var in target:
        pa = [val for key, val in prediction_accuracy.items() if re.search(var + "_pos", key)]
        fr = [round(val / (self.session.video.fps * 60), 2) for key, val in prediction_accuracy.items() if re.search(var + "_time", key)]
        PredictionAccuracy_byposition_Mapped(
            self,
            pa,
            fr=fr,
            title_add=(var + "_by_pos"),
            numrings=prediction_accuracy["num_circles"],
            num_slices=prediction_accuracy["num_slices"],
            bin_centre=prediction_accuracy["bin_centre"],
        )


def PlotPredictionAccuracy(self, prediction_accuracy, title):
    """
    Function to make a bar plot of the mean prediction accuracy for each angle
    """
    fig = go.Figure()

    if len(list(filter(lambda x: "randP" in x, title))) < 10:
        colorz = sample_colorscale("Rainbow", list(np.linspace(0, 1, len(title))))
    else:
        colorz = sample_colorscale("Rainbow", list(np.linspace(0, 1, len(list(filter(lambda x: "randP" not in x, title))) + 1)))

    for i, var in enumerate(list(filter(lambda x: "randP" not in x, title))):
        fig.add_trace(
            go.Bar(
                x=[var],
                y=[prediction_accuracy[var]],
                width=0.5,
                marker=dict(color=colorz[i], opacity=0.5),
                text=[f"{prediction_accuracy[var]:.2f}"],  # Format text to 2 decimal places
                textposition="outside",  # Position text at the top of the bar
                textfont=dict(size=14),
            )
        )  # Increase text size

    if len(list(filter(lambda x: "randP" in x, title))) < 10:
        for j, var in enumerate(list(filter(lambda x: "randP" in x, title))):
            fig.add_trace(go.Bar(x=[var], y=[prediction_accuracy[var]], width=0.5, marker=dict(color=colorz[i + j], opacity=0.5)))
    else:
        res = [val for key, val in prediction_accuracy.items() if re.search("randP", key)]
        var = "randP"
        fig.add_trace(go.Violin(x=[var] * len(res), y=res, points="all", jitter=0.05, marker=dict(size=3, color=colorz[i + 1])))

    fig.update_layout(showlegend=False)
    fig.update_yaxes(range=[0, 1.1])
    fig.update_yaxes(title_text="prediction accuracy")
    fig.update_xaxes(tickangle=-45)
    filename = str(self.savepath) + "/" + str(self.cluster_type) + "_" + str(self.condition) + "_LDA_pa" + ".png"
    fig.write_image(filename)


def PlotLSPredictionAccuracy(self, LS_compiled, title):
    """
    Function to make a violin plot of the mean prediction accuracy over all linear shifts
    """
    fig = go.Figure()
    if len(title) > 10:
        colorz = sample_colorscale("Rainbow", list(np.linspace(0, 1, 10)))
    else:
        colorz = sample_colorscale("Rainbow", list(np.linspace(0, 1, len(title))))

    for i, var in enumerate(title):
        if "randP" in var:  # don't plot this for random points, we-re going to map them
            break
        fig.add_trace(
            go.Violin(
                x=[var] * len(LS_compiled[var].pseudo_stats),
                y=LS_compiled[var].pseudo_stats,
                points="all",
                jitter=0.05,
                marker=dict(size=3, color=colorz[i]),
            )
        )
        fig.add_trace(
            go.Scatter(x=[var], y=[LS_compiled[var].real_stat], mode="markers", marker_color="rgb(255, 0, 0)", marker=dict(size=5, symbol="diamond"))
        )
        if LS_compiled[var].reject_null:
            fig.add_trace(go.Scatter(x=[var], y=[1], mode="markers", marker_color="rgb(0, 0, 0)", marker=dict(size=5, symbol="star")))

    fig.update_layout(showlegend=False)
    fig.update_yaxes(range=[0, 1.1])
    fig.update_yaxes(title_text="prediction accuracy")
    fig.update_xaxes(tickangle=-45)
    filename = str(self.savepath) + "/" + str(self.cluster_type) + "_" + str(self.condition) + "_LDA_LS_pa" + ".png"
    fig.write_image(filename)


def PredictionAccuracyMapped(self, pa, title_add="LDA", LS_thresh=None, pos=[], fr=[]):
    """
    Function to make a map of the prediction accuracy for the angle of the head to each point in the arena

    INPUT:
    pa = is a list of prediction accuracies for head-angle towards the points listed in self.tracking_data["randP_loc"]
    it can either be the prediction accuracy for the full datatset, or the mean of the shifted distribution geenrated with linear shift

    title_add = this is a string that will be added to the title of the figure when saving, it will help distinguish LDA full models from linear shift maps for example

    LS_thresh = an array of the same length as pa with the threshold for significance for each point, if passed a white dot will be added to each sqaure that is significant
    """

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(1, 1, 1)
    cbar_ax = fig.add_axes([0.91, 0.13, 0.01, 0.75])
    pa = np.array(pa)
    if len(pa[pa > 0]) > 0:
        vmin = np.amin(pa[pa > 0])
    else:
        vmin = 0
    vmax = np.amax(pa)

    # build heatmap
    if len(pos) == 0:
        pos = self.tracking_data["randP_loc"]
    ybins, y = np.unique(pos[:, 0], return_inverse=True)
    xbins, x = np.unique(pos[:, 1], return_inverse=True)
    heatmap = np.zeros(shape=(len(xbins), len(ybins)))
    heatmap[x, y] = pa
    heatmap_annot = np.zeros_like(heatmap)
    heatmap_annot[x, y] = fr

    # Plotting logic for the heatmap
    ax = sns.heatmap(
        heatmap,
        cmap="inferno",  # "coolwarm",
        cbar_ax=cbar_ax,
        robust=True,
        ax=ax,
        mask=(heatmap == 0),
        annot=heatmap_annot,
        cbar_kws={"label": "Prediction accuracy"},
        norm=plt.Normalize(vmin=vmin, vmax=vmax),
    )

    if LS_thresh != None:
        # significant points are the ones where predictiona ccuracy is greater than thresh
        significant_points = (pa - LS_thresh) > 0
        ax.scatter(x[significant_points] + 0.5, y[significant_points] + 0.5, s=3, c="w")

    add_features_binned(ax, self.condition, self.tracking_data, xbins, ybins)

    # Remove x and y tick labels and ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")
    ax.set_title(self.condition, fontsize=20)
    # The legend is the last axis so this is a hack to change the font size of the legend
    ax.figure.axes[-1].yaxis.label.set_size(16)
    ax.set_aspect("equal")

    # save plot
    filename = str(self.savepath) + "/" + str(self.cluster_type) + "_" + str(self.condition) + "_" + title_add + "_pa_map" + ".png"
    plt.savefig(filename)
    filename = str(self.savepath) + "/" + str(self.cluster_type) + "_" + str(self.condition) + "_" + title_add + "_pa_map" + ".eps"
    plt.savefig(filename, format="eps")
    if self.show_plots:
        plt.show()
    plt.close()


def PredictionAccuracy_byposition_Mapped(self, pa, numrings, num_slices, bin_centre, fr=[], title_add="LDA", LS_thresh=None):
    """
    Function to make a map of the prediction accuracy for the angle of the head to each point in the arena

    INPUT:
    pa = is a list of prediction accuracies for head-angle towards the points listed in self.tracking_data["randP_loc"]
    it can either be the prediction accuracy for the full datatset, or the mean of the shifted distribution geenrated with linear shift

    title_add = this is a string that will be added to the title of the figure when saving, it will help distinguish LDA full models from linear shift maps for example

    LS_thresh = an array of the same length as pa with the threshold for significance for each point, if passed a white dot will be added to each sqaure that is significant
    """

    # Define the radius of the circle
    radius = 460

    # set up figure
    _, ax = plt.subplots(subplot_kw={"aspect": "equal"})
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    pa = np.array(pa)
    if len(pa[pa > 0]) > 0:
        vmin = np.amin(pa[pa > 0])
    else:
        vmin = 0
    vmax = np.amax(pa)
    # Normalize the values to the range [0, 1]
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colormap = cm.inferno

    if np.shape(bin_centre)[1] > len(pa):
        bin_centre = bin_centre[:, 1:]

    # Calculate the radius of the inner ring
    radii = np.empty(numrings)
    for r in np.arange(numrings):
        radii[r] = np.sqrt(r + 1) * radius / np.sqrt(numrings)

    # Calculate the angles for the slices
    angles = np.linspace(-np.pi, np.pi, num_slices + 1)

    # Draw the segments
    for i in range(num_slices):
        theta1 = np.degrees(angles[i])
        theta2 = np.degrees(angles[i + 1])

        for idx, r in enumerate(np.flipud(radii)):
            which_pa = np.logical_and(bin_centre[0, :] == np.where(radii == r)[0], bin_centre[1, :] == i + 1)
            if pa[which_pa] > 0:
                color = colormap(norm(pa[which_pa]))
            else:
                color = [1, 1, 1]
            wedgie = Wedge((0, 0), r, theta1, theta2, facecolor=color)
            ax.add_patch(wedgie)

            if len(fr) > 0:
                # create text annotation in minutes of how much time was used for the LDA
                t = np.cos(np.mean([angles[i], angles[i + 1]])) * (r - np.mean(np.diff(radii)) / 2)
                r = (r - np.mean(np.diff(radii)) / 2) * (np.sin(np.mean([angles[i], angles[i + 1]])))
                ax.text(t, r, fr[np.where(which_pa)[0][0]], ha="center", va="center", color="w")

    if LS_thresh != None:
        logger.warning("You wanted to plot which bins are significant but this code isn't functional yet!")
        # significant points are the ones where predictiona ccuracy is greater than thresh
        # significant_points = (pa - LS_thresh) > 0
        # ax.scatter(x[significant_points]+.5,y[significant_points]+.5,s = 3, c = 'w')

    add_features(ax, self.condition, self.tracking_data, zero_centre=True)

    # Create a scalar mappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])  # Only needed for matplotlib < 3.1

    # Add colorbar to the plot
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("prediction accuracy")

    # Remove x and y tick labels and ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")
    ax.set_axis_off()
    ax.set_title(self.condition, fontsize=20)

    # save plot
    filename = str(self.savepath) + "/" + str(self.cluster_type) + "_" + str(self.condition) + "_" + title_add + "_pa_posmap" + ".png"
    plt.savefig(filename)
    filename = str(self.savepath) + "/" + str(self.cluster_type) + "_" + str(self.condition) + "_" + title_add + "_pa_posmap" + ".eps"
    plt.savefig(filename, format="eps")
    if self.show_plots:
        plt.show()
    plt.close()


def across_conditions_LDA_map(self):
    """
    Function to make a figure of the maps of the prediction accuracy for the angle of the head to each point in the arena
    all the maps for all the conditions are displayed together and color axes are adjusted to match across maps
    This also turns the map into heatmaps instead of scatterplots

    figure: each row is a compartment, each column is a condition
    """

    pa = []

    for comp in self.settings.compartment_split:
        for c in self.all_conditions:
            self.savepath = BuildSavingFolder(self.dir, self.settings, self.cluster_type, self.condition_types, c, comp)
            LDA_out = str(self.savepath) + "/" + str(self.cluster_type) + "_" + str(c) + "_LDA_pa" + ".pkl"
            with open(LDA_out, "rb") as dill_file:
                prediction_accuracy = pickle.load(dill_file)
            pa.append([val for key, val in prediction_accuracy.items() if re.search("randP", key)])

    pa = np.array(pa)
    if len(pa[pa > 0]) > 0:
        vmin = np.amin(pa[pa > 0])
    else:
        vmin = 0
    vmax = np.amax(pa)

    # figure set-up
    fig, axs = plt.subplots(
        nrows=len(self.settings.compartment_split),
        ncols=len(self.all_conditions) + 1,
        figsize=(24, 6 * len(self.settings.compartment_split)),
        sharey=True,
        sharex=True,
    )

    # Where to plot the colorbar, create new axis object at these coordinates
    cbar_ax = fig.add_axes([0.91, 0.13, 0.01, 0.75])  # The list represents [left, bottom, width, height],

    # Add subtitles for each condition in first column

    for c_counter, c in enumerate(self.settings.compartment_split):
        if len(self.settings.compartment_split) > 1:
            ax_idx = tuple([c_counter, 0])
        else:
            ax_idx = 0
        axs[ax_idx].text(0, 0.5, c, rotation="horizontal", va="center", ha="center", fontsize=23)
        axs[ax_idx].set_axis_off()

    data_counter = 0
    for c_idx, comp in enumerate(self.settings.compartment_split):
        for idx, condition in enumerate(self.all_conditions):
            if len(self.settings.compartment_split) > 1:
                ax_idx = tuple([c_idx, idx + 1])
            else:
                ax_idx = idx + 1
            # build heatmap
            ybins, y = np.unique(self.tracking_data["randP_loc"][:, 0], return_inverse=True)
            xbins, x = np.unique(self.tracking_data["randP_loc"][:, 1], return_inverse=True)
            heatmap = np.zeros(shape=(len(np.unique(self.tracking_data["randP_loc"][:, 0])), len(np.unique(self.tracking_data["randP_loc"][:, 1]))))
            heatmap[x, y] = pa[data_counter]
            data_counter = data_counter + 1

            # Plotting logic for the heatmap
            axs[ax_idx] = sns.heatmap(
                heatmap,
                cmap="inferno",  # "coolwarm",
                cbar_ax=cbar_ax,
                robust=True,
                ax=axs[ax_idx],
                mask=(heatmap == 0),
                cbar_kws={"label": "Prediction accuracy"},
                norm=plt.Normalize(vmin=vmin, vmax=vmax),
            )

            add_features_binned(axs[ax_idx], condition, self.tracking_data, xbins, ybins)

            # Remove x and y tick labels and ticks
            axs[ax_idx].set_xticklabels([])
            axs[ax_idx].set_yticklabels([])
            axs[ax_idx].xaxis.set_ticks_position("none")
            axs[ax_idx].yaxis.set_ticks_position("none")
            axs[ax_idx].set_title(condition, fontsize=20)
            # The legend is the last axis so this is a hack to change the font size of the legend
            axs[ax_idx].figure.axes[-1].yaxis.label.set_size(16)
            axs[ax_idx].set_aspect("equal")

    # Save and close the figure
    plt.subplots_adjust(wspace=0.05, hspace=0)
    savepath = BuildSavingFolder(self.dir, self.settings, self.cluster_type, self.condition_types)
    plt.savefig(str(savepath) + "/" + "pa_map_compare.png")
    plt.savefig(str(savepath) + "/" + "pa_map_compare.eps", format="eps")
    if self.settings.show_plots:
        plt.show()
    plt.close()


## --------SMALL PLOTTING UTILS


def real_predicted_trace(ax, real, predicted, fps, title, titleclass):
    x_time = np.arange(len(real)) / (fps * 60)
    ax.plot(x_time, predicted)
    ax.plot(x_time, real)
    ax.legend(["prediction", "real"])
    ax.set_xlim((0, len(real) / (fps * 60)))
    ax.set_title(title)
    ax.set_ylabel(titleclass)
    ax.set_xlabel("time (mins)")


def real_predicted_hist(ax, real, predicted, title, titleclass):
    xlim = [np.amin(real), np.amax(real)]
    bins = np.hstack((np.unique(real) - (np.mean(np.diff(np.unique(real))) / 2), np.unique(real)[-1] + (np.mean(np.diff(np.unique(real))) / 2)))
    ax.hist(predicted, bins, alpha=0.75)
    ax.hist(real, bins, alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel(titleclass)
    ax.set_ylabel("number of frames")


def residual_distribution(ax, real, predicted, titleclass):
    res = np.arctan2(np.sin(predicted - real), np.cos(predicted - real))
    c = [res[real == angles] for angles in np.unique(real)]
    ax.violinplot(c, showextrema=False, positions=np.unique(real))
    ax.set_xlabel("prediction " + titleclass)
    ax.set_ylabel("residual " + titleclass)
    ax.set_box_aspect(1)


## --------OLD VERSIONS


def PredictionAccuracyMapped_old(self, prediction_accuracy):
    """
    Function to make a map of the prediction accuracy for the angle of the head to each point in the arena
    """
    pa = [val for key, val in prediction_accuracy.items() if re.search("randP", key)]

    plt.figure(figsize=(15, 15))
    # add points with prediction accuracy
    s = np.mean(np.diff(np.unique(self.tracking_data["randP_loc"][:, 0]))) * 2
    sc = plt.scatter(self.tracking_data["randP_loc"][:, 0], self.tracking_data["randP_loc"][:, 1], c=pa, s=s, marker="s", cmap="Blues")
    plt.colorbar(sc)
    plt.axis("off")
    # prettify with arena features
    ax = plt.gca()
    Arena(ax=ax, shelter_coordinates=self.tracking_data["shelter_loc"], condition=self.condition, barrier_coordinates=self.session.barrier_location)
    # base_plotting(ax, self.tracking_data, self.condition, session = self.session)
    ax.invert_yaxis()
    ax.set_aspect("equal")

    # save plot
    filename = str(self.savepath) + "/" + str(self.cluster_type) + "_" + str(self.condition) + "_LDA_pa_map" + ".png"
    plt.savefig(filename)
    if self.show_plots:
        plt.show()
    plt.close()
