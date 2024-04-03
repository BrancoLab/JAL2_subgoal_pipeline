"""All the functions for plotting LDA results"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale
import numpy as np
import re
import pickle

from behave_analysis.analyze.LDA.LDA_utils import BuildSavingFolder
from behave_analysis.analyze.behaviour.spatial_efficiency import base_plotting
from behave_analysis.utils.heatplot_utils import add_features

## --------------- PLOTTING FUNCTIONS


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
        fig.add_trace(go.Bar(x=[var], y=[prediction_accuracy[var]], width=0.5, marker=dict(color=colorz[i], opacity=0.5)))

    if len(list(filter(lambda x: "randP" in x, title))) < 10:
        for j, var in enumerate(list(filter(lambda x: "randP" in x, title))):
            fig.add_trace(go.Bar(x=[var], y=[prediction_accuracy[var]], width=0.5, marker=dict(color=colorz[i + j], opacity=0.5)))
    else:
        res = [val for key, val in prediction_accuracy.items() if re.search("randP", key)]
        var = "randP"
        fig.add_trace(go.Violin(x=[var] * len(res), y=res, points="all", jitter=0.05, marker=dict(size=3, color=colorz[i + 1])))

    fig.update_layout(showlegend=False)
    fig.update_yaxes(range=[0, 1])
    fig.update_yaxes(title_text="prediction accuracy")
    fig.update_xaxes(tickangle=-45)
    filename = str(self.savepath) + "/" + str(self.cluster_type) + "_" + str(self.condition) + "_LDA_prediction_accuracy" + ".png"
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
        if i >= 10:
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
    filename = str(self.savepath) + "/" + str(self.cluster_type) + "_" + str(self.condition) + "_LDA_LS_prediction_accuracy" + ".png"
    fig.write_image(filename)


def PredictionAccuracyMapped(self, prediction_accuracy):
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
    base_plotting(ax, self.tracking_data, self.condition)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    filename = str(self.savepath) + "/" + str(self.cluster_type) + "_" + str(self.condition) + "_LDA_prediction_accuracy_map" + ".png"
    plt.savefig(filename)
    if self.show_plots:
        plt.show()
    plt.close()


def across_conditions_LDA_map(self, settings):
    """
    Function to make a figure of the maps of the prediction accuracy for the angle of the head to each point in the arena
    all the maps for all the conditions are displayed together and color axes are adjusted to match across maps
    This also turns the map into heatmaps instead of scatterplots

    figure: each row is a compartment, each column is a condition
    """

    pa = []

    for comp in settings.compartment_split:
        for c in self.all_conditions:
            self.savepath = BuildSavingFolder(self.dir, settings, self.cluster_type, self.condition_types, c, comp)
            LDA_out = str(self.savepath) + "/" + str(self.cluster_type) + "_" + str(c) + "_LDA_prediction_accuracy" + ".pkl"
            with open(LDA_out, "rb") as dill_file:
                prediction_accuracy = pickle.load(dill_file)
            pa.append([val for key, val in prediction_accuracy.items() if re.search("randP", key)])
    
    pa = np.array(pa)
    if len(pa[pa>0]) > 0:
        vmin = np.amin(pa[pa>0])
    else:
        vmin = 0
    vmax = np.amax(pa)

    # figure set-up
    fig, axs = plt.subplots(
        nrows=len(settings.compartment_split), ncols=len(self.all_conditions) + 1, figsize=(24, 6 * len(settings.compartment_split)), sharey=True, sharex=True
    )

    # Where to plot the colorbar, create new axis object at these coordinates
    cbar_ax = fig.add_axes([0.91, 0.13, 0.01, 0.75])  # The list represents [left, bottom, width, height],

    # Add subtitles for each condition in first column

    for c_counter, c in enumerate(settings.compartment_split):
        if len(settings.compartment_split) > 1:
            ax_idx = tuple([c_counter,0])
        else:
            ax_idx = 0
        axs[ax_idx].text(0, 0.5, c, rotation="horizontal", va="center", ha="center", fontsize=23)
        axs[ax_idx].set_axis_off()

    data_counter = 0
    for c_idx, comp in enumerate(settings.compartment_split):
        for idx, condition in enumerate(self.all_conditions):
            if len(settings.compartment_split) > 1:
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
                cmap="coolwarm",
                cbar_ax=cbar_ax,
                robust=True,
                ax=axs[ax_idx],
                mask=(heatmap == 0),
                cbar_kws={"label": "Prediction accuracy"},
                norm=plt.Normalize(vmin=vmin, vmax=vmax),
            )

            add_features(axs[ax_idx], condition, self.tracking_data, xbins, ybins)

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
    savepath = BuildSavingFolder(self.dir, settings, self.cluster_type, self.condition_types)
    plt.savefig(str(savepath) + "/" + "prediction_accuracy_map_compare.png")
    if settings.show_plots:
        plt.show()
    plt.close()