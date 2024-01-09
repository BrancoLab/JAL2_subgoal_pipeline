"""A script that:
1) first runs PCA on the polar plot data, 
2) then runs K means clustering on the PCA transformed data and saves the labels. 
3) Then the script plots the PCA transformed data and colors the
data points based on the K means labels."""

import os

import numpy as np
from loguru import logger
from sklearn import decomposition
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def run_pca_kmeans_plot(path_to_save, x, labels):
    """Runs PCA, K means clustering and plots the PCA transformed data"""
    pca = decomposition.PCA(n_components=5)
    pca.fit(x)
    logger.success("Completed PCA")
    transformed_data = pca.transform(x)
    kmeans_labels = k_means(transformed_data, path_to_save)
    # plot_pca(transformed_data, labels, pca, kmeans_labels)
    plot_pca_plotly(x, labels, pca, kmeans_labels)


def k_means(pca_transformed_data, path_to_save):
    """Runs K means clustering on the PCA transformed data and saves the labels"""
    kmeans = KMeans(n_clusters=9, random_state=0, n_init="auto")
    kmeans.fit(pca_transformed_data)
    np.save(os.path.join(path_to_save, "kmeans_labels.npy"), kmeans.labels_)
    logger.success("Completed K means clustering and saved labels")
    return kmeans.labels_


def plot_pca(x, labels, pca_model, kmeans_labels):
    """Plots the PCA transformed data and colors the data points based on the K means labels

    Currently not used as the plotly version is more interactive"""
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(3, 2)

    # 3D scatter plot - first column, spanning two rows
    ax0 = fig.add_subplot(gs[:, 0], projection="3d")
    ax0.scatter(
        x[:, 0],
        x[:, 1],
        x[:, 2],
        c=kmeans_labels,
        cmap=plt.cm.nipy_spectral,
        edgecolor="k",
    )
    ax0.set_xlabel("PC1")
    ax0.set_ylabel("PC2")
    ax0.set_zlabel("PC3")
    for i in range(len(x)):
        ax0.text(
            x[i, 0],
            x[i, 1],
            x[i, 2],
            "%s" % (labels[i]),
            size=8,
            zorder=1,
            color="k",
        )

    # 2D scatter plot - first and second PCs - top row, second column
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.scatter(
        x[:, 0], x[:, 1], c=kmeans_labels, cmap=plt.cm.nipy_spectral, edgecolor="k"
    )
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    for i in range(len(x)):
        ax1.text(x[i, 0], x[i, 1], "%s" % (labels[i]), size=8, zorder=1, color="k")
    # add the explained variance for each component in the title of the plot
    ax1.set_title(
        f"Explained Variance for PC 1: {pca_model.explained_variance_ratio_[0]} and PC 2: {pca_model.explained_variance_ratio_[1]}"
    )

    # 2D scatter plot - second and third PCs - bottom row, second column
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.scatter(
        x[:, 1], x[:, 2], c=kmeans_labels, cmap=plt.cm.nipy_spectral, edgecolor="k"
    )
    ax2.set_xlabel("PC2")
    ax2.set_ylabel("PC3")
    for i in range(len(x)):
        ax2.text(x[i, 1], x[i, 2], "%s" % (labels[i]), size=8, zorder=1, color="k")
    ax2.set_title(
        f"Explained Variance for PC 2: {pca_model.explained_variance_ratio_[1]} and PC 3: {pca_model.explained_variance_ratio_[2]}"
    )

    # 2D scatter plot - third and fourth PCs - bottom row, second column
    ax3 = fig.add_subplot(gs[2, 1])
    ax3.scatter(
        x[:, 2], x[:, 3], c=kmeans_labels, cmap=plt.cm.nipy_spectral, edgecolor="k"
    )
    ax3.set_xlabel("PC3")
    ax3.set_ylabel("PC4")
    for i in range(len(x)):
        ax3.text(x[i, 2], x[i, 3], "%s" % (labels[i]), size=8, zorder=1, color="k")
    ax3.set_title(
        f"Explained Variance for PC 3: {pca_model.explained_variance_ratio_[2]} and PC 4: {pca_model.explained_variance_ratio_[3]}"
    )

    # Create a colormap normalized by the number of clusters
    n_clusters = len(np.unique(kmeans_labels))
    cmap = plt.cm.nipy_spectral
    norm = mcolors.Normalize(vmin=0, vmax=n_clusters - 1)

    # Function to map the cluster index to a color
    def cluster_color(i):
        return cmap(norm(i))

    # Create legend patches
    patches = [
        mpatches.Patch(color=cluster_color(i), label=f"Cluster {i}")
        for i in range(n_clusters)
    ]

    # Place the legend on the plot
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.show()


def plot_pca_plotly(x, labels, pca_model, kmeans_labels):
    """Creates an interactive PCA plot in the broswer using plotly
    
    No saving of the plot is currently implemented. Could be buggy
    as first time using plotly"""
    # Create a subplot figure with 2 columns
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "scatter3d", "rowspan": 2}, {}], [None, {}]],
        subplot_titles=[
            "3D Scatter Plot: PC1 vs PC2 vs PC3",
            f"Explained Variance PC1: {pca_model.explained_variance_ratio_[0]:.2f}, PC2: {pca_model.explained_variance_ratio_[1]:.2f}",
            f"Explained Variance PC2: {pca_model.explained_variance_ratio_[1]:.2f}, PC3: {pca_model.explained_variance_ratio_[2]:.2f}",
            "",
        ],
        column_widths=[0.5, 0.5],
        vertical_spacing=0.1,
        horizontal_spacing=0,
    )

    # Get unique clusters and a color for each
    unique_clusters = np.unique(kmeans_labels)
    # Choose a color palette, change Dark24 to any other palette name
    colors = px.colors.qualitative.Dark2

    # Function to add traces for each cluster
    def add_cluster_traces(cluster):
        idx = kmeans_labels == cluster
        cluster_color = colors[cluster % len(colors)]
        cluster_labels = labels[idx]  # Get labels for this cluster
        # 3D Scatter plot
        fig.add_trace(
            go.Scatter3d(
                x=x[idx, 0],
                y=x[idx, 1],
                z=x[idx, 2],
                mode="markers+text",
                marker=dict(color=cluster_color, size=10),
                name=f"Cluster {cluster}",
                text=cluster_labels,
                textposition="top center",
            ),
            row=1,
            col=1,
        )
        # 2D Scatter plot for PC1 vs PC2
        fig.add_trace(
            go.Scatter(
                x=x[idx, 0],
                y=x[idx, 1],
                mode="markers+text",
                marker=dict(color=cluster_color, size=10),
                name=f"Cluster {cluster}",
                showlegend=False,
                text=labels,
                textposition="top center",
            ),
            row=1,
            col=2,
        )
        # 2D Scatter plot for PC2 vs PC3
        fig.add_trace(
            go.Scatter(
                x=x[idx, 1],
                y=x[idx, 2],
                mode="markers+text",
                marker=dict(color=cluster_color, size=10),
                name=f"Cluster {cluster}",
                showlegend=False,
                text=labels,
                textposition="top center",
            ),
            row=2,
            col=2,
        )

    # Add traces for each cluster
    for cluster in unique_clusters:
        add_cluster_traces(int(cluster))

    # Update layout with larger size
    fig.update_layout(
        height=1400,
        width=3500,
        legend=dict(font=dict(size=24)),
        title=dict(font=dict(size=30)),
        title_text="PCA and K-means Clustering",
    )

    # Update axes labels and font size for 3D scatter plot
    fig.update_layout(
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
            xaxis=dict(title_font=dict(size=20)),
            yaxis=dict(title_font=dict(size=20)),
            zaxis=dict(title_font=dict(size=20)),
        )
    )

    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].update(font=dict(size=30))

    fig.show()


def jasimine_plot(self, Kmeans_labels):
    """Heat plot per kmeans cluster for each condition and angle
    
    Not currently called anywhere or saved"""
    
    # Add one for labels
    rows = len(self.conditions) + 1
    columns = len(self.angles) + 1

    first_key = list(self.condition_data.keys())[0]
    second_key = list(self.condition_data[first_key].keys())[0]

    unique_neurons = np.unique(
        self.condition_data[first_key][second_key]["clusterID"]
    )

    ticks = np.linspace(-3.14, 3.14, 3)
    tick_pos = np.linspace(0, 17, 3)

    for kmeans_cluster in np.unique(Kmeans_labels):
        # Get all neurons in cluster
        ids_in_cluster = np.where(Kmeans_labels == kmeans_cluster)

        # Create figure for each kmeans cluster
        gs = gridspec.GridSpec(
            rows,
            columns,
            bottom=0.05,
            top=0.95,
            hspace=0.5,
            wspace=0.5,
            width_ratios=[1] + [3] * (columns - 1),
            height_ratios=[1] + [3] * (rows - 1),
        )
        fig = plt.figure(figsize=(30, 45), constrained_layout=True)
        gs.tight_layout(fig, pad=0, h_pad=0, w_pad=0)
        plt.suptitle(f"K means cluster {kmeans_cluster}")

        # Add subtitles for each angle in first row
        for a_counter, a in enumerate(self.angles):
            ax = plt.subplot(gs[0, a_counter + 1])
            ax.text(0.5, 0.5, a, rotation="horizontal", va="center", ha="center")
            ax.set_axis_off()

        # Add subtitles for each condition in first column
        for c_counter, c in enumerate(self.condition_data.keys()):
            ax = plt.subplot(gs[c_counter + 1, 0])
            ax.text(0, 0.5, c, rotation="horizontal", va="center", ha="center")
            ax.set_axis_off()

        # Calculate the global maximum firing rate
        global_max = 0
        for condition in self.condition_data.values():
            for angle_data in condition.values():
                firing_rates = np.array(
                    angle_data.filter(
                        pl.col("clusterID").is_in(
                            (unique_neurons[ids_in_cluster].tolist())
                        )
                    )["angle_firing_hist"]
                )
                com1 = np.array([sub_array[:18] for sub_array in firing_rates])
                com2 = np.array([sub_array[18:] for sub_array in firing_rates])
                current_max = max(com1.max(), com2.max())
                if current_max > global_max:
                    global_max = current_max

        for c_counter, condition in enumerate(self.condition_data):
            counter = ((columns) * (c_counter + 1)) + 1
            con_df = self.condition_data[condition]

            # for angle in condition
            for angle_counter, angle in enumerate(self.condition_data[condition]):
                df = con_df[angle]
                # ax = plt.subplot(rows, columns, axs_counter + 1)

                outer_grid = plt.subplot(gs[counter])
                counter = counter + 1
                inner_grid = gridspec.GridSpecFromSubplotSpec(
                    2, 1, subplot_spec=outer_grid
                )
                ax1 = plt.subplot(inner_grid[0])
                ax2 = plt.subplot(inner_grid[1])

                firing_rates = np.array(
                    df.filter(
                        pl.col("clusterID").is_in(
                            (unique_neurons[ids_in_cluster].tolist())
                        )
                    )["angle_firing_hist"]
                )

                # Apparently all the firing rate is in one list which is aboslutey mental so need to fix that but here we go
                # NOTE chance for bugs hard coded 18
                # What the hell is going on
                com1 = np.array([sub_array[:18] for sub_array in firing_rates])
                com2 = np.array([sub_array[18:] for sub_array in firing_rates])

                # NOTE not perfeclty centred as 18 % 2 == 0
                # sort and center the data
                target_index = 9

                # Function to roll each row so that the max value is in the middle
                def roll_row_to_middle(row):
                    max_index = np.argmax(row)  # return idx of max
                    rotation_steps = target_index - max_index
                    return np.roll(row, rotation_steps)

                rolled1 = np.apply_along_axis(roll_row_to_middle, 1, com1)
                rolled2 = np.apply_along_axis(roll_row_to_middle, 1, com2)

                ax1.imshow(rolled1, aspect="auto", vmax=global_max)
                ax2.imshow(rolled2, aspect="auto", vmax=global_max)
                ax2.set_xticks(tick_pos)
                ax2.set_xticklabels(ticks)

                # label
                ax1.set_ylabel("ID")
                ax2.set_ylabel("ID")

        plt.show()
