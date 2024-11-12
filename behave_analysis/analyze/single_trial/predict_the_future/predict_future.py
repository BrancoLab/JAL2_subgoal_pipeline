import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap
import hdbscan
import threading
from sklearn.cluster import KMeans
import pickle
from scipy.stats import circmean



# ---------------------- CHECKS -------------------------------
def check_spike_homing_list(spike_homing_list):
    """Check that the spike homing list is a list of numpy arrays"""
    assert isinstance(spike_homing_list, list), "The spike homing list should be a list"
    assert spike_homing_list[0] != spike_homing_list[1], "The first two homings should be different"


####################################################################


####################################################################################################
# MAIN

def get_average_behaviour_in_each_bin_looking_back(homing_list, bin_size, video_df):
    behaviour = {}
    for i, homing in enumerate(homing_list):
        onset = homing["frames"][0]
        start = onset - bin_size
        filtered_df = video_df[start : start + 10] # grab the 10 frames before the homing event
        avg_x = filtered_df["mouse_x_position"].mean()
        avg_y = filtered_df["mouse_y_position"].mean()
        hdir = circmean(filtered_df["hdir"])
        behaviour[i] = {"avg_x": avg_x, "avg_y": avg_y, "hdir": hdir}
    return behaviour
        


def look_at_neural_activity_before_homing(homing, frame_by_cluster_matrix, look_back_bin_size):
    """If a negative bin is supplied, this function will look at the neural activity before the homing event"""
    grab_first_frame = homing["frames"][0]
    start_frame = grab_first_frame - look_back_bin_size
    # neural_data = frame_by_cluster_matrix[start_frame : grab_first_frame]
    neural_data = frame_by_cluster_matrix[start_frame : start_frame + 10]
    bin_sum = neural_data.sum(axis=0)  # Sum the neural activity over the frames to get a single value for each neuron
    return bin_sum


def look_at_neural_activity_after_homing(spike_homing, look_forward_bin_size):
    """Sum the first look_forward_bin_size frames of the homing list"""
    frame_end = look_forward_bin_size
    frame_start = frame_end - 10
    return spike_homing[frame_start:look_forward_bin_size].sum(axis=0)


def select_neural_activity_chunk(spike_homing_list, bin_sizes, classes, frame_by_cluster_matrix, homing_list, save_path, video_df):
    """Selects a chunk of the neural activity before and the homing event, does UMAP on the data to see if it clusters

    Returns:
    homing_dict: A dictionary of the neural activity for each homing event
        of the form homing_dict[homing_number][bin_size][neuron_number]
    """
    check_spike_homing_list(spike_homing_list)
    homing_dict = {}
    for i, (spike_homing, homing) in enumerate(zip(spike_homing_list, homing_list)):
        arg_dict = {}
        for bin in bin_sizes:
            assert bin != 0, "You can't look at zero frames, remove 0 from the bin_sizes list"
            if bin < 0:
                arg_dict[bin] = look_at_neural_activity_before_homing(homing, frame_by_cluster_matrix, abs(bin))
            elif bin >= 0:
                arg_dict[bin] = look_at_neural_activity_after_homing(spike_homing, bin)
        homing_dict[i] = arg_dict
    
    bin_behve = {}
    for bin in bin_sizes:
        if bin < 0:
            behaviour = get_average_behaviour_in_each_bin_looking_back(homing_list, bin, video_df)
            bin_behve[bin] = behaviour
            

    # save = save_path / "homing_dict.pkl"
    # with open(save, "wb") as f:
    #     pickle.dump(homing_dict, f)

    # class_save = save_path / "classes.pkl"
    # with open(class_save, "wb") as f:
    #     pickle.dump(classes, f)
    
    UMAP_plot_dimentionality_reduction(homing_dict, bin_sizes, classes, save_path)
    # UMAP_plot_dimentionality_reduction(homing_dict, bin_sizes, classes, save_path, behaviour=bin_behve)
    return homing_dict


def explore_neural_activity_over_time(spike_data_per_homing, left_or_right_labels):
    """Explore the neural activity over time for each homing event"""

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    colors = ["r" if c == 1 else "b" for c in left_or_right_labels]  # Make color labels based on class list for matplotlib: 1 is red, 0 is blue
    for i, homing_spikes in enumerate(spike_data_per_homing):
        pca = PCA(n_components=10)
        spike_data = pca.fit_transform(homing_spikes)
        ax.scatter(spike_data[:, 0], spike_data[:, 1], spike_data[:, 2], color=colors[i])
        ax.plot(spike_data[:, 0], spike_data[:, 1], spike_data[:, 2], color=colors[i], linewidth=0.5, alpha=0.5)
        # plt.show()

        # Plot the spike data with a line through each homing event

        # Plot the spike


# ---------------------- PLOTTING -------------------------------


# PCA
def PCA_plot_dimentionality_reduction(homing_dict, bin_sizes, classes=None, save_path=None):
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("This function should run in the main thread.")

    fig, axs = plt.subplots(len(bin_sizes), 3, figsize=(12, 5 * len(bin_sizes)))

    if classes is None:
        colors = ["r"] * len(homing_dict)

    else:

        colors = ["r" if c == 1 else "b" for c in classes]  # Make color labels based on class list for matplotlib: 1 is red, 0 is blue

    for i, bin in enumerate(bin_sizes):
        X = None
        for h, homing in enumerate(homing_dict):
            if h == 0:
                X = homing_dict[homing][bin]
            else:
                X = np.vstack((X, homing_dict[homing][bin]))

        # PCA embedding and clustering
        pca = PCA(n_components=10)
        embedding = pca.fit_transform(X)

        kmeans = KMeans(n_clusters=2, random_state=1337).fit(embedding)
        cluster_labels = kmeans.labels_

        # Plot PC1 vs PC2, PC2 vs PC3, PC3 vs PC4
        for pc in range(3):
            axs[i, pc].scatter(embedding[:, pc], embedding[:, pc + 1], alpha=0.7, c=colors)
            axs[i, pc].set_title(f"Reduce and cluster for bin size {bin}")
            axs[i, pc].set_xlabel(f"PC{pc + 1}")
            axs[i, pc].set_ylabel(f"PC{pc + 2}")
            for j, label in enumerate(cluster_labels):
                axs[i, pc].text(embedding[j, pc], embedding[j, pc + 1], str(label), fontsize=14)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path / "PCA.png")
    plt.close()


# UMAP
def UMAP_plot_dimentionality_reduction(homing_dict, bin_sizes, classes=None, save_path=None, behaviour=None):
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("This function should run in the main thread.")

    fig, axs = plt.subplots(len(bin_sizes), figsize=(12, 5 * len(bin_sizes)))
    if classes is None:
        # randomly select colours of length of homing_dict

        colors = ["r"] * len(homing_dict)
    
    else:
        colors = ["r" if c == 1 else "b" for c in classes]  # Make colour labels based on class list for matplotlib 1 is red, 0 is blue
        
    for i, bin in enumerate(bin_sizes):
        for h, homing in enumerate(homing_dict):
            # stack homings into a single array
            if h == 0:
                X = homing_dict[h][bin]
            else:
                X = np.vstack((X, homing_dict[h][bin]))
                
        if behaviour:
            if bin < 0:
                beh = behaviour[bin]
                
                import matplotlib.cm as cm
                import matplotlib.colors as mcolors
                #norm = mcolors.Normalize(vmin=0, vmax=1000) # x position
                norm = mcolors.Normalize(vmin=0, vmax=6.28) # hdir
                cmap = cm.get_cmap('hsv')
                
                # loop through dictionary to get all the xs
                h_dir = np.array([beh[i]["hdir"] for i in range(len(beh))])
                #x_coords = np.array([beh[i]["avg_x"] for i in range(len(beh))])
                #colors = cmap(norm(x_coords))
                colors = cmap(norm(h_dir))

        # standard embedding
        # embedding = umap.UMAP(random_state=1337).fit_transform(X)

        if 1:
            # clusterable embedding
            clusterable_embedding = umap.UMAP(
                n_neighbors=30,
                min_dist=0.0,
                n_components=2,
                random_state=1337,
            ).fit_transform(X)
            embedding = clusterable_embedding

            #cluster_labels = hdbscan.HDBSCAN(min_cluster_size=5).fit_predict(clusterable_embedding)  # clusters are groups of points (neurons) that are close to each other not neurons

        ax = axs[i] if len(bin_sizes) > 1 else axs
        ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7, c=colors)
        ax.set_title(f"Reduce and cluster for bin size {bin}")
        ax.set_xlabel("B1")
        ax.set_ylabel("B2")
        
        # # plot homing labels onto the PCA plot
        for j, _ in enumerate(classes):
            ax.text(embedding[j, 0], embedding[j, 1], j, fontsize=14)

        # # plot cluster labels as well
        # for j, label in enumerate(cluster_labels):
        #     if label != -1:
        #         ax.text(embedding[j, 0], embedding[j, 1], label, fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path / "UMAP.png")
    plt.close()
