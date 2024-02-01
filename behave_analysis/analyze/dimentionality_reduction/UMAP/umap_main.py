"""check clusters that have the average angle simialrity and then make that the hdir cluster

TODO"""

import os

import pickle
import umap
import hdbscan
import matplotlib.pyplot as plt
import numpy as np

def run_umap_then_hdbscan(x: np.array, cell_ids, save_path: str) -> dict:
    """Run UMAP then HDBSCAN on the data
    
    Returns:
    -- cluster_dic (dic) a dictionary of cluster labels and the corresponding cell ids"""

    # Create embeddings
    standard_embedding = umap.UMAP(random_state=42).fit_transform(x)
    clusterable_embedding = umap.UMAP(
        n_neighbors=30,
        min_dist=0.0,
        n_components=2,
        random_state=42,
    ).fit_transform(x)

    # Cluster with HDBSCAN
    cluster_labels = hdbscan.HDBSCAN(min_cluster_size=5).fit_predict(clusterable_embedding) # clusters are groups of points (neurons) that are close to each other not neurons
    clustered = cluster_labels >= 0 # Only accept postive clusters why?

    # create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    fig.suptitle('UMAP projection of the dataset: Standard embedding vs clustered embedding', fontsize=16)

    # Plot standard embedding
    ax1.scatter(standard_embedding[:, 0], standard_embedding[:, 1])

    # Plot clustered embedding
    ax2.scatter(standard_embedding[~clustered, 0], standard_embedding[~clustered, 1], color=(0.5, 0.5, 0.5))
    ax2.scatter(standard_embedding[clustered, 0], standard_embedding[clustered, 1], c=cluster_labels[clustered])

    # Plot cell id labels onto the standard embedding
    for i, label in enumerate(cell_ids):
        ax1.text(standard_embedding[i, 0], standard_embedding[i, 1], label, fontsize=4)
    
    # For the second axes plot the cluster label to the correspdonging point
    for i, label in enumerate(cluster_labels):
        ax2.text(standard_embedding[i, 0], standard_embedding[i, 1], label, fontsize=4)
    
    cluster_dict = return_clustered_cell_ids(cluster_labels, cell_ids)
    path = os.path.join(save_path, "UMAP_HBDSCAN.png")
    plt.savefig(path)
    
    # save dictionary with pickle to save path
    with open(os.path.join(save_path, "UMAP_HBDSCAN_cluster_labels_dictionary.pickle"), "wb") as f:
        pickle.dump(cluster_dict, f)

    return cluster_dict
    
def return_clustered_cell_ids(cluster_labels, cell_ids) -> dict:
    """Return the cell ids belonging to each cluster"""
    # Dictionary to hold cluster labels and corresponding cell IDs for the return
    cluster_dict = {}
    for label, cell_id in zip(cluster_labels, cell_ids):
        if label not in cluster_dict:
            cluster_dict[label] = []
        cluster_dict[label].append(cell_id)
    return cluster_dict