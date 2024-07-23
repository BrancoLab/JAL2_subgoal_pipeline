import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap

# TODO - Grab time before the homing as well

def select_neural_activity_chunk(homing_list, bin_sizes, classes):
    """Selects a chunk of the neural activity at the start of a trial

    Args:
    - number of frames to select"""
    
    # Create differnt bins of neural activity
    # THis creates a homing_dct[homing_number][bin_size][neuron_number]
    assert homing_list[0] != homing_list[1], "The first two homings should be different"
    homing_dict = {}
    for i, homing in enumerate(homing_list):
        arg_dict = {}
        for bin in bin_sizes:
            arg_dict[bin] = homing[:bin].sum(axis=0)  # Sum the first arg frames of the homing list
        homing_dict[i] = arg_dict

    # plot dimensionality reduction
    fig, axs = plt.subplots(len(bin_sizes), figsize=(10, 5 * len(bin_sizes)))
    # Make colour labels based on class list for matplotlib 1 is red, 0 is blue
    colors = ['r' if c == 1 else 'b' for c in classes]

    for i, bin in enumerate(bin_sizes):
        # stack homings into a single array
        for h, homing in enumerate(homing_dict):
            if h == 0:
                X = homing_dict[h][bin]
            else:
                X = np.vstack((X, homing_dict[h][bin]))
        
        if 0:
            pca = PCA(n_components=2)
            embedding = pca.fit_transform(X)
         
        if 1:
            embedding = umap.UMAP(random_state=1337).fit_transform(X)
        
        # Plot the PCA results
        ax = axs[i] if len(bin_sizes) > 1 else axs
        ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7, c=colors)
        ax.set_title(f'PCA for bin size {bin}')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        
        # plot homing labels onto the PCA plot
        for j, txt in enumerate(classes):
            ax.text(embedding[j, 0], embedding[j, 1], j, fontsize=14)
        
    plt.tight_layout()
    plt.show()
    
