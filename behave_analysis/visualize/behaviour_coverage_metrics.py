"""
TODO 

1) Currently the grids are plotted in an order that is not intuitive, need to fix this. They don't follow the order of the arena
in other plots so you have to manually figure out which grid is which.

2) Save the plots to a folder

3) remove times in shelter

4) manually define the grid coordinates
"""

# OS Libaries
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import entropy
from loguru import logger

# Plotting settings
sns.set_theme(style="ticks")

class CoverageStatistics:
    """
    This class is designed to take in a video data frame and then compute the coverage statistics for the whole arena and for
    different segments of the arena. This is done by dividing the arena into a grid and then computing the coverage statistics. The resposbility
    of this class is to inform the user if the mouse has sampled all the areas uniformly or if there are areas that are not being sampled.
    """
    def __init__(self, video_data_frame, is_barrier_experiment):
        self.is_barrier_experiment = is_barrier_experiment
        self.video_data_frame = video_data_frame
        _, self.xedges, self.yedges = self.divide_arena_into_grid(video_data_frame, grid_number = 3) # 3 creates a 3x3 grid of the arena
        self.mapped_coordinates = self.map_bin_edges_to_grid_coordinates()
        self.total_samples, self.grid = self.computation_by_grid(video_data_frame, self.xedges, self.yedges, is_barrier_experiment)
        self.kl_divergences = self.compute_kl_divergences(self.grid)
        self.plot_grid_coverage(self.grid)
        self.plot_whole_arena_coverage_and_correlations(self.video_data_frame)
        self.plot_heat_map_of_position()

    def divide_arena_into_grid(self, video_data_frame, grid_number) -> tuple:
        """
        A function that chunks up the arena into grids and imforms the user of the number of samples in each grid.
        """
        # Defining the range for x and y based on their min and max
        x_coords = video_data_frame['mouse_x_position']
        y_coords = video_data_frame['mouse_y_position']
        x_range = (x_coords.min(), x_coords.max())
        y_range = (y_coords.min(), y_coords.max())
        
        # Create a 2D histogram
        H, xedges, yedges = np.histogram2d(x_coords, y_coords, bins = grid_number, range=[x_range, y_range])
        logger.info(f"The xedges are {xedges}")
        logger.info(f"The yedges are {yedges}")
        
        # Unit
        assert H.sum() == len(video_data_frame), f"The number of samples in the grid ({H.sum()}) does not match the number of samples in the video data frame ({len(video_data_frame)})"
        
        return H, xedges, yedges

    def map_bin_edges_to_grid_coordinates(self) -> list:
        """
        Goes through the bin edges and maps them to the grid coordinates, produces a flattened list for plotting
        """
        grid_coordinates = []
        for x in range(3):
            for y in range(3):
                grid_coordinates.append((self.xedges[x], self.yedges[y]))
        return grid_coordinates

    def count_samples_in_grid(self, angle_data_frame) -> dict:
        """
        For each column / angle and a given set of bins across the range of radians. How many samples fall into each bin?
        """
        bin_edges = np.linspace(-np.pi, np.pi, 20)
        result = {}
        
        for column in angle_data_frame.columns:
            bin_indices = np.digitize(angle_data_frame[column].to_numpy(), bins=bin_edges) # Bin the data
            bin_counts = np.bincount(bin_indices) # Count occurrences of each bin index
            result[column] = bin_counts
            
        # Unit test
        total = sum(np.sum(values) for values in result.values())
        assert total == len(angle_data_frame) * len(angle_data_frame.columns), f"Total samples in bins ({total}) does not match total samples in data frame ({len(angle_data_frame)})"
        
        return result

    def computation_by_grid(self, video_data_frame, xedges, yedges, barrier_experiment) -> tuple:
        """
        Returns a dictionary of dictionaries. The first key is the grid number, the second key is the angle, the value is the number of samples in each bin
        which will be of length 20. And returns the total number of samples in the video data frame.
        """
        total_samples = len(video_data_frame)
        grid = {}
        counter = 0
        
        # Loop over x and y bins
        for i in range(len(xedges)-1):
            for j in range(len(yedges)-1):
                
                # Filter rows that fall into current bin
                cell = video_data_frame.filter((pl.col('mouse_x_position').gt(xedges[i])) & 
                                            (pl.col('mouse_x_position').lt(xedges[i+1])) &
                                            (pl.col('mouse_y_position').gt(yedges[j])) & 
                                            (pl.col('mouse_y_position').lt(yedges[j+1])))
                
                # Extract angles from cell
                if barrier_experiment:
                    angle_data_frame = cell.select(['hdir', 'hsa', 'h_bar_north_a', 'h_bar_south_a'])
                
                else:
                    angle_data_frame = cell.select(['hdir', 'hsa'])
                
                # Compute
                grid[counter] = self.count_samples_in_grid(angle_data_frame)
                counter += 1
                
        return total_samples, grid
    
    def compute_kl_divergences(self, dictionary):
        """
        Compute the Kullback-Leibler divergence of the empirical distribution of each grid from the uniform distribution over [-pi, pi].
        """
        
        # Define the uniform distribution over 20 bins
        uniform_dist = np.full((20,), 1/20)  # Assuming 20 bins

        kl_divergences = {}

        for i in range(9):  # Looping over the first 9 dictionaries (or however many you have)
            kl_divergences[i] = {}
            for key, value in dictionary[i].items():
                # Normalize the histogram counts to get a probability distribution
                # Add a small constant to prevent taking the logarithm of zero
                empirical_dist = (value + 1e-10) / np.sum(value + 1e-10)

                # Compute KL divergence
                kl_divergence = entropy(empirical_dist, uniform_dist)
                kl_divergences[i][key] = kl_divergence

        return kl_divergences

    def plot_grid_coverage(self, dictionary):
        """
        Plot the sampling coverage in each grid. 
        """
        fig, axs = plt.subplots(3, 3, figsize=(20, 7))  # Creating a 3x3 grid of Axes
        axs = axs.flatten()  # Flattening the 2D grid of Axes to 1D for easier indexing
        bin_edges = np.linspace(-np.pi, np.pi, 21)
        
        # for each grid
        for i in range(9):  # Looping over the first 9 dictionaries (or however many you have)
            
            # For each angle, plot the histogram of samples in the grid
            for key, value in dictionary[i].items():
                axs[i].bar(bin_edges[:-1], value, width=np.diff(bin_edges), align="edge", alpha=0.5, label=f'{key}, KL: {self.kl_divergences[i][key]:.2f}')

            axs[i].legend(loc='upper right')
            axs[i].set_title(f'Grid: {i+1} | Coordinates: ({self.mapped_coordinates[i][0]:.2f}, {self.mapped_coordinates[i][1]:.2f})')
            axs[i].set_xlabel('Radians')
            axs[i].set_ylabel('Number of samples')
            axs[i].grid(False)
            
        plt.tight_layout()  # Adjusts subplot params so that subplots fit into the figure area
        plt.show()

    def plot_whole_arena_coverage_and_correlations(self, video_data_frame):
        
        if self.is_barrier_experiment:
            angle_data_frame = video_data_frame.select(['hdir', 'hsa', 'h_bar_north_a', 'h_bar_south_a']).to_pandas()
            angle_data_frame.rename(columns={'h_bar_north_a': 'North Edge', 
                                             'h_bar_south_a': 'South Edge', 
                                             'hdir': 'Head Direction', 
                                             'hsa': 'Head Shelter'}, inplace=True)
            
            sns.set_context("talk", font_scale=1.25)
            sns.pairplot(angle_data_frame, diag_kind="kde", corner=True, plot_kws={'s': 2},height= 1.5)
            plt.show()
            x = 10
        
        else:
            angle_data_frame = video_data_frame.select(['hdir', 'hsa']).to_pandas()
            sns.pairplot(angle_data_frame, diag_kind="kde", corner=True, plot_kws={'s': 2},height= 1.5)
            plt.show()
    
    def plot_heat_map_of_position(self):
        x_coords = self.video_data_frame['mouse_x_position']
        y_coords = self.video_data_frame['mouse_y_position']

        plt.figure(figsize=(10, 8))
        heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=(50, 50))

        # Draw heatmap
        ax = sns.heatmap(heatmap, cmap="crest", robust=True)

        # Remove x and y tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Remove x and y ticks
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        
        # Plot
        plt.title('Mouse Position Heatmap')
        plt.xlabel('Mouse X Position')
        plt.ylabel('Mouse Y Position')
        plt.show()