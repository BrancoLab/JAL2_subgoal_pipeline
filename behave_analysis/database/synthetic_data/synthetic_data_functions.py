"""ALGORITHM 1: Homogeneous Poisson process; Taken from Thinning Algorithms for Simulating Point Processes 
by Yuanda Chen, 2016, https://www.math.fsu.edu/~ychen/research/Thinning%20algorithm.pdf"""

import numpy as np
import matplotlib.pyplot as plt

def homogeneous_poisson_process(lam, T):
    n = 0
    t0 = 0
    times = []

    while True:
        u = np.random.uniform(low = 0.01, high = 1)
        w = -np.log(u) / lam # So that w ~ Exp(λ)
        tn = t0 + w

        if tn > T:
            return times
        
        else:
            n += 1
            t0 = tn
            times.append(tn)
            
"""ALGORITHM 2: Imhomogeneous Poisson process; Taken from Thinning Algorithms for Simulating Point Processes 
by Yuanda Chen, 2016, https://www.math.fsu.edu/~ychen/research/Thinning%20algorithm.pdf"""

class ImhomogeneousProcess:
    def __init__(self, time_end, peak_time, width, peak_intensity, kernel = "guassian"):
        """
        A class that generates an imhomogeneous Poisson process to the specified kernel function. This can be used
        to generate a latent feature for a time series for a single neuron.
        """
        self.kernel = kernel
        self.T = time_end
        self.peak_time = peak_time
        self.width = width
        self.peak_intensity = peak_intensity
        
        if kernel == "gaussian":
            # print("A gaussian kernel was used to generate the latent feature.")
            self.events = self.simulate_gaussian_intensity()
            
    # Define the Gaussian function
    def gaussian(self, x, mu, sigma):
        return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    # Define the cif_function using the Gaussian function with peak_time, width, and peak_intensity parameters
    def cif_function_gaussian(self, t):
        return self.gaussian(t, mu = self.peak_time, sigma = self.width) * self.peak_intensity
    
    def sample_inhomogeneous_pp_thinning_v2(self):
        """
        ALGORITHM 2: Imhomogeneous Poisson process; Taken from Thinning Algorithms for Simulating Point Processes
        In general I disregard the homogeneous Poisson process and only return the imhomogeneous Poisson process.
        I specifically use this algorithm to generate a latent feature. 

        Args:
            cif_function (_type_): Some kernel function e.g Gaussian, Exponential, etc.
            T (Int): Samples are taken from the interval [0, T]

        Returns:
            Dictionary: Imhomogeneous and homogeneous Poisson processes as per the algorithm
        """
        n = 0
        m = 0
        point = [0]
        s = [0]
        t_values = np.arange(0, self.T, 0.01)
        lambda_bar = np.max(self.cif_function_gaussian(t_values))
        
        while s[m] < self.T:
            # convert to float so polars can accept it doesnt like numpy.float64
            
            u = np.random.uniform(low = 0.01, high = 1)
            w = float(-np.log(u) / lambda_bar) # time event to append 
            s.append(s[m] + w)
            D = np.random.uniform(low = 0.01, high = 1)

            if D <= self.cif_function_gaussian(s[m + 1]) / lambda_bar:
                point.append(s[m + 1])
                n += 1

            m += 1

        if point[-1] <= self.T:
            return {"inhomogeneous": point[1:], "homogeneous": s[1:]}
        
        else:
            return {"inhomogeneous": point[1:-1], "homogeneous": s[1:-1]}

    # Simulate an inhomogeneous Poisson process with the Gaussian intensity rate function
    def simulate_gaussian_intensity(self):
        """
        Returns the inhomogeneous Poisson process events only
        """
        result = self.sample_inhomogeneous_pp_thinning_v2()
        inhomogeneous = result["inhomogeneous"]
        return inhomogeneous

class PointProcessPlotting:
    def __init__(self, inhomogeneous_events, T, num_bins):
        self.inhomogeneous_events = inhomogeneous_events
        self.T = T
        self.num_bins = num_bins
        
        self.compute_hist(inhomogeneous_events, T, num_bins, bin_duration, num_bins)
        self.plot()
        
    # Compute hist
    def compute_hist(self, events, T, bins, bin_duration , num_bins):
        
        # Calculate the bin edges
        self.bin_edges = np.arange(0, T, bin_duration)[:num_bins+1]
        self.hist, _ = np.histogram(events, bins=bins, range=(0, T))
            
    def plot(self):
        
        # Create a 1x2 grid of subplots
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 4), sharex=True)

        # Plot the event plot
        axes[0].eventplot(self.inhomogeneous_events, linelengths=0.8, color='black')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Events')
        axes[0].set_title('Event plot with Gaussian intensity')
        axes[0].grid()
        axes[0].set_xlim(0, T)

        # Plot the histogram
        axes[1].bar(self.bin_edges, self.hist, width = bin_duration, align='edge', edgecolor='black')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Event Count')
        axes[1].set_title('Event histogram with Gaussian intensity')
        axes[1].grid()

        # Display the plots
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
        
    # Parameters for the simulation
    T = 10 
    peak_time_of_kernel = 5 
    width_of_kernel = 1
    peak_intensity_of_kernel = 5
    num_bins = T
    bin_duration = T / num_bins
    
    object = ImhomogeneousProcess(time_end = T, 
                                peak_time = peak_time_of_kernel, 
                                width = width_of_kernel, 
                                peak_intensity = peak_intensity_of_kernel,
                                kernel = "gaussian")

    spike_times = object.events # Tunned to the kernel function
    
    # Parameters for the background events
    lam = 0
    
    background_events = homogeneous_poisson_process(T, lam) # Background events
    event_times = np.concatenate((spike_times, background_events)) # Concatenate the background events and the spike times
    plot_object = PointProcessPlotting(inhomogeneous_events = event_times, T = object.T, num_bins = num_bins)
        
