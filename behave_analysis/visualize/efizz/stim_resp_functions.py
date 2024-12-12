'''a set of functions for visualizing the stimulus (i.e. threat) response of neurons'''

# set up
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

def PSTH_all_neurons(session, data, stim_type, show_plots, save_path):
    """
    Plot the mean firing rate of all cells to each trial. For each trial, retrieve:
    - the onset frame of that stimulus
    - the duration of that stimulus
    """
    
    # Hyperparameters
    timeBeforeStim = 5 # seconds
    stimulus_durations = np.amax(session.__dict__[stim_type].stimulus_durations)

    # plot a line of mean activity for each trial
    for trial_num, onset_frames in enumerate(session.__dict__[stim_type].onset_frames):
        time1 = (onset_frames / session.video.fps) - timeBeforeStim 
        time2 = (onset_frames / session.video.fps) + stimulus_durations
        
        # Mask spikes that are within the time window
        spikes_trial = data.filter((data['aligned_spike_times'] > time1) & (data['aligned_spike_times'] < time2))
        
        # Bin the spikes
        mult = 10 # binsize for looking at data - 1/10 of a second so 100ms bins 
        binEdges = np.arange(time1, time2, 1 / mult)
        firingrate, _ = np.histogram(spikes_trial['aligned_spike_times'].to_numpy(), binEdges)
        assert len(firingrate) == len(binEdges) - 1, "firingrate and binedges are not the same length"
        
        # Generate x values for plotting
        xValues = binEdges - time1 - timeBeforeStim
        assert xValues[0] == -timeBeforeStim, f"xValues[0] is not -{timeBeforeStim}"
        
        # Plot the PSTH
        # plt.plot(xValues[:-1], gaussian_filter1d(firingrate * mult, sigma = 1), label = f"Trial #: {trial_num}") # because our bin size is 1/mult of a second
        plt.plot(xValues[:-1], firingrate * mult, label = f"Trial #: {trial_num}") # because our bin size is 1/mult of a second

        plt.axvline(x = 0, color = 'k', linestyle = '-')
        plt.ylabel('Firing rate for all cells (Hz)')
        plt.xlabel('time (s)')
        plt.legend()
    
    plt.title('Trial by trial response PSTH for stimulus type: ' + stim_type)
    plt.savefig(save_path)
    
    if show_plots: 
        plt.show()
        
    plt.close()

def PSTH_single_neurons(data, session, stim_type, save_path, show_plots):
    """
    Plot the mean firing rate of each cluster averaged across all trials.
    """
    
    timeBeforeStim = 5
    stimulus_durations = np.amax(session.__dict__[stim_type].stimulus_durations) + 2
    xlim = [timeBeforeStim * -1,stimulus_durations]

    # Mask spikes that are within the time window
    for trial, onset_frames in enumerate(session.__dict__[stim_type].onset_frames):
        time1 = (onset_frames / session.video.fps) - timeBeforeStim 
        time2 = (onset_frames / session.video.fps) + stimulus_durations
        filt = data.filter(
                    (data['aligned_spike_times'] > time1) &
                    (data['aligned_spike_times'] < time2)
        )
        if hasattr(pl.col('aligned_spike_times'),'apply'):
            filt = filt.select([pl.col('aligned_spike_times').apply(lambda x: x -(onset_frames/session.video.fps)),
                                pl.col('spike_clusters'),
                                pl.Series("trial", np.ones(len(filt)).astype(int)*(trial+1))])
        else:
            filt = filt.select([(pl.col('aligned_spike_times') - (onset_frames / session.video.fps)).alias('aligned_spike_times'),
                                pl.col('spike_clusters'),
                                pl.lit((trial + 1)).cast(int).alias('trial')])
        if trial == 0: spikes_trial = filt
        else: spikes_trial = spikes_trial.vstack(filt)      

    # How many plots do we need?
    number_of_clusters = data["spike_clusters"].unique()
    number_of_plots = len(number_of_clusters)
    max_plots_per_figure = 20
    
    # How many columns and rows should the plot have
    num_cols = int(np.ceil(np.sqrt(max_plots_per_figure)))
    num_rows = int(np.ceil(max_plots_per_figure / num_cols))
    
    # Across how many figures
    num_figures = int(np.ceil(number_of_plots / max_plots_per_figure))
    
    # Create the figures
    plot_counter = 0

    # firing rate binning
    mult = 10 # binsize for looking at data - 1/10 of a second so 100ms bins 
    binEdges = np.arange(xlim[0], xlim[1], 1 / mult)
    xValues = binEdges
    
    # For each figure
    for figure_idx in range(num_figures):
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(24, 8))
        
        # For each plot
        for rows in range(num_rows):
            for columns in range(num_cols):
                if plot_counter < number_of_plots:
                    cluster = number_of_clusters[plot_counter]
                    spikes_trial_cluster = spikes_trial.filter(spikes_trial['spike_clusters'] == cluster)
                    firingrate, _ = np.histogram(spikes_trial_cluster['aligned_spike_times'].to_numpy(), binEdges)
                    # Plot the PSTH
                    # plt.plot(xValues[:-1], gaussian_filter1d(firingrate * mult, sigma = 1), label = f"Trial #: {trial_num}") # because our bin size is 1/mult of a second
                    axes[rows, columns].plot(xValues[:-1], firingrate * mult)
                    axes[rows, columns].set_title(f"Cluster: {cluster}")
                    axes[rows, columns].vlines(0, 0, np.amax(firingrate * mult), colors='r', linestyles='solid')
                    axes[rows, columns].set_xlim(xlim)
                    axes[rows, columns].set_ylabel('Firing rate for all cells (Hz)')
                    axes[rows, columns].set_xlabel('time (s)')
                
                # Remove the extra axes if there are no more plots
                else:
                    fig.delaxes(axes[rows, columns])
                
                plot_counter += 1
        
        # SAVE FIGURE
        fig.tight_layout()
        plt.savefig(save_path + "_cluster_PSTH_" + str(figure_idx) + ".png")                
    
    if show_plots: 
        plt.show()

def rasters(data, session, stim_type, save_path, show_plots):
    """
    A function that extracts spike times and aligns it to trials as a raster plot
    """
    
    # make a raster plot for each trial
    ntrial = len(session.__dict__[stim_type].onset_frames)
    plt.figure(figsize=(15, 12))
    plt.subplots_adjust(hspace=0.2)

    # set number of rows and calculate number of columns
    nrows = 3
    ncols = ntrial // nrows + (ntrial % nrows > 0)
    timeBeforeStim = 5 # in seconds
    all_stimulus_durations = np.amax(session.__dict__[stim_type].stimulus_durations)+2

    for trial_num, (onset_frames, stim_duration) in enumerate(zip(session.__dict__[stim_type].onset_frames, session.__dict__[stim_type].stimulus_durations)):
        ax = plt.subplot(nrows, ncols, trial_num + 1)
        time1 = (onset_frames/session.video.fps) - timeBeforeStim
        time2 = (onset_frames/session.video.fps) + all_stimulus_durations
        spikes_trial = data.filter((data['aligned_spike_times'] > time1) & (data['aligned_spike_times'] < time2))
        ax.scatter(spikes_trial['aligned_spike_times'].to_numpy()-(onset_frames/session.video.fps),
                    spikes_trial['spike_clusters'].to_numpy(),
                    marker='|', s=5, c='k')
        ax.plot([0,0],[0, np.amax(spikes_trial['spike_clusters'].to_numpy())],'r-')
        ax.plot([stim_duration,stim_duration],[0, np.amax(spikes_trial['spike_clusters'].to_numpy())],'r-')
        ax.set_ylabel('clusters')
        ax.set_xlabel('time from stim (s)')
    plt.savefig(save_path)
    
    if show_plots: 
        plt.show()
    
    plt.close()

def single_cluster_raster(data, session, stim_type, save_path, show_plots):
    """
    A function that extracts spike times for each cluster and aligns it to trials as a raster plot
    """
    
    timeBeforeStim = 5
    stimulus_durations = np.amax(session.__dict__[stim_type].stimulus_durations) + 2
    xlim = [timeBeforeStim * -1,stimulus_durations]

    # Mask spikes that are within the time window
    for trial, onset_frames in enumerate(session.__dict__[stim_type].onset_frames):
        time1 = (onset_frames / session.video.fps) - timeBeforeStim 
        time2 = (onset_frames / session.video.fps) + stimulus_durations
        filt = data.filter((data['aligned_spike_times'] > time1) & (data['aligned_spike_times'] < time2))
        if hasattr(pl.col('aligned_spike_times'), 'apply'):
            filt = filt.select([pl.col('aligned_spike_times').apply(lambda x: x -(onset_frames/session.video.fps)),
                                pl.col('spike_clusters'),
                                pl.Series("trial", np.ones(len(filt)).astype(int)*(trial+1))])
        else:
            filt = filt.select([(pl.col('aligned_spike_times') - (onset_frames / session.video.fps)).alias('aligned_spike_times'),
                                pl.col('spike_clusters'),
                                pl.Series("trial", np.ones(len(filt)).astype(int)*(trial+1))])
        if trial == 0: spikes_trial = filt
        else: spikes_trial = spikes_trial.vstack(filt)      

    # How many plots do we need?
    number_of_clusters = data["spike_clusters"].unique()
    number_of_plots = len(number_of_clusters)
    max_plots_per_figure = 20
    
    # How many columns and rows should the plot have
    num_cols = int(np.ceil(np.sqrt(max_plots_per_figure)))
    num_rows = int(np.ceil(max_plots_per_figure / num_cols))
    
    # Across how many figures
    num_figures = int(np.ceil(number_of_plots / max_plots_per_figure))
    
    # Create the figures
    plot_counter = 0
    
    # For each figure
    for figure_idx in range(num_figures):
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(24, 8))
        
        # For each plot
        for rows in range(num_rows):
            for columns in range(num_cols):
                if plot_counter < number_of_plots:
                    cluster = number_of_clusters[plot_counter]
                    spikes_trial_cluster = spikes_trial.filter(spikes_trial['spike_clusters'] == cluster)
                    axes[rows, columns].scatter(spikes_trial_cluster['aligned_spike_times'].to_numpy(),
                                                spikes_trial_cluster['trial'].to_numpy(),
                                                marker='|', s=10, c='k')
                    axes[rows, columns].set_title(f"Cluster: {cluster}")
                    axes[rows, columns].vlines(0, 1, len(session.__dict__[stim_type].onset_frames), colors='r', linestyles='solid')
                    axes[rows, columns].set_xlim(xlim)
                
                # Remove the extra axes if there are no more plots
                else:
                    fig.delaxes(axes[rows, columns])
                
                plot_counter += 1
        
        # SAVE FIGURE
        fig.tight_layout()
        plt.savefig(save_path + "_single_cluster_raster_" + str(figure_idx) + ".png")                
    
    if show_plots: 
        plt.show()