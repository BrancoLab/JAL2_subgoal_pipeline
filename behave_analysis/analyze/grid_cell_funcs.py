"""This script contains the logic to generate grid cell related plots such as rate maps, 
spatial autocorrelations and more. Might also include head direction calculations etc."""

# OS Libaries
from matplotlib.backends.backend_pdf import PdfPages # For saving to a pdf
import pickle
from ephysiopy.common.binning import RateMap # Robin's rate map function
import numpy as np 
from loguru import logger
import matplotlib.pyplot as plt
from ephysiopy.common import gridcell

# ---------------------------------- High level plot function ------------------

def mother_plot(self):
    """A mother plot for generating multiple charts. The self instance refers
    to the Analyze class and carries with it all the required attributes to plot
    some grid cells.
    
    This function saves to a pdf for each cluster:
    + Rate map
    + SAC
    
    To do:
    + spike overlay to trajectory
    + head direction
    
    Return: 
    + pdf
    
    Notes:
    self instance is class Analyze from analyze.py
    """
    
    pdf_obj = PdfPages('NoShelterThenShelter_22MAY31.pdf') # Location of pdf, change name
    process_spike_dic = process_overlay_spikes(self) #
    file = open(self.interp_path, "rb")  # Load interpolated data
    interp_dic = pickle.load(file) # x, y, speed
    pos_data = np.vstack((process_spike_dic["x"],
                          process_spike_dic["y"])) # make the right shape for plotting

    # Good clusters taken from spike sorting, multi unit and unsure - could refactor somehow
    good_clusters = [218, 230, 231, 232, 241, 242, 247, 259, 260, 263, 264, 272, 275, 276, 282, 286, 287, 291, 292, 298, 300, 301, 307, 312, 314, 315, 316, 320, 327, 340, 343, 352, 354, 357, 360, 371, 372, 381, 382, 387, 389, 396, 398, 405, 406, 413, 416, 418, 420, 422, 429, 437, 439, 440, 441, 442, 443, 445, 447, 455, 461, 463, 469, 475, 496, 497, 505, 531, 551, 560, 569, 581, 591, 378, 400, 411, 419, 557, 210, 215, 216, 226, 277, 289, 290, 302, 313, 320, 332, 333, 334, 335, 344, 346, 347, 351, 353, 356, 361, 367, 368, 374, 375, 382, 388, 397, 436, 480, 517, 523, 540, 564, 566]
    
    for cluster_id in good_clusters:
        cluster_id -= 1 # Remove one index as matlab starts a 1 and python starts at 0
        fig = plt.figure()
        create_rate_map(self,
                        process_spike_dic, 
                        cluster_id,
                        fig,
                        pos_data)
        
        create_sac(self,
                   cluster_id,
                   pos_data,
                   fig,
                   process_spike_dic)
        pdf_obj.savefig(fig)
    
    pdf_obj.close()
    
# ---------------------------------- Individual plot functions ------------------

def create_rate_map(self,
                    process_spike_dic, 
                    cluster_id,
                    fig,
                    pos_data):
    
    """This function uses functionality from the Barry lab to produce rate maps

    Returns:
        obj: A rate map added to the pdf obj in mother plot
    """
    
    if not pos_data.size or not process_spike_dic["spike_times"].size:
        return # Don't plot empty cells as it causes errors down the line

    rate_map_class = RateMap(xy = pos_data,
                             ppm = 1000,
                             smooth_sz = 5,
                             cmsPerBin = 30) # Hyper parameters for changing the rate map

    cluster_type = self.session.ephys.annotations[cluster_id] # Retrieve annotation for that specific cluster
    new_mask = self.create_cluster_specific_mask(cluster_id, 
                                                 process_spike_dic["len_bon"]
                                                 ) # filter the spike mask by cluster ID
    new_mask = np.where(process_spike_dic["speed_idx"], 
                        new_mask, 
                        0
                        ) # remove spikes where mouse was not moving 5cm, by returning 0

    rmap = rate_map_class.getMap(spkWeights = new_mask)
    ratemap = np.ma.MaskedArray(rmap[0], np.isnan(rmap[0]), copy=True)
    x, y = np.meshgrid(rmap[1][1][0:-1], rmap[1][0][0:-1])
    vmax = np.nanmax(np.ravel(ratemap))
    
    logger.info(f"Plotting cluster ID: {cluster_id}")
    ax = fig.add_subplot(221)
    mesh = ax.pcolormesh(x, y, ratemap, cmap=plt.cm.get_cmap("jet"), edgecolors='face', vmax=vmax, shading='auto')
    ax.set_aspect('equal')
    ax.set_title(f"Cluster ID: {cluster_id}\ntype: {cluster_type}")
    
    return fig

def create_sac(self, 
               cluster_id,
               pos_data,
               fig,
               process_spike_dic):
    """Generates a SAC for a given cluster and returns a fig to mother plot

    Returns:
        obj: A SAC added to the pdf obj in mother plot
    """

    # Don't plot empty cells as it causes errors down the line
    if not pos_data.size or not process_spike_dic["spike_times"].size:
        return

    # Define classes
    rate_map_class = RateMap(xy = pos_data,
                             ppm = 1000,
                             smooth_sz = 5,
                             cmsPerBin = 30) # also hyper parameters for SAC change if changed rate map
    
    # Extract annotation
    cluster_type = self.session.ephys.annotations[cluster_id]

    # Print which cluster your on
    logger.info(f"Plotting cluster ID: {cluster_id}")
    
    # filter the spike mask by cluster ID
    new_mask = self.create_cluster_specific_mask(cluster_id, 
                                                 process_spike_dic["len_bon"])
    new_mask = np.where(process_spike_dic["speed_idx"], 
                        new_mask, 
                        0
                        ) # remove spikes where mouse was not moving 5cm, by returning 0

    rmap, _ = rate_map_class.getMap(spkWeights = new_mask) # compute rate map 
    S = gridcell.SAC()
    nodwell = ~np.isfinite(rmap)
    sac = S.autoCorr2D(rmap, nodwell)
    try:
        measures = S.getMeasures(sac) # Some clusterings worryinly don't work trying a hack to fix by moving to next cluster
    except ValueError as error:
        logger.info(f"An error was encountered of type: {error} - Leaving function and move to next cluster")
        return
        
    grid_score = measures["gridscore"] 
    ax = fig.add_subplot(222)
    ax = show_SAC(sac, measures, ax) # Plot SAC
    ax.set_title(f"Cluster ID: {cluster_id}\n grid score: {grid_score}\n cluster type: {cluster_type}")
    
def show_SAC(A: np.array, 
             inDict: dict, 
             ax: plt.axes=None, 
             **kwargs) -> plt.axes:
    """
    Code from barry lab
    
    Displays the result of performing a spatial autocorrelation (SAC)
    on a grid cell.
    Uses the dictionary containing measures of the grid cell SAC to
    make a pretty picture
    Parameters
    ----------
    A : array_like
        The spatial autocorrelogram
    inDict : dict
        The dictionary calculated in getmeasures
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        If given the plot will get drawn in these axes. Default None
    Returns
    -------
    fig : matplotlib.Figure instance
        The Figure on which the SAC is shown
    See Also
    --------
    ephysiopy.common.binning.RateMap.autoCorr2D()
    ephysiopy.common.ephys_generic.FieldCalcs.getMeaures()
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
    Am = A.copy()
    Am[~inDict['dist_to_centre']] = np.nan
    Am = np.ma.masked_invalid(np.atleast_2d(Am))
    x, y = np.meshgrid(np.arange(0, np.shape(A)[1]), np.arange(0, np.shape(A)[0]))
    vmax = np.nanmax(np.ravel(A))
    ax.pcolormesh(x, y, A, cmap=plt.cm.get_cmap("gray_r"), edgecolors='face', vmax=vmax, shading='auto')
    import copy
    cmap = copy.copy(plt.cm.get_cmap("jet"))
    cmap.set_bad('w', 0)
    ax.pcolormesh(x, y, Am, cmap=cmap, edgecolors='face', vmax=vmax, shading='auto')
    # horizontal green line at 3 o'clock
    _y = (np.shape(A)[0]/2, np.shape(A)[0]/2)
    _x = (np.shape(A)[1]/2, np.shape(A)[0])
    ax.plot(_x, _y, c='g')
    mag = inDict['scale'] * 0.5
    th = np.linspace(0, inDict['orientation'], 50)
    from ephysiopy.common.utils import rect
    [x, y] = rect(mag, th, deg=1)
    # angle subtended by orientation
    ax.plot( x + (inDict['dist_to_centre'].shape[1] / 2), (inDict['dist_to_centre'].shape[0] / 2) - y, 'r', **kwargs)
    
    # plot lines from centre to peaks above middle
    for p in inDict['closest_peak_coords']:
        if p[0] <= inDict['dist_to_centre'].shape[0] / 2:
            ax.plot(
                (inDict['dist_to_centre'].shape[1]/2, p[1]),
                (inDict['dist_to_centre'].shape[0] / 2, p[0]), 'k', **kwargs)
    ax.invert_yaxis()
    all_ax = ax.axes
    all_ax.set_aspect('equal')
    all_ax.set_xlim((0.5, inDict['dist_to_centre'].shape[1]-1.5))
    all_ax.set_ylim((inDict['dist_to_centre'].shape[0]-.5, -.5))
    
    return ax

# ---------------------------------- Lower level preprocessing functions

def process_overlay_spikes(self):
    
    """A function that returns a whole lot of things to preprocess spikes matched to trajectory.

    Returns:
        _type_: _description_
    """

    # Retrieve the ephys data
    spike_times = self.session.ephys.spike_times
    cluster_ids = self.session.ephys.cluster_ids
    spike_mask  = self.session.ephys.spike_mask

    # len of bonsai
    len_bon = len(self.session.ttl.bonsai_TTL)

    #Retrieve the interpolated positional and speed data
    file = open(self.interp_path, "rb") 
    interp_dic = pickle.load(file) # x, y, speed

    #cut off ends
    x = interp_dic['x'][:len_bon]
    y = interp_dic['y'][:len_bon]
    speed = interp_dic['speed'][:len_bon]
    spike_mask = spike_mask[:len_bon]

    # Assertions
    assert len(x) == len(self.session.ttl.bonsai_TTL), "The interpolated x data should match the length of the bonsai signal"
    assert len(y) == len(self.session.ttl.bonsai_TTL), "The interpolated y  data should match the length of the bonsai signal"
    assert len(speed) == len(self.session.ttl.bonsai_TTL), "The interpolated speed data should match the length of the bonsai signal"
    assert len(spike_mask) == len(self.session.ttl.bonsai_TTL), "The spike mask data should match the length of the bonsai signal"

    # Filter data where speed is above 5 - Currently not used
    bool_idx_speed_threshold = speed > 2.5 # What indexes does speed go over 5cm^2

    # create bins
    bins = np.arange(80, 1000, 2) # From 80 to 1000 based on the coordinates of the camera, will have to change if camera moves
    X_bin_dex = np.digitize(x, bins) # take x position and ascribe a bin to it
    Y_bin_dex = np.digitize(y, bins)  # take y position and ascribe a bin to it
    
    # process spike_dic
    process_spike_dic = {"spike_times": spike_times,
                        "cluster_ids": cluster_ids,
                        "spike_mask": spike_mask,
                        "bins": bins,
                        "X_bin_dex": X_bin_dex, # take x position and ascribe a bin to it
                        "Y_bin_dex": Y_bin_dex, # take y position and ascribe a bin to it
                        "speed_idx": bool_idx_speed_threshold, # the spike idxs that have met the threshold
                        "len_bon": len_bon, # the length of bonsai signal
                        "x": x,
                        "y": y,
                        "speed": speed
                        }

    return process_spike_dic
        