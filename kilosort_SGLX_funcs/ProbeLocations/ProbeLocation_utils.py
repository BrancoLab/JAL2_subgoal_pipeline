import colorsys
import numpy as np
from scipy.signal import find_peaks
import os
import re

#------------COLORMAPS------------

def reorder_colors_for_contrast(colors, n_colors_needed):
    # Create a function to reorder colors for maximum contrast between adjacent regions

    # If we need more colors than tab20b provides (20), we'll generate additional ones
    if n_colors_needed > len(colors):
        # Generate additional colors with HSV for better control
        additional_colors = generate_distinct_colors(n_colors_needed - len(colors))
        colors = list(colors) + additional_colors
    
    # Reordering strategy: take colors from different parts of the palette
    # to maximize distinction between adjacent values
    reordered = []
    positions = []
    
    # First, use every 4th color
    for i in range(0, n_colors_needed, 4):
        if i < n_colors_needed:
            positions.append(i)
    
    # Then fill in with every 4th+2 color
    for i in range(2, n_colors_needed, 4):
        if i < n_colors_needed and i not in positions:
            positions.append(i)
    
    # Then fill in with every 4th+1 color
    for i in range(1, n_colors_needed, 4):
        if i < n_colors_needed and i not in positions:
            positions.append(i)
    
    # Then fill in with every 4th+3 color
    for i in range(3, n_colors_needed, 4):
        if i < n_colors_needed and i not in positions:
            positions.append(i)
    
    # Create final reordered list
    positions = sorted(positions)
    for i in range(n_colors_needed):
        reordered.append(colors[positions.index(i)])
    
    return reordered

# Generate additional distinct colors if needed
def generate_distinct_colors(n):
    HSV_tuples = [(x/n, 0.8, 0.9) for x in range(n)]
    RGB_tuples = [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]
    return RGB_tuples

# -------------COMPUTE STATS-------------
def estimate_firing_rate(filtered_data, fs, threshold_std=4, refractory_period_ms=1):
    """
    Estimate firing rate from bandpass filtered data using threshold crossings
    
    Parameters:
    -----------
    filtered_data : 2D array (samples x channels)
        Bandpass filtered neural data
    fs : float
        Sampling rate in Hz
    threshold_std : float
        Threshold in standard deviations of the signal
    refractory_period_ms : float
        Refractory period in ms to avoid double counting
        
    Returns:
    --------
    firing_rates : array
        Estimated firing rates in Hz for each channel
    """
    n_channels = filtered_data.shape[1]
    firing_rates = np.zeros(n_channels)
    all_spike_times = []
    
    # Convert refractory period to samples
    ref_period_samples = int(refractory_period_ms * fs / 1000)
    
    for ch in range(n_channels):
        # Get channel data
        signal = filtered_data[:, ch]
        
        # Calculate threshold (negative threshold since spikes are negative deflections)
        threshold = -threshold_std * np.std(signal)
        
        # Find negative peaks (spikes are typically negative in extracellular recordings)
        peaks, _ = find_peaks(-signal, height=-threshold, distance=ref_period_samples)
        
        # Calculate firing rate (Hz)
        duration_sec = len(signal) / fs
        firing_rate = len(peaks) / duration_sec
        
        firing_rates[ch] = firing_rate
        all_spike_times.append(peaks)
    
    return firing_rates, all_spike_times

#------------UTILS------------

def extract_subject_and_date(path_string):
    # Normalize path separators in case of mixed slashes
    normalized_path = os.path.normpath(path_string)
    
    # Split the path into components
    parts = normalized_path.split(os.sep)
    
    # Extract the subject ID (JAL006)
    subject_id = parts[2]
    
    # Extract the survey date part (SvyPrb_2024-03-21)
    # Using regex to get everything before the "T" followed by time
    survey_date_match = re.match(r'(SvyPrb_\d{4}-\d{2}-\d{2})', parts[3])
    survey_date = survey_date_match.group(1) if survey_date_match else None
    
    return subject_id, survey_date