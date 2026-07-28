from collections import namedtuple
import numpy as np

"""Set up params"""
def get_params(ks_dir, raw_file_path, meta_file_path, mouse):
    param_path = r"c:\Users\Jasmine\Documents\GitHub\JAL2_subgoal_pipeline\behave_analysis\postprocess\bc_process\bc_default_params_JAL008_10may.npy"
    param = np.load(param_path, allow_pickle=True).item()
    
    # general params:
    if mouse == "JAL003":
        param["probeType"] = 1  # NP1.0
    else:
        param["probeType"] = 2  # NP2.0
    param["reextractRaw"] = False

    JR_params = np.load(r"Z:\Jasmine_Laurence\bombcell\bombcell_params.npy", allow_pickle=True).item()
    # Update param with JR_params
    for key, value in JR_params.items():
        param[key] = value

    return param

def get_metric_info_dict_JR(param):
    """
    Constructs a list of metric information objects for spike sorting quality metrics.

    Each entry in the returned list is a namedtuple containing:
    - the metric's full name,
    - a human-readable short name,
    - one or two threshold values used for visualization or filtering,
    - a condition that determines whether the metric should be plotted,
    - and a predefined line color array for use in plots.

    The list of metrics is hard-coded and includes waveform, amplitude, noise,
    and clustering quality measures. Plot conditions are dynamically determined
    based on `param` flags and availability of data in `quality_metrics`.

    Parameters
    ----------
    param : dict
        Dictionary containing configuration parameters and thresholds
        used to determine which metrics to compute and display.

    quality_metrics : dict
        Dictionary of extracted quality metric values for all units.
        May include NaNs for metrics that are not computed.

    Returns
    -------
    dictionary of namedtuple
        Each namedtuple (MetricInfo) contains:
        - name (str): full metric name,
        - short_name (str): label for plotting
        - min_threshold (float or None): lower or upper threshold,
        - max_threshold (float or None): secondary threshold if applicable,


    """
    MetricInfo = namedtuple("MetricInfo", "name, short_name, metric_type, min_threshold, max_threshold")

    metric_info_list = [MetricInfo(name="nPeaks", short_name="# peaks", metric_type="noise", min_threshold=None, max_threshold=param.get("maxNPeaks")),

    MetricInfo(name="nTroughs", short_name="# troughs", metric_type="noise", min_threshold=None, max_threshold=param.get("maxNTroughs")),

    MetricInfo(name="waveformBaselineFlatness", short_name="baseline flatness", metric_type="noise", min_threshold=None, max_threshold=param.get("maxWvBaselineFraction")),

    MetricInfo(
        name="waveformDuration_peakTrough", short_name="waveform duration", metric_type="noise", min_threshold=param.get("minWvDuration"), max_threshold=param.get("maxWvDuration")
    ),

    MetricInfo(name="scndPeakToTroughRatio", short_name="peak_2/trough", metric_type="noise", min_threshold=None, max_threshold=param.get("maxScndPeakToTroughRatio_noise")),

    MetricInfo(
        name="spatialDecaySlope",
        short_name="spatial decay",
        metric_type="noise",
        min_threshold=param.get("minSpatialDecaySlope") if param.get("spDecayLinFit") else param.get("minSpatialDecaySlopeExp"),
        max_threshold=None if param.get("spDecayLinFit") else param.get("maxSpatialDecaySlopeExp"),
    ),

    MetricInfo(name="peak1ToPeak2Ratio", short_name="peak_1/peak_2", metric_type="nonsomatic", min_threshold=None, max_threshold=param.get("maxPeak1ToPeak2Ratio_nonSomatic")),

    MetricInfo(
        name="mainPeakToTroughRatio", short_name="peak_{main}/trough", metric_type="nonsomatic", min_threshold=None, max_threshold=param.get("maxMainPeakToTroughRatio_nonSomatic")
    ),

    MetricInfo(
        name="rawAmplitude",
        short_name="amplitude",
        metric_type="mua",
        min_threshold=param.get("minAmplitude"),
        max_threshold=None,
    ),

    MetricInfo(
        name="signalToNoiseRatio",
        short_name="signal/noise (SNR)",
        metric_type="mua",
        min_threshold=param.get("minSNR"),
        max_threshold=None,
    ),

    MetricInfo(name="fractionRPVs_estimatedTauR", short_name="refractory period viol. (RPV)", metric_type="mua", min_threshold=None, max_threshold=param.get("maxRPVviolations")),

    MetricInfo(name="nSpikes", short_name="# spikes", metric_type="mua", min_threshold=param.get("minNumSpikes"), max_threshold=None),

    MetricInfo(
        name="presenceRatio",
        short_name="presence ratio",
        metric_type="mua",
        min_threshold=param.get("minPresenceRatio"),
        max_threshold=None,
    ),

    MetricInfo(
        name="percentageSpikesMissing_gaussian",
        short_name="% spikes missing",
        metric_type="mua",
        min_threshold=None,
        max_threshold=param.get("maxPercSpikesMissing"),
    ),

    MetricInfo(
        name="maxDriftEstimate",
        short_name="maximum drift",
        metric_type="mua",
        min_threshold=None,
        max_threshold=param.get("maxDrift"),
    ),

    MetricInfo(
        name="isolationDistance",
        short_name="isolation dist.",
        metric_type="mua",
        min_threshold=param.get("isoDmin"),
        max_threshold=None,
    ),

    MetricInfo(
        name="Lratio",
        short_name="L-ratio",
        metric_type="mua",
        min_threshold=None,
        max_threshold=param.get("lratioMax"),
    )]

    return {mi.name: mi for mi in metric_info_list}

def get_waveform_peak_channel_safe(template_waveforms):
    # template_waveforms expected shape: n_templates x n_samples x n_channels
    n_templates = template_waveforms.shape[0]
    max_channels = np.zeros(n_templates, dtype=int)

    # Valid templates have at least one finite value
    valid = ~np.all(np.isnan(template_waveforms), axis=(1, 2))
    n_invalid = int((~valid).sum())
    if n_invalid > 0:
        print(f"BombCell patch: {n_invalid} all-NaN templates found; assigning fallback channel 0.")

    if np.any(valid):
        wv = template_waveforms[valid]
        max_value = np.nanmax(wv, axis=1)
        min_value = np.nanmin(wv, axis=1)
        ptp = max_value - min_value
        max_channels_valid = np.nanargmax(ptp, axis=1)
        max_channels[valid] = max_channels_valid

    # invalid templates remain channel 0 fallback
    return max_channels