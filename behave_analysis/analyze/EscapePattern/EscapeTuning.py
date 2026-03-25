"""This script defines the EscapeTuning dataclass"""

from dataclasses import dataclass
from behave_analysis.analyze.EscapePattern.escape_pattern_utils import define_bin_edges, parse_side, parse_residual_string


@dataclass(frozen=False)
class EscapeTuning:
    name: str
    nbins: int
    tuning_var: str
    settings: object
    escape_pattern_time: str
    bin_edges: float = None
    savepath: str = ""
    homing_vector: float = 0.0
    escape_vector: float = 0.0
    explore_vector: float = 0.0
    neural_matrix: float = 0.0
    condition: float = 0.0
    discretized_var: float = 0.0
    # full tuning
    loo_reliability: float = 0.0
    R_full: float = 0.0
    y_fitted_full: float = 0.0
    fr_full: float = 0.0
    params_full: float = 0.0
    mat_num_cond: float = 0.0
    # shift tuning
    shifts: int = 0
    y_fitted_shift: float = 0.0
    R_shift: float = 0.0
    fr_shift: float = 0.0
    params_shifts: float = 0.0
    mat_shift_cond: float = 0.0
    loo_shift: float = 0.0
    # residual tuning
    residual_var2_all_time: float = 0.0


def init_escape_tuning(settings, tuning):

    if "residual" in tuning:
        tuning_var, escape_pattern_time, _, _ = parse_residual_string(tuning)
    else:
        tuning_var, escape_pattern_time = parse_side(tuning)

    return EscapeTuning(
        name=tuning,
        settings=settings,  # TODO: maybe we only want to save the EscapeTuning settings, not all the aefizz ones too about other methods
        tuning_var=tuning_var,
        escape_pattern_time=escape_pattern_time,
        nbins=settings.ep_bins,
        bin_edges=define_bin_edges(settings, tuning_var),
    )
