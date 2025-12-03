"""This script defines the EscapeTuning dataclass"""

from dataclasses import dataclass
from behave_analysis.analyze.EscapePattern.escape_pattern_utils import define_bin_edges

@dataclass(frozen=False)
class EscapeTuning:
    nbins: int
    tuning_var: str
    bin_edges: float = None
    savepath: str = ''
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
    y_fitted_shift: float = 0.0
    R_shift: float = 0.0
    fr_shift: float = 0.0
    params_shift: float = 0.0
    mat_shift_cond: float = 0.0
    loo_shift: float = 0.0

def init_escape_tuning(settings):
    return EscapeTuning(
        nbins=settings.escape_tuning_bins,
        tuning_var=settings.escape_tuning_var,
        bin_edges = define_bin_edges(settings)
    )
