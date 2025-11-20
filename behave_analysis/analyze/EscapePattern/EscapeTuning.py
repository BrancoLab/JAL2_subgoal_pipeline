"""This script defines the EscapeTuning dataclass"""

from dataclasses import dataclass


@dataclass(frozen=False)
class EscapeTuning:
    nbins: int
    bin_edges: list
    savepath: str
    tuning_var: str
    fr_real: float
    params_real: float
    fr_shift: float
    params_shift: float
    fr_0shift: float
    params_0shift: float


def init_escape_tuning(settings):
    return EscapeTuning(
        nbins=settings.escape_pattern_nbins,
        savepath=settings.escape_pattern_savepath,
        tuning_var=settings.escape_pattern_tuning_var,
    )
