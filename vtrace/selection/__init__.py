"""Fast backbone / head selection for TRACE.

Training-free transferability estimation (rank pretrained backbones on cached
frozen features) + cheap proxy-head selection. See
``model-selection-recipe.md (research notes, archived)``.
"""

from .transferability import (
    METHODS,
    gbc_score,
    hscore_score,
    logme_score,
    ncti_score,
    rankme_score,
    score_transferability,
    sfda_score,
    transrate_score,
)
from .lead import lead_score

METHODS = {**METHODS, "lead": lead_score}

__all__ = [
    "METHODS",
    "score_transferability",
    "logme_score",
    "hscore_score",
    "transrate_score",
    "gbc_score",
    "sfda_score",
    "ncti_score",
    "rankme_score",
    "lead_score",
]
