"""Python access to the GeDS R package."""

from ._backend import BackendUnavailableError, diagnostics
from ._estimators import (
    GeDSBoostRegressor,
    GeDSGAMRegressor,
    GeDSGeneralizedRegressor,
    GeDSRegressor,
)
from ._plotting import plot_fit
from ._validation import GeDSCrossValidationResult, cross_validate_geds

__all__ = [
    "BackendUnavailableError",
    "GeDSBoostRegressor",
    "GeDSCrossValidationResult",
    "GeDSGAMRegressor",
    "GeDSGeneralizedRegressor",
    "GeDSRegressor",
    "diagnostics",
    "cross_validate_geds",
    "plot_fit",
]

__version__ = "0.1.0"
