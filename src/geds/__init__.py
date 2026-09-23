"""Python access to the GeDS R package."""

from ._backend import BackendUnavailableError, diagnostics
from ._estimators import GeDSGeneralizedRegressor, GeDSRegressor
from ._plotting import plot_fit

__all__ = [
    "BackendUnavailableError",
    "GeDSGeneralizedRegressor",
    "GeDSRegressor",
    "diagnostics",
    "plot_fit",
]

__version__ = "0.1.0a4"
