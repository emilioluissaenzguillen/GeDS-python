"""Python access to the GeDS R package."""

from ._backend import BackendUnavailableError, diagnostics
from ._estimators import GeDSGeneralizedRegressor, GeDSRegressor

__all__ = [
    "BackendUnavailableError",
    "GeDSGeneralizedRegressor",
    "GeDSRegressor",
    "diagnostics",
]

__version__ = "0.1.0a1"
