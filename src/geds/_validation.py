"""Python inputs for R GeDS's specialized cross-validation routine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone

from ._backend import get_backend
from ._estimators import (
    GeDSBoostRegressor, GeDSGAMRegressor, GeDSGeneralizedRegressor,
    GeDSRegressor,
)


@dataclass(frozen=True)
class GeDSCrossValidationResult:
    """R's best parameter combination and full cross-validation table."""

    best_params: pd.DataFrame
    results: pd.DataFrame


def cross_validate_geds(
    estimator: GeDSRegressor | GeDSGeneralizedRegressor | GeDSGAMRegressor | GeDSBoostRegressor,
    X: Any,
    y: Any,
    parameter_grid: Mapping[str, Sequence[float]],
    *,
    n_folds: int = 5,
    n_cores: int = 1,
    random_state: int | None = None,
) -> GeDSCrossValidationResult:
    """Tune GeDS parameters with R ``crossv_GeDS()`` (Gaussian models only).

    This is R's specialized grid search, not scikit-learn cross-validation.
    It returns R's MSE and knot/iteration summary without fitting the input
    estimator. R currently applies its own fitting defaults for non-grid
    options in NGeDS, GGeDS, and NGeDSgam cross-validation.
    """
    if not isinstance(estimator, (
        GeDSRegressor, GeDSGeneralizedRegressor,
        GeDSGAMRegressor, GeDSBoostRegressor,
    )):
        raise TypeError("estimator must be a GeDS Python estimator.")
    if not isinstance(n_folds, (int, np.integer)) or n_folds < 2:
        raise ValueError("n_folds must be an integer of at least 2.")
    if not isinstance(n_cores, (int, np.integer)) or n_cores < 1:
        raise ValueError("n_cores must be a positive integer.")
    if random_state is not None and (
        not isinstance(random_state, (int, np.integer)) or random_state < 0
    ):
        raise ValueError("random_state must be a non-negative integer or None.")
    if hasattr(estimator, "family") and estimator.family.lower() != "gaussian":
        raise ValueError(
            "R crossv_GeDS currently uses the Gaussian default for these "
            "models; non-Gaussian estimators are not supported here."
        )
    if hasattr(estimator, "link") and estimator.link is not None:
        raise ValueError("A custom link is not supported by R crossv_GeDS.")

    # R's non-boost cross-validation routine only forwards beta, phi, q, and
    # the order-derived higher_order flag. Reject other custom settings so
    # they are never silently ignored.
    if not isinstance(estimator, GeDSBoostRegressor):
        defaults_estimator = type(estimator)()
        forwarded = {
            "spline_features", "spline_terms", "linear_features",
            "order", "higher_order", "beta", "phi", "q", "family", "link",
        }
        unsupported = [
            name for name, value in estimator.get_params(deep=False).items()
            if name not in forwarded
            and not np.array_equal(value, getattr(defaults_estimator, name))
        ]
        if unsupported:
            raise ValueError(
                "R crossv_GeDS does not forward these custom settings: "
                + ", ".join(sorted(unsupported))
            )

    working = clone(estimator)
    working._validate_configuration()
    if isinstance(working, (GeDSGAMRegressor, GeDSBoostRegressor)):
        frame, formula, _ = working._prepare_additive_data(X, y, None)
    else:
        frame, formula, _ = working._prepare_training_data(X, y, None, None)
    if n_folds > len(frame):
        raise ValueError("n_folds cannot exceed the number of observations.")

    defaults: dict[str, float] = {
        "beta": 0.5 if working.beta is None else float(working.beta),
        "phi": float(working.phi),
        "q": float(working.q),
    }
    if isinstance(working, GeDSBoostRegressor):
        defaults.update({
            "int_knots_init": float(working.int_knots_init),
            "shrinkage": float(working.shrinkage),
        })
    extra = set(parameter_grid) - set(defaults)
    if extra:
        raise ValueError(f"Unsupported parameter grid names: {sorted(extra)}.")
    parameters: dict[str, np.ndarray] = {}
    for name, default in defaults.items():
        values = np.asarray(parameter_grid.get(name, [default]), dtype=float)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError(f"{name} grid must contain finite numeric values.")
        if name in {"beta", "phi"} and np.any((values < 0) | (values > 1)):
            raise ValueError(f"{name} grid values must lie in [0, 1].")
        if name in {"q", "int_knots_init"} and (
            np.any(values != np.floor(values))
            or np.any(values < (1 if name == "q" else 0))
        ):
            raise ValueError(f"{name} grid values must be valid non-negative integers.")
        if name == "shrinkage" and np.any((values <= 0) | (values > 1)):
            raise ValueError("shrinkage grid values must lie in (0, 1].")
        r_name = "int.knots_init" if name == "int_knots_init" else name
        parameters[f"{r_name}_grid"] = values

    if isinstance(working, GeDSBoostRegressor):
        model_name = "NGeDSboost"
        fit_kwargs = {
            "max_iterations": working.max_iterations,
            "min_iterations": working.min_iterations,
            "normalize_data": working.normalize_data,
            "initial_learner": working.initial_learner,
            "int.knots_boost": working.int_knots_boost,
            "phi_boost_exit": working.phi_boost_exit,
            "q_boost": working.q_boost,
            "boosting_with_memory": working.boosting_with_memory,
        }
        fit_kwargs = {k: v for k, v in fit_kwargs.items() if v is not None}
    elif isinstance(working, GeDSGAMRegressor):
        model_name, fit_kwargs = "NGeDSgam", {}
    elif isinstance(working, GeDSGeneralizedRegressor):
        model_name, fit_kwargs = "GGeDS", {}
    else:
        model_name, fit_kwargs = "NGeDS", {}

    best, results = get_backend().cross_validate(
        model_name, formula, frame, parameters, working.order,
        int(n_folds), int(n_cores), random_state, **fit_kwargs,
    )
    return GeDSCrossValidationResult(best_params=best, results=results)
