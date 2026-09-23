"""Optional Python-native plotting helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted


def plot_fit(
    estimator: Any,
    X: Any,
    y: Any | None = None,
    *,
    ax: Any | None = None,
    grid_size: int = 500,
    show_knots: bool = True,
) -> Any:
    """Plot a fitted univariate GeDS model and return its Matplotlib axes.

    This helper only visualizes predictions already produced by GeDS; it does
    not implement any statistical calculation in Python.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            'Plotting requires Matplotlib; install "geds-python[plot]".'
        ) from exc

    check_is_fitted(estimator, "_r_model_")
    frame, named_input = estimator._frame(X)
    if frame.shape[1] != 1 or estimator.n_features_in_ != 1:
        raise ValueError("plot_fit supports fitted models with one feature only.")
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2.")
    values = np.asarray(frame.iloc[:, 0], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("X must contain only finite values.")
    grid_values = np.linspace(values.min(), values.max(), grid_size)
    if named_input:
        grid = pd.DataFrame({frame.columns[0]: grid_values})
    else:
        grid = grid_values.reshape(-1, 1)
    fitted = estimator.predict(grid)

    if ax is None:
        _, ax = plt.subplots()
    if y is not None:
        response = np.asarray(y, dtype=float)
        if response.ndim != 1 or len(response) != len(values):
            raise ValueError("y must be one-dimensional and have the same length as X.")
        ax.scatter(values, response, s=12, alpha=0.35, label="Data")
    ax.plot(grid_values, fitted, linewidth=2, label="GeDS fit")
    if show_knots and estimator.knots_ is not None:
        knots = np.asarray(estimator.knots_, dtype=float).ravel()
        for index, knot in enumerate(knots):
            ax.axvline(
                knot,
                color="tab:red",
                linestyle="--",
                alpha=0.55,
                label="Internal knots" if index == 0 else None,
            )
    ax.set(xlabel=str(frame.columns[0]), ylabel="Response", title="GeDS spline regression")
    ax.legend()
    return ax
