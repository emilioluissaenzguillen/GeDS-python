"""Scikit-learn-style estimators that delegate all statistics to R GeDS."""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from ._backend import get_backend


FeatureSelector = Sequence[str | int] | None


class _GeDSBase(RegressorMixin, BaseEstimator):
    _fit_function = ""

    @staticmethod
    def _offset_array(offset: Any, length: int) -> np.ndarray:
        values = np.asarray(offset, dtype=float)
        if values.ndim != 1 or len(values) != length or not np.isfinite(values).all():
            raise ValueError("offset must contain one finite value per observation.")
        return values

    def _validate_configuration(self) -> None:
        if self.order not in (2, 3, 4):
            raise ValueError("order must be 2, 3, or 4.")
        if not self.higher_order and self.order != 2:
            raise ValueError("order must be 2 when higher_order=False.")
        if self.beta is not None and not 0 <= self.beta <= 1:
            raise ValueError("beta must lie in [0, 1].")
        if not 0 <= self.phi <= 1:
            raise ValueError("phi must lie in [0, 1].")
        if not isinstance(self.q, (int, np.integer)) or self.q < 1:
            raise ValueError("q must be a positive integer.")
        if self.stop_type not in {"SR", "RD", "LR"}:
            raise ValueError("stop_type must be 'SR', 'RD', or 'LR'.")
        for name in ("min_internal_knots", "max_internal_knots"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, (int, np.integer)) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None.")
        if (
            self.min_internal_knots is not None
            and self.max_internal_knots is not None
            and self.min_internal_knots > self.max_internal_knots
        ):
            raise ValueError(
                "min_internal_knots must not exceed max_internal_knots."
            )
        self._validate_range("x_range", self.x_range)
        self._validate_range("y_range", self.y_range)

    @staticmethod
    def _validate_range(name: str, value: Sequence[float] | None) -> None:
        if value is None:
            return
        array = np.asarray(value, dtype=float)
        if array.shape != (2,) or not np.isfinite(array).all() or array[0] >= array[1]:
            raise ValueError(
                f"{name} must contain two finite, strictly increasing values."
            )

    @staticmethod
    def _frame(X: Any) -> tuple[pd.DataFrame, bool]:
        if isinstance(X, pd.DataFrame):
            if X.columns.has_duplicates:
                raise ValueError("X must not contain duplicate column names.")
            return X.copy(), True
        array = np.asarray(X)
        if array.ndim != 2:
            raise ValueError("X must be a two-dimensional array or DataFrame.")
        return (
            pd.DataFrame(array, columns=[f"x{i}" for i in range(array.shape[1])]),
            False,
        )

    @staticmethod
    def _resolve(
        selectors: FeatureSelector, columns: list[Any], *, default: list[int]
    ) -> list[int]:
        if selectors is None:
            return default
        if isinstance(selectors, (str, int)):
            selectors = [selectors]
        indices: list[int] = []
        for selector in selectors:
            if isinstance(selector, str):
                if selector not in columns:
                    raise ValueError(f"Unknown feature {selector!r}.")
                index = columns.index(selector)
            elif isinstance(selector, (int, np.integer)):
                index = int(selector)
                if index < 0 or index >= len(columns):
                    raise ValueError(f"Feature index {index} is out of range.")
            else:
                raise TypeError("Feature selectors must be column names or indices.")
            if index not in indices:
                indices.append(index)
        return indices

    def _prepare_training_data(
        self, X: Any, y: Any, sample_weight: Any, offset: Any
    ) -> tuple[pd.DataFrame, str, Any | None]:
        frame, named_input = self._frame(X)
        y_array = np.asarray(y, dtype=float)
        if y_array.ndim != 1 or len(y_array) != len(frame):
            raise ValueError("y must be one-dimensional and have the same length as X.")
        if not np.isfinite(y_array).all():
            raise ValueError("y must contain only finite values.")

        columns = list(frame.columns)
        spline = self._resolve(
            self.spline_features, columns, default=list(range(len(columns)))
        )
        if not spline:
            raise ValueError("At least one spline feature is required.")
        if offset is not None and len(spline) != 1:
            raise ValueError("offset currently requires exactly one spline feature.")
        remainder = [index for index in range(len(columns)) if index not in spline]
        linear = self._resolve(self.linear_features, columns, default=remainder)
        overlap = sorted(set(spline).intersection(linear))
        if overlap:
            raise ValueError("Spline and linear feature selections must not overlap.")
        non_numeric = [
            columns[index]
            for index in spline
            if not is_numeric_dtype(frame.iloc[:, index])
        ]
        if non_numeric:
            raise TypeError(f"Spline features must be numeric; got {non_numeric}.")
        if frame.isna().to_numpy().any():
            raise ValueError("X must not contain missing values.")

        self.n_features_in_ = frame.shape[1]
        self._input_columns_ = columns
        self._named_input_ = named_input
        if named_input and all(isinstance(column, str) for column in columns):
            self.feature_names_in_ = np.asarray(columns, dtype=object)
        self._internal_columns_ = [f"x{index}" for index in range(len(columns))]
        self._spline_indices_ = spline
        self._linear_indices_ = linear
        self.spline_features_ = np.asarray(
            [columns[index] for index in spline], dtype=object
        )
        self.linear_features_ = np.asarray(
            [columns[index] for index in linear], dtype=object
        )

        internal = frame.copy()
        internal.columns = self._internal_columns_
        internal.insert(0, "response", y_array)
        spline_term = "f(" + ", ".join(self._internal_columns_[i] for i in spline) + ")"
        linear_terms = [self._internal_columns_[i] for i in linear]
        self._has_offset_ = offset is not None
        if self._has_offset_:
            internal["geds_offset"] = self._offset_array(offset, len(frame))
        formula_terms = [spline_term, *linear_terms]
        if self._has_offset_:
            formula_terms.append("offset(geds_offset)")
        formula = "response ~ " + " + ".join(formula_terms)
        self.formula_ = formula

        weights = None
        if sample_weight is not None:
            weight_array = np.asarray(sample_weight, dtype=float)
            if weight_array.ndim != 1 or len(weight_array) != len(frame):
                raise ValueError(
                    "sample_weight must be one-dimensional and match the length of X."
                )
            if not np.isfinite(weight_array).all() or np.any(weight_array < 0):
                raise ValueError(
                    "sample_weight must contain only finite, non-negative values."
                )
            weights = get_backend().vector(weight_array)
        return internal, formula, weights

    def _prepare_new_data(self, X: Any, offset: Any = None) -> pd.DataFrame:
        check_is_fitted(self, "_r_model_")
        frame, named_input = self._frame(X)
        if frame.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {frame.shape[1]} features; expected {self.n_features_in_}."
            )
        if self._named_input_:
            missing = [
                name for name in self._input_columns_ if name not in frame.columns
            ]
            if missing:
                raise ValueError(f"X is missing columns: {missing}.")
            frame = frame.loc[:, self._input_columns_]
        elif named_input:
            frame = frame.iloc[:, : self.n_features_in_]
        if frame.isna().to_numpy().any():
            raise ValueError("X must not contain missing values.")
        non_numeric = [
            self._input_columns_[index]
            for index in self._spline_indices_
            if not is_numeric_dtype(frame.iloc[:, index])
        ]
        if non_numeric:
            raise TypeError(f"Spline features must be numeric; got {non_numeric}.")
        frame.columns = self._internal_columns_
        if self._has_offset_:
            if offset is None:
                raise ValueError("This model requires offset for prediction.")
            frame["geds_offset"] = self._offset_array(offset, len(frame))
        elif offset is not None:
            raise ValueError("offset was not supplied when this model was fitted.")
        return frame

    def _common_fit_kwargs(self, weights: Any | None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "phi": self.phi,
            "q": self.q,
            "show_iters": self.verbose,
            "stoptype": self.stop_type,
            "higher_order": self.higher_order,
        }
        optional = {
            "beta": self.beta,
            "min_intknots": self.min_internal_knots,
            "max_intknots": self.max_internal_knots,
            "Xextr": (
                get_backend().vector(self.x_range) if self.x_range is not None else None
            ),
            "Yextr": (
                get_backend().vector(self.y_range) if self.y_range is not None else None
            ),
            "weights": weights,
        }
        kwargs.update(
            {key: value for key, value in optional.items() if value is not None}
        )
        return kwargs

    def _finish_fit(self, model: Any) -> None:
        backend = get_backend()
        self._r_model_ = model
        self.coef_ = backend.coefficients(model, self.order)
        self.knots_ = backend.knots(model, self.order)
        self.deviance_ = backend.deviance(model, self.order)
        self.n_iter_ = backend.component(model, "iters")

    def predict(self, X: Any, *, offset: Any = None) -> np.ndarray:
        frame = self._prepare_new_data(X, offset)
        backend = get_backend()
        return backend.predict(
            self._r_model_, backend.dataframe_to_r(frame), self.order, "response"
        )

    def predict_link(self, X: Any, *, offset: Any = None) -> np.ndarray:
        """Return predictions on the link scale."""
        frame = self._prepare_new_data(X, offset)
        backend = get_backend()
        return backend.predict(
            self._r_model_, backend.dataframe_to_r(frame), self.order, "link"
        )

    def predict_terms(self, X: Any, *, offset: Any = None) -> pd.DataFrame:
        """Return R's spline and parametric contributions on the link scale."""
        frame = self._prepare_new_data(X, offset)
        backend = get_backend()
        return backend.predict(
            self._r_model_, backend.dataframe_to_r(frame), self.order, "terms"
        )

    def get_coefficients(self, order: int | None = None) -> Any:
        check_is_fitted(self, "_r_model_")
        selected_order = self.order if order is None else order
        self._validate_requested_order(selected_order)
        return get_backend().coefficients(self._r_model_, selected_order)

    def get_knots(self, order: int | None = None) -> Any:
        check_is_fitted(self, "_r_model_")
        selected_order = self.order if order is None else order
        self._validate_requested_order(selected_order)
        return get_backend().knots(self._r_model_, selected_order)

    def get_deviance(self, order: int | None = None) -> float:
        """Return the R GeDS deviance for a selected spline order."""
        check_is_fitted(self, "_r_model_")
        selected_order = self.order if order is None else order
        self._validate_requested_order(selected_order)
        return get_backend().deviance(self._r_model_, selected_order)

    def get_log_likelihood(self, order: int | None = None) -> float:
        """Return the R GeDS log likelihood for a selected spline order."""
        check_is_fitted(self, "_r_model_")
        selected_order = self.order if order is None else order
        self._validate_requested_order(selected_order)
        return get_backend().log_likelihood(self._r_model_, selected_order)

    def get_confidence_intervals(
        self, order: int | None = None, *, level: float = 0.95
    ) -> pd.DataFrame:
        """Return R GeDS coefficient intervals with lower and upper columns."""
        check_is_fitted(self, "_r_model_")
        selected_order = self.order if order is None else order
        self._validate_requested_order(selected_order)
        if not np.isfinite(level) or not 0 < level < 1:
            raise ValueError("level must be a finite number strictly between 0 and 1.")
        return get_backend().confidence_intervals(
            self._r_model_, selected_order, level
        )

    def _validate_requested_order(self, order: int) -> None:
        if order not in (2, 3, 4):
            raise ValueError("order must be 2, 3, or 4.")
        if not self.higher_order and order != 2:
            raise ValueError("Only order 2 is available when higher_order=False.")

    def save(self, path: str | Path) -> None:
        """Serialize the estimator and its opaque R model with pickle."""
        check_is_fitted(self, "_r_model_")
        with Path(path).open("wb") as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "_GeDSBase":
        """Load an estimator saved by :meth:`save` from a trusted source."""
        with Path(path).open("rb") as stream:
            model = pickle.load(stream)
        if not isinstance(model, cls):
            raise TypeError(f"The file does not contain a {cls.__name__} estimator.")
        return model


class GeDSRegressor(_GeDSBase):
    """Normal-response GeDS estimator backed by ``GeDS::NGeDS``."""

    _fit_function = "NGeDS"

    def __init__(
        self,
        *,
        spline_features: FeatureSelector = None,
        linear_features: FeatureSelector = None,
        order: int = 3,
        beta: float = 0.5,
        phi: float = 0.99,
        min_internal_knots: int | None = None,
        max_internal_knots: int | None = None,
        q: int = 2,
        x_range: Sequence[float] | None = None,
        y_range: Sequence[float] | None = None,
        stop_type: str = "RD",
        higher_order: bool = True,
        verbose: bool = False,
    ) -> None:
        self.spline_features = spline_features
        self.linear_features = linear_features
        self.order = order
        self.beta = beta
        self.phi = phi
        self.min_internal_knots = min_internal_knots
        self.max_internal_knots = max_internal_knots
        self.q = q
        self.x_range = x_range
        self.y_range = y_range
        self.stop_type = stop_type
        self.higher_order = higher_order
        self.verbose = verbose

    def fit(
        self, X: Any, y: Any, sample_weight: Any = None, *, offset: Any = None
    ) -> "GeDSRegressor":
        self._validate_configuration()
        frame, formula, weights = self._prepare_training_data(X, y, sample_weight, offset)
        backend = get_backend()
        model = backend.fit(
            self._fit_function,
            backend.formula(formula),
            backend.dataframe_to_r(frame),
            **self._common_fit_kwargs(weights),
        )
        self._finish_fit(model)
        return self


class GeDSGeneralizedRegressor(_GeDSBase):
    """Exponential-family GeDS estimator backed by ``GeDS::GGeDS``."""

    _fit_function = "GGeDS"

    def __init__(
        self,
        *,
        family: str = "gaussian",
        link: str | None = None,
        spline_features: FeatureSelector = None,
        linear_features: FeatureSelector = None,
        order: int = 3,
        beta: float | None = None,
        phi: float = 0.99,
        min_internal_knots: int | None = None,
        max_internal_knots: int | None = None,
        q: int = 2,
        x_range: Sequence[float] | None = None,
        y_range: Sequence[float] | None = None,
        stop_type: str = "SR",
        higher_order: bool = True,
        verbose: bool = False,
    ) -> None:
        self.family = family
        self.link = link
        self.spline_features = spline_features
        self.linear_features = linear_features
        self.order = order
        self.beta = beta
        self.phi = phi
        self.min_internal_knots = min_internal_knots
        self.max_internal_knots = max_internal_knots
        self.q = q
        self.x_range = x_range
        self.y_range = y_range
        self.stop_type = stop_type
        self.higher_order = higher_order
        self.verbose = verbose

    def fit(
        self, X: Any, y: Any, sample_weight: Any = None, *, offset: Any = None
    ) -> "GeDSGeneralizedRegressor":
        self._validate_configuration()
        frame, formula, weights = self._prepare_training_data(X, y, sample_weight, offset)
        backend = get_backend()
        kwargs = self._common_fit_kwargs(weights)
        kwargs["family"] = backend.family(self.family, self.link)
        model = backend.fit(
            self._fit_function,
            backend.formula(formula),
            backend.dataframe_to_r(frame),
            **kwargs,
        )
        self._finish_fit(model)
        return self
