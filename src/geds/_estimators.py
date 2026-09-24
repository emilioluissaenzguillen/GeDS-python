"""Scikit-learn-style estimators that delegate all statistics to R GeDS."""

from __future__ import annotations

from copy import copy
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
        if hasattr(self, "shape_constraints_"):
            raise NotImplementedError(
                "Standard coefficient confidence intervals are not available "
                "for shape-constrained fits."
            )
        selected_order = self.order if order is None else order
        self._validate_requested_order(selected_order)
        if not np.isfinite(level) or not 0 < level < 1:
            raise ValueError("level must be a finite number strictly between 0 and 1.")
        return get_backend().confidence_intervals(
            self._r_model_, selected_order, level
        )

    def _require_univariate_spline(self) -> None:
        check_is_fitted(self, "_r_model_")
        if len(self._spline_indices_) != 1 or self._linear_indices_:
            raise ValueError(
                "This operation requires a fitted univariate spline without "
                "additional linear features."
            )

    def derive(
        self, x: Any, *, derivative_order: int = 1, order: int | None = None
    ) -> np.ndarray:
        """Evaluate R ``Derive`` on the predictor scale at one or more x values."""
        self._require_univariate_spline()
        selected_order = self.order if order is None else order
        self._validate_requested_order(selected_order)
        if (not isinstance(derivative_order, (int, np.integer))
                or not 1 <= derivative_order < selected_order):
            raise ValueError("derivative_order must be an integer from 1 to order - 1.")
        values = np.atleast_1d(np.asarray(x, dtype=float))
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError("x must contain finite numeric values.")
        return get_backend().derive(
            self._r_model_, values, derivative_order, selected_order
        )

    def integrate(
        self, lower: Any, upper: Any, *, order: int | None = None
    ) -> np.ndarray:
        """Evaluate R ``Integrate`` on the predictor scale over given limits."""
        self._require_univariate_spline()
        selected_order = self.order if order is None else order
        self._validate_requested_order(selected_order)
        lower_values = np.atleast_1d(np.asarray(lower, dtype=float))
        upper_values = np.atleast_1d(np.asarray(upper, dtype=float))
        if lower_values.ndim != 1 or upper_values.ndim != 1 or not len(upper_values):
            raise ValueError("Integration limits must be scalars or one-dimensional.")
        if len(lower_values) not in (1, len(upper_values)):
            raise ValueError("lower must be scalar or have the same length as upper.")
        if np.isnan(lower_values).any() or np.isnan(upper_values).any():
            raise ValueError("Integration limits must not be NaN.")
        return get_backend().integrate(
            self._r_model_, lower_values, upper_values, selected_order
        )

    def piecewise_polynomial(
        self, *, order: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return R ``PPolyRep`` knots and polynomial coefficient matrix."""
        self._require_univariate_spline()
        selected_order = self.order if order is None else order
        self._validate_requested_order(selected_order)
        return get_backend().piecewise_polynomial(self._r_model_, selected_order)

    def shape_constrain(
        self, constraints: str | Sequence[str] = "increasing", *,
        order: int | None = None, base_learner: str | None = None,
        eps: float = 0.0, ridge: float = 1e-8,
    ) -> "_GeDSBase":
        """Return a new estimator constrained by R ``shapeConstrain``."""
        check_is_fitted(self, "_r_model_")
        selected_order = self.order if order is None else order
        self._validate_requested_order(selected_order)
        selected = [constraints] if isinstance(constraints, str) else list(constraints)
        allowed = {"increasing", "decreasing", "convex", "concave"}
        if not selected or len(selected) != len(set(selected)) or not set(selected) <= allowed:
            raise ValueError(f"constraints must be distinct values from {sorted(allowed)}.")
        if ({"increasing", "decreasing"} <= set(selected)
                or {"convex", "concave"} <= set(selected)):
            raise ValueError("Opposing shape constraints cannot be combined.")
        if not np.isfinite(eps) or eps < 0 or not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("eps must be finite and non-negative; ridge must be positive.")
        if hasattr(self, "_base_learner_lookup_"):
            if self.family.lower() != "gaussian" or self.normalize_data:
                raise ValueError("Shape constraints require Gaussian, unnormalized additive fits.")
            if base_learner is not None:
                if base_learner not in self._base_learner_lookup_:
                    raise ValueError(f"Unknown base learner {base_learner!r}.")
                internal_learner = self._base_learner_lookup_[base_learner]
                if base_learner not in self.base_learner_names_ or "," in internal_learner:
                    raise ValueError("base_learner must name a univariate spline term.")
            else:
                univariate = [name for name in self.base_learner_names_ if "," not in name]
                if len(univariate) != 1:
                    raise ValueError("Specify one univariate spline base_learner.")
                internal_learner = self._base_learner_lookup_[univariate[0]]
        else:
            if self._fit_function != "NGeDS" or len(self._spline_indices_) != 1:
                raise ValueError("Shape constraints require a univariate normal GeDS fit.")
            if base_learner is not None:
                raise ValueError("base_learner applies only to GAM and boosting fits.")
            internal_learner = None
        backend = get_backend()
        result = copy(self)
        result.order = selected_order
        constrained = backend.shape_constrain(
            self._r_model_, selected_order, selected, eps, ridge, internal_learner
        )
        result._finish_fit(constrained)
        result.shape_constraints_ = tuple(selected)
        return result

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


class _GeDSAdditiveBase(_GeDSBase):
    """Shared Python data handling for R's additive GAM and boosting fits."""

    def _validate_configuration(self) -> None:
        if self.order not in (2, 3, 4):
            raise ValueError("order must be 2, 3, or 4.")
        if not self.higher_order and self.order != 2:
            raise ValueError("order must be 2 when higher_order=False.")
        if not 0 <= self.beta <= 1 or not 0 < self.phi < 1:
            raise ValueError("beta must be in [0, 1] and phi in (0, 1).")
        if not isinstance(self.q, (int, np.integer)) or self.q < 1:
            raise ValueError("q must be a positive integer.")
        for name in ("min_iterations", "max_iterations"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, (int, np.integer)) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None.")
        if (self.min_iterations is not None and self.max_iterations is not None
                and self.min_iterations > self.max_iterations):
            raise ValueError("min_iterations must not exceed max_iterations.")
        if self.max_iterations == 0:
            raise ValueError("max_iterations must be positive when specified.")

    def _prepare_additive_data(
        self, X: Any, y: Any, sample_weight: Any
    ) -> tuple[pd.DataFrame, str, Any | None]:
        frame, named_input = self._frame(X)
        response = np.asarray(y, dtype=float)
        if response.ndim != 1 or len(response) != len(frame) or not np.isfinite(response).all():
            raise ValueError("y must contain one finite numeric value per row of X.")
        if frame.isna().to_numpy().any():
            raise ValueError("X must not contain missing values.")

        columns = list(frame.columns)
        linear = self._resolve(self.linear_features, columns, default=[])
        if self.spline_terms is None:
            terms = [[index] for index in range(len(columns)) if index not in linear]
        else:
            terms = [self._resolve(group, columns, default=[]) for group in self.spline_terms]
        if not terms or any(not group for group in terms):
            raise ValueError("spline_terms must contain at least one non-empty feature group.")
        spline = [index for group in terms for index in group]
        if len(spline) != len(set(spline)):
            raise ValueError("A feature may occur in only one spline term.")
        if set(spline).intersection(linear):
            raise ValueError("Spline and linear feature selections must not overlap.")
        non_numeric = [columns[index] for index in spline if not is_numeric_dtype(frame.iloc[:, index])]
        if non_numeric:
            raise TypeError(f"Spline features must be numeric; got {non_numeric}.")

        self.n_features_in_ = frame.shape[1]
        self._input_columns_ = columns
        self._named_input_ = named_input
        if named_input and all(isinstance(column, str) for column in columns):
            self.feature_names_in_ = np.asarray(columns, dtype=object)
        self._internal_columns_ = [f"x{index}" for index in range(len(columns))]
        self._spline_indices_ = spline
        self._linear_indices_ = linear
        self._has_offset_ = False
        self.spline_terms_ = [
            tuple(columns[index] for index in group) for group in terms
        ]
        self.spline_features_ = np.asarray([columns[index] for index in spline], dtype=object)
        self.linear_features_ = np.asarray([columns[index] for index in linear], dtype=object)

        internal = frame.copy()
        internal.columns = self._internal_columns_
        internal.insert(0, "response", response)
        self._r_base_learner_names_ = [
            "f(" + ", ".join(self._internal_columns_[index] for index in group) + ")"
            for group in terms
        ]
        self.base_learner_names_ = [
            "f(" + ", ".join(str(columns[index]) for index in group) + ")"
            for group in terms
        ]
        self._base_learner_lookup_ = dict(zip(self.base_learner_names_, self._r_base_learner_names_))
        self._base_learner_lookup_.update(
            {str(columns[index]): self._internal_columns_[index] for index in linear}
        )
        formula_terms = [*self._r_base_learner_names_, *[self._internal_columns_[i] for i in linear]]
        self.formula_ = "response ~ " + " + ".join(formula_terms)

        weights = None
        if sample_weight is not None:
            weight_array = np.asarray(sample_weight, dtype=float)
            if (weight_array.ndim != 1 or len(weight_array) != len(frame)
                    or not np.isfinite(weight_array).all() or np.any(weight_array < 0)):
                raise ValueError("sample_weight must contain one finite, non-negative value per row.")
            weights = get_backend().vector(weight_array)
        return internal, self.formula_, weights

    def predict_terms(self, X: Any, *, offset: Any = None) -> pd.DataFrame:
        raise NotImplementedError(
            "R's NGeDSgam/NGeDSboost predict method does not support type='terms'; "
            "use predict_component() for a named base learner."
        )

    def predict_component(
        self, X: Any, base_learner: str, *, prediction_type: str = "response"
    ) -> np.ndarray:
        """Return R's prediction for one named base learner (for example ``f(x)``)."""
        check_is_fitted(self, "_r_model_")
        if base_learner not in self._base_learner_lookup_:
            raise ValueError(f"Unknown base learner {base_learner!r}.")
        if prediction_type not in {"response", "link"}:
            raise ValueError("prediction_type must be 'response' or 'link'.")
        frame = self._prepare_new_data(X)
        backend = get_backend()
        return backend.predict(
            self._r_model_, backend.dataframe_to_r(frame), self.order,
            prediction_type, base_learner=self._base_learner_lookup_[base_learner],
        )


class GeDSGAMRegressor(_GeDSAdditiveBase):
    """Additive GeDS estimator backed entirely by ``GeDS::NGeDSgam``."""

    def __init__(
        self, *, family: str = "gaussian", link: str | None = None,
        spline_terms: Sequence[Sequence[str | int]] | None = None,
        linear_features: FeatureSelector = None, order: int = 3,
        normalize_data: bool = False, min_iterations: int | None = None,
        max_iterations: int | None = None, phi_gam_exit: float = 0.99,
        q_gam: int = 2, beta: float = 0.5, phi: float = 0.99,
        internal_knots: int = 500, q: int = 2, higher_order: bool = True,
    ) -> None:
        self.family = family
        self.link = link
        self.spline_terms = spline_terms
        self.linear_features = linear_features
        self.order = order
        self.normalize_data = normalize_data
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.phi_gam_exit = phi_gam_exit
        self.q_gam = q_gam
        self.beta = beta
        self.phi = phi
        self.internal_knots = internal_knots
        self.q = q
        self.higher_order = higher_order

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> "GeDSGAMRegressor":
        self._validate_configuration()
        if not 0 < self.phi_gam_exit < 1:
            raise ValueError("phi_gam_exit must lie in (0, 1).")
        if not isinstance(self.q_gam, (int, np.integer)) or self.q_gam < 1:
            raise ValueError("q_gam must be a positive integer.")
        if not isinstance(self.internal_knots, (int, np.integer)) or self.internal_knots < 0:
            raise ValueError("internal_knots must be a non-negative integer.")
        frame, formula, weights = self._prepare_additive_data(X, y, sample_weight)
        if self.family.lower() == "binomial":
            levels = np.unique(frame["response"])
            if len(levels) != 2 or not np.array_equal(levels, [0, 1]):
                raise ValueError("Binomial GAM responses must contain both 0 and 1.")
            frame["response"] = pd.Categorical(
                frame["response"].astype(int).astype(str), categories=["0", "1"]
            )
        backend = get_backend()
        kwargs: dict[str, Any] = {
            "family": backend.family(self.family, self.link),
            "normalize_data": self.normalize_data,
            "phi_gam_exit": self.phi_gam_exit, "q_gam": self.q_gam,
            "beta": self.beta, "phi": self.phi,
            "internal_knots": self.internal_knots, "q": self.q,
            "higher_order": self.higher_order,
        }
        for name, value in (("weights", weights), ("min_iterations", self.min_iterations),
                            ("max_iterations", self.max_iterations)):
            if value is not None:
                kwargs[name] = value
        model = backend.fit("NGeDSgam", backend.formula(formula), backend.dataframe_to_r(frame), **kwargs)
        self._finish_fit(model)
        return self


class GeDSBoostRegressor(_GeDSAdditiveBase):
    """Boosted GeDS estimator backed entirely by ``GeDS::NGeDSboost``."""

    def __init__(
        self, *, family: str = "gaussian", link: str | None = None,
        spline_terms: Sequence[Sequence[str | int]] | None = None,
        linear_features: FeatureSelector = None, order: int = 3,
        normalize_data: bool = False, initial_learner: bool = True,
        int_knots_init: int = 2, min_iterations: int | None = None,
        max_iterations: int | None = None, shrinkage: float = 1.0,
        phi_boost_exit: float = 0.99, q_boost: int = 2,
        beta: float = 0.5, phi: float = 0.99,
        int_knots_boost: int | None = None, q: int = 2,
        higher_order: bool = True, boosting_with_memory: bool = False,
    ) -> None:
        self.family = family
        self.link = link
        self.spline_terms = spline_terms
        self.linear_features = linear_features
        self.order = order
        self.normalize_data = normalize_data
        self.initial_learner = initial_learner
        self.int_knots_init = int_knots_init
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.shrinkage = shrinkage
        self.phi_boost_exit = phi_boost_exit
        self.q_boost = q_boost
        self.beta = beta
        self.phi = phi
        self.int_knots_boost = int_knots_boost
        self.q = q
        self.higher_order = higher_order
        self.boosting_with_memory = boosting_with_memory

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> "GeDSBoostRegressor":
        self._validate_configuration()
        if not 0 < self.shrinkage <= 1:
            raise ValueError("shrinkage must lie in (0, 1].")
        if not 0 < self.phi_boost_exit < 1:
            raise ValueError("phi_boost_exit must lie in (0, 1).")
        if not isinstance(self.q_boost, (int, np.integer)) or self.q_boost < 1:
            raise ValueError("q_boost must be a positive integer.")
        for name in ("int_knots_init", "int_knots_boost"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, (int, np.integer)) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None.")
        frame, formula, weights = self._prepare_additive_data(X, y, sample_weight)
        if self.family.lower() == "binomial" and not np.isin(frame["response"], [-1, 1]).all():
            raise ValueError("Binomial boosting responses must be encoded as -1 and 1.")
        backend = get_backend()
        kwargs: dict[str, Any] = {
            "family": backend.boost_family(self.family),
            "normalize_data": self.normalize_data,
            "initial_learner": self.initial_learner,
            "int.knots_init": self.int_knots_init,
            "shrinkage": self.shrinkage,
            "phi_boost_exit": self.phi_boost_exit, "q_boost": self.q_boost,
            "beta": self.beta, "phi": self.phi, "q": self.q,
            "higher_order": self.higher_order,
            "boosting_with_memory": self.boosting_with_memory,
        }
        if self.link is not None:
            kwargs["link"] = self.link
        for name, value in (("weights", weights), ("min_iterations", self.min_iterations),
                            ("max_iterations", self.max_iterations),
                            ("int.knots_boost", self.int_knots_boost)):
            if value is not None:
                kwargs[name] = value
        model = backend.fit("NGeDSboost", backend.formula(formula), backend.dataframe_to_r(frame), **kwargs)
        self._finish_fit(model)
        return self

    def get_base_learner_importance(
        self, *, boosting_iterations_only: bool = False
    ) -> pd.Series:
        """Return R ``bl_imp()`` in-bag risk reductions by base learner."""
        check_is_fitted(self, "_r_model_")
        importance = get_backend().base_learner_importance(
            self._r_model_, boosting_iterations_only
        )
        inverse = {value: key for key, value in self._base_learner_lookup_.items()}
        importance.index = [inverse.get(name, name) for name in importance.index]
        return importance

    def save_boosting_diagnostics(
        self, path: str | Path, *, iterations: Sequence[int] = (0,),
        final_fits: bool = False, overwrite: bool = False,
    ) -> Path:
        """Save R ``visualize_boosting()`` plots as a multipage PDF."""
        check_is_fitted(self, "_r_model_")
        if self.n_features_in_ != 1 or len(self.spline_terms_) != 1:
            raise ValueError(
                "R boosting visualization requires one univariate spline feature."
            )
        target = Path(path)
        if target.suffix.lower() != ".pdf":
            raise ValueError("path must name a PDF file.")
        if target.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {target}")
        selected = list(iterations)
        n_iterations = int(np.asarray(self.n_iter_).item())
        if not selected or any(
            not isinstance(value, (int, np.integer))
            or value < 0 or value > n_iterations for value in selected
        ):
            raise ValueError(
                "iterations must contain integers from 0 through n_iter_."
            )
        get_backend().save_boosting_diagnostics(
            self._r_model_, target, [int(value) for value in selected],
            final_fits,
        )
        return target
