from __future__ import annotations

import pickle
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_regressor

import geds
import geds.check as check_module
from geds import GeDSGeneralizedRegressor, GeDSRegressor, plot_fit
from geds.check import main as check_main


def test_distribution_and_module_versions_match():
    assert geds.__version__ == version("geds-python")


def _r_reference_frame(expression: str) -> pd.DataFrame:
    backend = geds._backend.get_backend()
    with backend.locked(), backend._converter.context():
        value = backend.ro.r(expression)
        return backend.ro.conversion.get_conversion().rpy2py(value)


def test_diagnostics():
    info = geds.diagnostics()
    assert "R version 4." in info["r_version"]
    installed = tuple(int(part) for part in info["geds_version"].split(".")[:3])
    assert installed >= (0, 3, 6)
    assert info["minimum_geds_version"] == "0.3.6"


def test_environment_check(capsys):
    assert check_main([]) == 0
    assert "GeDS environment check: OK" in capsys.readouterr().out


def test_environment_check_json(capsys):
    assert check_main(["--json"]) == 0
    output = capsys.readouterr().out
    assert '"status": "ok"' in output
    assert '"geds_version"' in output


def test_environment_check_failure(monkeypatch, capsys):
    def unavailable():
        raise geds.BackendUnavailableError("R was not found.")

    monkeypatch.setattr(check_module, "diagnostics", unavailable)
    assert check_main([]) == 1
    error = capsys.readouterr().err
    assert "GeDS environment check: FAILED" in error
    assert "R_HOME" in error


def test_ngeds_reference_values_and_pickle(tmp_path):
    frame = _r_reference_frame(
        """
        set.seed(123)
        N <- 80
        X <- sort(runif(N, min=-2, max=2))
        Y <- (10*X/(1+100*X^2))*4+4+rnorm(N, sd=0.1)
        data.frame(X=X, Y=Y)
        """
    )
    estimator = GeDSRegressor(order=3, beta=0.6, phi=0.95).fit(
        frame[["X"]], frame["Y"].to_numpy().ravel()
    )
    prediction = estimator.predict(frame[["X"]].iloc[:5])
    np.testing.assert_allclose(
        prediction,
        [3.769455, 3.780928, 3.785291, 3.785762, 3.785793],
        atol=1e-6,
        rtol=0,
    )
    assert len(estimator.knots_) == 9
    assert estimator.deviance_ == pytest.approx(0.570885, abs=1e-6)
    restored = pickle.loads(pickle.dumps(estimator))
    np.testing.assert_array_equal(restored.predict(frame[["X"]].iloc[:5]), prediction)
    model_path = tmp_path / "model.pkl"
    estimator.save(model_path)
    loaded = GeDSRegressor.load(model_path)
    np.testing.assert_array_equal(loaded.predict(frame[["X"]].iloc[:5]), prediction)

    axes = plot_fit(estimator, frame[["X"]], frame["Y"], grid_size=50)
    assert axes.get_title() == "GeDS spline regression"
    assert len(axes.lines) == len(np.asarray(estimator.knots_).ravel()) + 1


def test_ggeds_poisson_reference_values():
    frame = _r_reference_frame(
        """
        set.seed(123)
        N <- 80
        X <- sort(runif(N, min=-2, max=2))
        eta <- sin(X)+1
        Y <- rpois(N, lambda=exp(eta))
        data.frame(X=X, Y=Y)
        """
    )
    estimator = GeDSGeneralizedRegressor(
        family="poisson", order=3, beta=0.2, phi=0.95
    ).fit(frame[["X"]], frame["Y"].to_numpy().ravel())
    np.testing.assert_allclose(
        estimator.predict(frame[["X"]].iloc[:5]),
        [0.469763, 0.529133, 0.576, 0.585782, 0.586556],
        atol=1e-6,
        rtol=0,
    )
    np.testing.assert_allclose(
        estimator.predict_link(frame[["X"]].iloc[:5]),
        [-0.755527, -0.636516, -0.551648, -0.534808, -0.533487],
        atol=1e-6,
        rtol=0,
    )


def test_numeric_parametric_component_and_sklearn_clone():
    X = pd.DataFrame(
        {
            "x": np.linspace(-1, 1, 60),
            "z": np.resize([-1.0, 0.0, 1.0], 60),
        }
    )
    y = np.sin(3 * X["x"]) + 0.25 * X["z"]
    estimator = GeDSRegressor(
        spline_features=["x"], linear_features=["z"], phi=0.9
    )
    cloned = clone(estimator)
    assert cloned.get_params() == estimator.get_params()
    assert is_regressor(estimator)
    prediction = estimator.fit(
        X, y, sample_weight=np.linspace(0.5, 1.5, len(X))
    ).predict(X)
    assert prediction.shape == (60,)
    assert np.isfinite(prediction).all()


def test_bivariate_numpy_input():
    axis = np.linspace(-1, 1, 8)
    x1, x2 = np.meshgrid(axis, axis)
    X = np.column_stack((x1.ravel(), x2.ravel()))
    y = np.sin(2 * X[:, 0]) + np.cos(2 * X[:, 1])
    estimator = GeDSRegressor(
        order=2,
        higher_order=False,
        phi=0.9,
        max_internal_knots=3,
    ).fit(X, y)
    prediction = estimator.predict(X[:6])
    assert prediction.shape == (6,)
    assert np.isfinite(prediction).all()
    assert isinstance(estimator.knots_, dict)


def test_categorical_parametric_feature():
    X = pd.DataFrame(
        {
            "x": np.linspace(-1, 1, 60),
            "group": pd.Categorical(np.resize(["a", "b", "c"], 60)),
        }
    )
    y = np.sin(3 * X["x"]) + X["group"].map(
        {"a": 0.0, "b": 0.3, "c": -0.2}
    ).astype(float)
    estimator = GeDSRegressor(
        spline_features=["x"], linear_features=["group"], phi=0.9
    ).fit(X, y)
    prediction = estimator.predict(X)
    assert prediction.shape == (60,)
    assert np.isfinite(prediction).all()
    assert estimator.formula_ == "response ~ f(x0) + x1"
    np.testing.assert_array_equal(estimator.spline_features_, ["x"])
    np.testing.assert_array_equal(estimator.linear_features_, ["group"])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"q": 0}, "q must be a positive integer"),
        ({"stop_type": "unknown"}, "stop_type must be"),
        (
            {"min_internal_knots": 3, "max_internal_knots": 2},
            "min_internal_knots must not exceed",
        ),
        ({"x_range": [1, 1]}, "x_range must contain two finite"),
    ],
)
def test_invalid_configuration_fails_before_calling_r(kwargs, message):
    with pytest.raises(ValueError, match=message):
        GeDSRegressor(**kwargs).fit([[0.0], [1.0]], [0.0, 1.0])


def test_invalid_training_data_fails_clearly():
    with pytest.raises(TypeError, match="Spline features must be numeric"):
        GeDSRegressor().fit(pd.DataFrame({"group": ["a", "b"]}), [0.0, 1.0])

    with pytest.raises(ValueError, match="y must contain only finite"):
        GeDSRegressor().fit([[0.0], [1.0]], [0.0, np.nan])

    with pytest.raises(ValueError, match="sample_weight must contain"):
        GeDSRegressor().fit(
            [[0.0], [1.0]], [0.0, 1.0], sample_weight=[1.0, -1.0]
        )


def test_requested_order_is_validated():
    X = np.linspace(-1, 1, 30).reshape(-1, 1)
    estimator = GeDSRegressor(order=2, higher_order=False, phi=0.9).fit(
        X, np.sin(X[:, 0])
    )
    with pytest.raises(ValueError, match="Only order 2"):
        estimator.get_knots(order=3)
    with pytest.raises(ValueError, match="order must be"):
        estimator.get_coefficients(order=0)
