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
    installed = tuple(int(part) for part in info["geds_version"].split("."))
    assert installed >= (0, 3, 6, 9000)
    assert info["minimum_geds_version"] == "0.3.6.9000"


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
    assert prediction.flags.owndata
    assert estimator.coef_.flags.owndata
    assert estimator.knots_.flags.owndata
    np.testing.assert_allclose(
        prediction,
        [3.769455, 3.780928, 3.785291, 3.785762, 3.785793],
        atol=1e-6,
        rtol=0,
    )
    assert len(estimator.knots_) == 9
    assert estimator.deviance_ == pytest.approx(0.570885, abs=1e-6)
    assert estimator.get_deviance() == pytest.approx(estimator.deviance_)
    assert np.isfinite(estimator.get_log_likelihood())
    intervals = estimator.get_confidence_intervals()
    assert intervals.columns.tolist() == ["lower", "upper"]
    assert intervals.shape == (len(np.asarray(estimator.coef_)), 2)
    assert np.isfinite(intervals.to_numpy()).all()
    backend = geds._backend.get_backend()
    with backend.locked():
        direct_intervals = backend.stats.confint(
            estimator._r_model_, n=3, level=0.90
        )
        direct_names = list(backend.ro.r("rownames")(direct_intervals))
    intervals_90 = estimator.get_confidence_intervals(level=0.90)
    np.testing.assert_allclose(intervals_90.to_numpy(), np.asarray(direct_intervals))
    assert intervals_90.index.tolist() == direct_names
    with pytest.raises(ValueError, match="level must be"):
        estimator.get_confidence_intervals(level=1.0)
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
    assert np.isfinite(estimator.get_log_likelihood())
    assert np.isfinite(estimator.get_confidence_intervals().to_numpy()).all()

    backend = geds._backend.get_backend()
    r_training = pd.DataFrame({"response": frame["Y"], "x0": frame["X"]})
    r_model = backend.fit(
        "GGeDS",
        backend.formula("response ~ f(x0)"),
        backend.dataframe_to_r(r_training),
        family=backend.family("poisson", None),
        beta=0.2,
        phi=0.95,
        q=2,
        show_iters=False,
        stoptype="SR",
        higher_order=True,
    )
    r_newdata = backend.dataframe_to_r(r_training[["x0"]].iloc[:5])
    with backend.locked():
        r_response = np.asarray(
            backend.stats.predict(r_model, newdata=r_newdata, n=3, type="response")
        )
        r_link = np.asarray(
            backend.stats.predict(r_model, newdata=r_newdata, n=3, type="link")
        )
        r_log_likelihood = float(backend.stats.logLik(r_model, n=3)[0])
    np.testing.assert_allclose(estimator.predict(frame[["X"]].iloc[:5]), r_response)
    np.testing.assert_allclose(estimator.predict_link(frame[["X"]].iloc[:5]), r_link)
    assert estimator.get_log_likelihood() == pytest.approx(r_log_likelihood)


def test_univariate_offset_and_terms_match_r():
    x = np.linspace(-1.5, 1.5, 90)
    exposure = np.linspace(1.1, 2.0, len(x))
    offset = np.log(exposure)
    z = np.resize([-1.0, 0.0, 1.0], len(x))
    y = np.random.default_rng(321).poisson(
        exposure * np.exp(1.4 + np.sin(x) + 0.2 * z)
    ) + 1
    X = pd.DataFrame({"x": x, "z": z})
    backend = geds._backend.get_backend()
    estimator = GeDSGeneralizedRegressor(
        family="poisson", spline_features=["x"], linear_features=["z"],
        order=2, higher_order=False, beta=0.2, phi=0.9,
        max_internal_knots=4,
    ).fit(X, y, offset=offset)

    r_data = pd.DataFrame(
        {"response": y, "x0": x, "x1": z, "geds_offset": offset}
    )
    r_model = backend.fit(
        "GGeDS", backend.formula("response ~ f(x0) + x1 + offset(geds_offset)"),
        backend.dataframe_to_r(r_data), family=backend.family("poisson", None),
        beta=0.2, phi=0.9, q=2, show_iters=False, stoptype="SR",
        higher_order=False, max_intknots=4,
    )
    new_data = backend.dataframe_to_r(r_data[["x0", "x1", "geds_offset"]])
    with backend.locked():
        r_link = np.array(
            backend.stats.predict(r_model, newdata=new_data, n=2, type="link"),
            copy=True,
        )
        r_terms = backend.stats.predict(r_model, newdata=new_data, n=2, type="terms")
        r_term_values = np.array(r_terms, copy=True)
        r_term_names = [str(name) for name in backend.ro.r("colnames")(r_terms)]
    np.testing.assert_allclose(estimator.predict_link(X, offset=offset), r_link)
    np.testing.assert_allclose(estimator.predict(X, offset=offset), np.exp(r_link))
    terms = estimator.predict_terms(X, offset=offset)
    assert isinstance(terms, pd.DataFrame)
    assert terms.columns.tolist() == r_term_names
    np.testing.assert_allclose(terms.to_numpy(), r_term_values)
    np.testing.assert_allclose(
        terms.sum(axis=1).to_numpy() + offset, estimator.predict_link(X, offset=offset)
    )
    with pytest.raises(ValueError, match="requires offset"):
        estimator.predict(X)
    with pytest.raises(ValueError, match="one finite value"):
        estimator.predict(X, offset=offset[:-1])


def test_offset_rejects_bivariate_spline():
    x = np.linspace(-1, 1, 20)
    X = pd.DataFrame({"x": x, "z": x * x})
    with pytest.raises(ValueError, match="exactly one spline feature"):
        GeDSRegressor().fit(X, x, offset=np.zeros(len(x)))


def test_low_count_poisson_offset():
    x = np.linspace(-1.5, 1.5, 90)
    exposure = np.linspace(1.1, 2.0, len(x))
    z = np.resize([-1.0, 0.0, 1.0], len(x))
    y = np.random.default_rng(321).poisson(
        exposure * np.exp(0.4 + np.sin(x) + 0.2 * z)
    )
    X = pd.DataFrame({"x": x, "z": z})
    GeDSGeneralizedRegressor(
        family="poisson", spline_features=["x"], linear_features=["z"],
        order=2, higher_order=False, beta=0.2, phi=0.9,
        max_internal_knots=4,
    ).fit(X, y, offset=np.log(exposure))


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
    backend = geds._backend.get_backend()
    weights = np.linspace(0.5, 1.5, len(X))
    with backend.locked():
        backend.ro.r("set.seed(321)")
    prediction = estimator.fit(X, y, sample_weight=weights).predict(X)
    terms = estimator.predict_terms(X)
    assert terms.columns.tolist() == ["Spline", "x1"]
    np.testing.assert_allclose(terms.sum(axis=1), prediction)
    coefficients_before_r_fit = estimator.coef_.copy()
    assert prediction.shape == (60,)
    assert np.isfinite(prediction).all()
    np.testing.assert_allclose(estimator.predict(X[["z", "x"]]), prediction)

    # Fit the same weighted model through the R entry point, independently of
    # the Python estimator's fit/predict methods.
    r_training = pd.DataFrame(
        {"response": y.to_numpy(), "x0": X["x"], "x1": X["z"]}
    )
    with backend.locked():
        backend.ro.r("set.seed(321)")
    r_model = backend.fit(
        "NGeDS",
        backend.formula("response ~ f(x0) + x1"),
        backend.dataframe_to_r(r_training),
        beta=0.5,
        phi=0.9,
        q=2,
        show_iters=False,
        stoptype="RD",
        higher_order=True,
        weights=backend.vector(weights),
    )
    r_newdata = backend.dataframe_to_r(r_training[["x0", "x1"]])
    with backend.locked():
        r_prediction = np.asarray(
            backend.stats.predict(r_model, newdata=r_newdata, n=3, type="response")
        )
        r_coefficients = np.asarray(backend.stats.coef(r_model, n=3))
    np.testing.assert_allclose(prediction, r_prediction, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(estimator.coef_, coefficients_before_r_fit)
    np.testing.assert_allclose(estimator.coef_, r_coefficients, rtol=1e-10)


def test_weighted_poisson_matches_independent_r_fit():
    x = np.linspace(-1.5, 1.5, 80)
    y = np.random.default_rng(84).poisson(np.exp(0.4 + 0.7 * np.sin(2 * x)))
    weights = np.linspace(0.5, 2.0, len(x))
    X = pd.DataFrame({"x": x})
    backend = geds._backend.get_backend()
    with backend.locked():
        backend.ro.r("set.seed(678)")
    estimator = GeDSGeneralizedRegressor(
        family="poisson", order=2, higher_order=False,
        beta=0.2, phi=0.9, max_internal_knots=4,
    ).fit(X, y, sample_weight=weights)

    r_training = pd.DataFrame({"response": y, "x0": x})
    with backend.locked():
        backend.ro.r("set.seed(678)")
    r_model = backend.fit(
        "GGeDS",
        backend.formula("response ~ f(x0)"),
        backend.dataframe_to_r(r_training),
        family=backend.family("poisson", None),
        beta=0.2,
        phi=0.9,
        q=2,
        show_iters=False,
        stoptype="SR",
        higher_order=False,
        max_intknots=4,
        weights=backend.vector(weights),
    )
    r_newdata = backend.dataframe_to_r(r_training[["x0"]])
    with backend.locked():
        r_response = np.asarray(
            backend.stats.predict(r_model, newdata=r_newdata, n=2, type="response")
        )
        r_link = np.asarray(
            backend.stats.predict(r_model, newdata=r_newdata, n=2, type="link")
        )
        r_deviance = float(backend.stats.deviance(r_model, n=2)[0])
    np.testing.assert_allclose(estimator.predict(X), r_response, rtol=1e-10)
    np.testing.assert_allclose(estimator.predict_link(X), r_link, rtol=1e-10)
    assert estimator.get_deviance() == pytest.approx(r_deviance)


def test_bivariate_numpy_input():
    axis = np.linspace(-1, 1, 8)
    x1, x2 = np.meshgrid(axis, axis)
    X = np.column_stack((x1.ravel(), x2.ravel()))
    y = np.sin(2 * X[:, 0]) + np.cos(2 * X[:, 1])
    backend = geds._backend.get_backend()
    with backend.locked():
        backend.ro.r("set.seed(321)")
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
    assert np.isfinite(estimator.get_log_likelihood(order=2))

    r_training = pd.DataFrame(
        {"response": y, "x0": X[:, 0], "x1": X[:, 1]}
    )
    with backend.locked():
        backend.ro.r("set.seed(321)")
    r_model = backend.fit(
        "NGeDS",
        backend.formula("response ~ f(x0, x1)"),
        backend.dataframe_to_r(r_training),
        beta=0.5,
        phi=0.9,
        q=2,
        show_iters=False,
        stoptype="RD",
        higher_order=False,
        max_intknots=3,
    )
    r_newdata = backend.dataframe_to_r(r_training[["x0", "x1"]].iloc[:6])
    with backend.locked():
        r_prediction = np.asarray(
            backend.stats.predict(r_model, newdata=r_newdata, n=2, type="response")
        )
        r_log_likelihood = float(backend.stats.logLik(r_model, n=2)[0])
        r_intervals = backend.stats.confint(r_model, n=2, level=0.90)
        r_interval_names = list(backend.ro.r("rownames")(r_intervals))
    np.testing.assert_allclose(prediction, r_prediction, rtol=1e-10, atol=1e-10)
    assert estimator.get_log_likelihood(order=2) == pytest.approx(r_log_likelihood)
    intervals = estimator.get_confidence_intervals(order=2, level=0.90)
    assert np.isfinite(intervals.to_numpy()).any()
    np.testing.assert_allclose(
        intervals.to_numpy(), np.asarray(r_intervals), equal_nan=True
    )
    assert intervals.index.tolist() == r_interval_names


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
