# GeDS for Python

This package provides a Python interface to the
[GeDS R package](https://github.com/emilioluissaenzguillen/GeDS). The R package
is the sole implementation of the statistical methods. Python supplies a
scikit-learn-style API, pandas/NumPy conversion, environment diagnostics, and
model serialization.

## Requirements

- R 4.4 or newer (R 4.6.1 is used for development)
- GeDS 0.3.6 or newer with the Python bridge fixes
- Python 3.10 or newer

Install the Python package, including the optional plotting dependency used in
the example:

```console
python -m pip install "geds-python[plot]"
```

Install the R package separately, using R 4.6.1 or another supported R
installation. Install the tested GeDS 0.3.6 build from GitHub:

```r
install.packages("remotes")
remotes::install_git(
  "https://github.com/emilioluissaenzguillen/GeDS.git",
  ref = "91b8ddd13aae8f39994c87fc356f05da4799f911",
  dependencies = NA, upgrade = "never"
)
```

An older GeDS build, even one reporting version `0.3.6`, may lack the fixes
required by this wrapper. `install.packages("GeDS")` alone is not guaranteed
to provide them while the CRAN review follows its separate schedule.
Check the installed R version with `packageVersion("GeDS")`.

On Windows, building the GitHub source package requires Rtools compatible
with the selected R installation. GeDS remains version `0.3.6` on GitHub;
the wrapper checks an internal compatibility marker for the fit and prediction
fixes as well as the package version.

The wrapper discovers the newest R installation under `Program Files/R` on
Windows or uses `Rscript` from `PATH` on other platforms. Set `R_HOME` to select
a particular R installation. If GeDS is installed in a non-default R library,
set `GEDS_R_LIBRARY` to that library directory before importing `geds`.

The Python and R packages have independent release cycles. `geds-python`
checks the installed GeDS version when its backend first starts and reports the
selected R installation and package library through `geds.diagnostics()`.

Check the backend before fitting:

```console
python -m geds.check
```

For a machine-readable report, use `python -m geds.check --json`. The same
information is available inside Python:

```python
import geds

print(geds.diagnostics())
```

### Selecting R and its package library

Usually no configuration is necessary. If several R installations are
available, select one before starting Python:

```powershell
$env:R_HOME = "C:\Program Files\R\R-4.6.1"
python -m geds.check
```

```bash
export R_HOME="/Library/Frameworks/R.framework/Resources"  # macOS
# export R_HOME="/usr/lib/R"                               # Linux
python -m geds.check
```

If GeDS is installed in a personal or otherwise non-default R library, set
`GEDS_R_LIBRARY` to the directory that contains the `GeDS` folder. You can
find that directory from R with `find.package("GeDS")`; use its parent
directory as `GEDS_R_LIBRARY`.

If the check reports that R is missing, install R or set `R_HOME`. If it finds
R but not GeDS, start that same R installation and run
one of the GeDS installation commands above, then rerun the check.

## Example

Install the optional plotting dependency with
`python -m pip install "geds-python[plot]"`, then fit and visualize a nonlinear
regression:

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geds import GeDSRegressor, plot_fit

rng = np.random.RandomState(123)
n = 500


def f_1(x):
    return (10 * x / (1 + 100 * x**2)) * 4 + 4


x = np.sort(rng.uniform(-2.0, 2.0, size=n))
means = f_1(x)
y = rng.normal(means, scale=0.1)
X = pd.DataFrame({"x": x})

model = GeDSRegressor(order=3).fit(X, y)
knots = np.asarray(model.knots_, dtype=float)

print("Internal knots:", knots)

fig, ax = plt.subplots()
plot_fit(model, X, y, ax=ax)
grid_x = np.linspace(x.min(), x.max(), 500)
ax.plot(
    grid_x,
    f_1(grid_x),
    color="0.25",
    linestyle=":",
    linewidth=2,
    label="True mean",
)
ax.set(ylabel="y")
ax.legend()
fig.tight_layout()
plt.show()
```

With GeDS 0.3.6 and R 4.6.1, this seeded example fits 16 internal knots.
The dashed vertical lines show how GeDS places more knots around the sharp
variation near zero while retaining knots across the wider domain.

`GeDSRegressor` delegates to `GeDS::NGeDS()`. For exponential-family models,
use `GeDSGeneralizedRegressor`, which delegates to `GeDS::GGeDS()`.

For fitted models, `get_deviance(order=...)`, `get_log_likelihood(order=...)`,
and `get_confidence_intervals(order=..., level=...)` call the corresponding R
methods. Confidence intervals are returned as a pandas DataFrame with `lower`
and `upper` columns. As in R, these are coefficient intervals, not confidence
bands for the fitted curve.

The estimators also work with standard scikit-learn tools such as
`cross_val_score()` and `GridSearchCV`. Use sequential execution (`n_jobs=1`)
when cross-validating: the wrapper embeds R in the Python process, and
parallel-worker behavior is not part of the supported interface.

R also has a specialized `crossv_GeDS()` routine, which returns a parameter
grid with cross-validated mean squared error and knot/iteration summaries.
Use its Python interface when those R-specific results are needed:

```python
from geds import cross_validate_geds

cv = cross_validate_geds(
    GeDSRegressor(order=3), X, y,
    {"beta": [0.5, 0.7], "phi": [0.95], "q": [2]},
    n_folds=5, n_cores=1, random_state=123,
)
print(cv.best_params)
print(cv.results)
```

This delegates the entire search to R and does not fit or change the input
estimator. It currently supports Gaussian models only, accepts the R tuning
parameters `beta`, `phi`, `q`, and (for boosting) `int_knots_init` and
`shrinkage`, and defaults to one R worker. R's current non-boost routine does
not forward other fitting settings; the Python interface rejects custom
settings it would otherwise silently ignore. Use scikit-learn's grid search
when you need those settings or a non-Gaussian family.

For a fitted univariate spline without extra linear features, R's calculus
and spline-conversion utilities are available as model methods:

```python
slopes = model.derive([-0.5, 0.0, 0.5], derivative_order=1)
areas = model.integrate(-1.0, [-0.5, 0.0, 0.5])
piece_knots, piece_coefficients = model.piecewise_polynomial()
```

`derive()` and `integrate()` operate on the predictor (link) scale, as in R.
`piecewise_polynomial()` returns the R `PPolyRep()` knot vector and coefficient
matrix; its last coefficient row is extraneous in R's representation. These
methods use the estimator's selected spline order unless `order=` is given.

For a normal univariate fit, impose a shape constraint without changing the
original fitted model:

```python
increasing_model = model.shape_constrain("increasing")
increasing_and_convex = model.shape_constrain(["increasing", "convex"])
```

This calls R's `shapeConstrain()` and returns a new Python estimator. R also
supports constraints on one selected univariate smoother in Gaussian GAM and
boosting fits, via `shape_constrain(..., base_learner="f(x)")`. Those additive
fits must use `normalize_data=False`. Constrained fits do not provide the usual
unconstrained coefficient confidence intervals.

For count data, the generalized estimator uses `GeDS::GGeDS()` and supports
both response-scale and link-scale prediction:

```python
import numpy as np
import pandas as pd
from geds import GeDSGeneralizedRegressor, plot_fit

rng = np.random.default_rng(123)
x = np.sort(rng.uniform(-2, 2, 120))
X = pd.DataFrame({"x": x})
counts = rng.poisson(np.exp(1 + np.sin(x)))

model = GeDSGeneralizedRegressor(
    family="poisson", beta=0.2, phi=0.95, min_internal_knots=3
).fit(X, counts)
mean_counts = model.predict(X)
log_mean_counts = model.predict_link(X)
ax = plot_fit(model, X, counts)
```

`min_internal_knots` controls the minimum number of stage-A knots; it is used
here to make a small sample's fitted spline visible. GeDS determines the
final knot positions.

For a univariate spline with a known offset (for example log exposure in a
Poisson model), pass one offset value per observation to both fitting and
prediction. These values are on the link scale:

```python
import numpy as np
import pandas as pd
from geds import GeDSGeneralizedRegressor

rng = np.random.default_rng(321)
x = np.linspace(-1.5, 1.5, 90)
X = pd.DataFrame({"x": x})
exposure = np.linspace(1.1, 2.0, len(x))
counts = rng.poisson(exposure * np.exp(1.4 + np.sin(x))) + 1
log_exposure = np.log(exposure)
model = GeDSGeneralizedRegressor(
    family="poisson", spline_features=["x"], order=2,
    higher_order=False,
).fit(X, counts, offset=log_exposure)
expected_counts = model.predict(X, offset=log_exposure)
contributions = model.predict_terms(X, offset=log_exposure)
```

`predict_terms()` returns a DataFrame of the R spline and parametric term
contributions. Its rows sum to the link prediction after adding the offset;
the offset is not itself a term column. Offset prediction currently supports
one spline feature only because the R bivariate prediction method does not
apply new-data offsets consistently. A model fitted with an offset requires
an offset at prediction time.

This offset interface requires the GeDS GitHub fit and prediction fixes. A
earlier GeDS `0.3.6` installation without those fixes may mishandle
generalized-model offsets; the backend rejects that version.

Choose spline and parametric components explicitly for mixed data:

```python
model = GeDSRegressor(
    spline_features=["x"],
    linear_features=["group"],
).fit(X, y)
```

Spline features must be numeric. Parametric features may be numeric or
categorical; their encoding is performed by the R package so fitting and
prediction use R's native factor semantics.
If `spline_features` is omitted, all columns are used in a single joint spline
term. Select `spline_features=["x"]` and `linear_features=["group"]` to keep
`group` parametric instead. Two spline features create a joint bivariate
surface, not two separate additive smooths; R's support for more than two
spline features is experimental. With named pandas columns, prediction may
receive columns in a different order because the wrapper restores the fitted
column order before calling R.

### Additive GAM and boosting models

Use `GeDSGAMRegressor` for R's `NGeDSgam()` and `GeDSBoostRegressor` for
`NGeDSboost()`. Each entry in `spline_terms` is one additive smooth. Put two
features in the same entry for a joint surface. If omitted, each non-linear
feature gets its own smooth; `linear_features` selects parametric terms.

```python
import numpy as np
import pandas as pd
from geds import GeDSGAMRegressor, GeDSBoostRegressor

x = np.linspace(-2, 2, 100)
X = pd.DataFrame({"x": x, "z": x**2})
y = np.sin(x) + 0.3 * x**2

gam = GeDSGAMRegressor(
    spline_terms=[("x",), ("z",)], max_iterations=10
).fit(X, y)
boost = GeDSBoostRegressor(
    spline_terms=[("x",), ("z",)], max_iterations=20
).fit(X, y)

gam_predictions = gam.predict(X)
boost_predictions = boost.predict(X)
x_contribution = gam.predict_component(X, "f(x)")
importance = boost.get_base_learner_importance()
```

Both estimators expose `predict_link()`, order-specific coefficients, knots,
deviance, log likelihood, and coefficient confidence intervals through R.
`predict_component()` delegates a named learner prediction to R. The R
GAM/boost prediction method does not support `type="terms"`, so these
estimators do not offer `predict_terms()`. The GAM wrapper supports the
families accepted by `GeDSGeneralizedRegressor`; for binomial fits it accepts
0/1 responses and creates the factor required by R. The boosting
wrapper maps `gaussian`, `poisson`, `binomial`, and `gamma` to mboost families.
The fitted boosting estimator's `n_iter_` is R's total boosting iteration
count. `get_base_learner_importance()` returns R's `bl_imp()` in-bag risk
reductions as a pandas Series with the original Python feature names.
For a boosted fit with one univariate spline feature, R's iteration plots can
be saved to a multipage PDF without opening R directly:

```python
single_boost = GeDSBoostRegressor(max_iterations=10).fit(X[["x"]], y)
single_boost.save_boosting_diagnostics(
    "boosting.pdf", iterations=[0, 1, 2], final_fits=True
)
```

The method refuses to replace an existing file unless `overwrite=True`.
For binomial boosting, R expects responses encoded as -1 and 1. Offset
prediction is not offered for these additive estimators yet.

Fitted estimators contain a serialized R model and can be saved with
`model.save(path)` and restored with `GeDSRegressor.load(path)`. As with any
pickle-based format, only load files from trusted sources.

## Development

Clone the repository, then install the development dependencies and run the
integration tests with:

```console
git clone https://github.com/emilioluissaenzguillen/GeDS-python.git
cd GeDS-python
python -m pip install -e ".[dev]"
python -m pytest
python -m build
```

The tests start an embedded R session and therefore require a working GeDS
installation; they do not substitute or reimplement any GeDS calculations.

## Contact

For questions about the Python interface, contact Emilio L. Sáenz Guillén at
[emilioluissaenzguillen@gmail.com](mailto:emilioluissaenzguillen@gmail.com).
