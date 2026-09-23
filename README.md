# GeDS for Python

This package provides a Python interface to the
[GeDS R package](https://github.com/emilioluissaenzguillen/GeDS). The R package
is the sole implementation of the statistical methods. Python supplies a
scikit-learn-style API, pandas/NumPy conversion, environment diagnostics, and
model serialization.

## Requirements

- R 4.4 or newer (R 4.6.1 is used for development)
- GeDS 0.3.6 or newer
- Python 3.10 or newer

Install the Python package, including the optional plotting dependency used in
the example:

```console
python -m pip install "geds-python[plot]"
```

Install the R package separately, using R 4.6.1 or another supported R
installation:

```r
install.packages("GeDS")
```

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
`install.packages("GeDS")`, then rerun the check.

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
