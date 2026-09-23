# Changelog

All notable changes to GeDS for Python are documented in this file.

The project follows [Semantic Versioning](https://semver.org/). Versions use
the Python packaging form of pre-release identifiers, such as `0.1.0a1` for
the first alpha release.

## 0.1.0a3 - 2026-09-23

- Add `python -m geds.check` for human-readable or JSON environment checks.
- Add `geds.plot_fit()` for Python-native visualization of univariate fits and
  internal knots.
- Expand installation and R-library troubleshooting guidance.

## 0.1.0a2 - 2026-09-20

- Preload R's core numerical DLLs on Windows so embedded R 4.6 can load
  recommended packages without requiring Rtools on the user's `PATH`.

## 0.1.0a1 - 2026-09-20

Initial alpha release.

- Add scikit-learn-style estimators for normal and generalized GeDS models.
- Delegate all statistical fitting, prediction, coefficient extraction, and
  knot extraction to the GeDS R package.
- Support pandas and NumPy inputs, numeric and categorical parametric terms,
  sample weights, model serialization, and environment diagnostics.
- Support Windows and Linux with R 4.6.1 and Python 3.10 or 3.12 in CI.
- Add a nonlinear regression example illustrating adaptive knot placement.
