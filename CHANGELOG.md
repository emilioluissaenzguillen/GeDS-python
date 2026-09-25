# Changelog

All notable changes to GeDS for Python are documented in this file.

The project follows [Semantic Versioning](https://semver.org/). Versions use
the Python packaging form of pre-release identifiers, such as `0.1.0a1` for
the first alpha release.

## Unreleased

## 0.1.0 - 2026-09-25

- Expose R's boosted base-learner importance alongside its existing iteration
  count, and clarify remaining specialized R-only utilities in the feature audit.
- Add a Gaussian-only Python interface to R's specialized `crossv_GeDS()`
  grid search and its MSE/knot/iteration summaries.
- Save R's single-learner boosting-iteration visualization as a multipage PDF
  through a Python model method.
- Expose R's `Derive()`, `Integrate()`, `PPolyRep()`, and `shapeConstrain()`
  through fitted-model methods, with direct R parity tests and explicit model
  restrictions.
- Check alternate spline orders, generalized-model utilities, mixed-term
  additive constraints, and sequential scikit-learn cross-validation.
- Add additive GAM and gradient-boosting estimators backed by the R package's
  `NGeDSgam()` and `NGeDSboost()` functions, with independently checked R/Python
  predictions and no duplicate statistical implementation.
- Support named additive spline terms, joint spline terms, parametric features,
  selected loss families, and individual base-learner predictions.

## 0.1.0a4 - release candidate (not published)

- Begin an R-to-Python feature audit for the next coordinated release.
- Add order-specific access to R's deviance, log likelihood, and coefficient
  confidence intervals.
- Add univariate fit/prediction offsets and named `predict_terms()` output,
  delegated to the locally fixed R package. These require the corresponding
  R GitHub changes before publication.
- Expand independent R/Python parity tests for weighted normal and Poisson
  fits, bivariate fits and confidence intervals; clarify joint-spline feature
  selection.
- Copy arrays extracted from R into Python-owned memory so later R calls cannot
  change stored coefficients, knots, or predictions.
- Require GeDS 0.3.6 with the Python bridge capability marker, and pin the
  tested GitHub commit in installation and CI instructions.

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
