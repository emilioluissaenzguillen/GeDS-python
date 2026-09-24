# GeDS R-to-Python feature audit

This audit compares the current Python wrapper with the R package's exported
interfaces. The R package remains the implementation of fitting and
statistical calculations.

| R capability | Python status | Next step |
| --- | --- | --- |
| `NGeDS()` and `GGeDS()` | Available as `GeDSRegressor` and `GeDSGeneralizedRegressor` | Keep parity tests for representative families and model shapes. |
| Spline and parametric formula components | Available through feature selectors | Reordered prediction columns are tested; the joint-spline versus parametric distinction is documented. |
| Prior weights | Available as `sample_weight` | Seeded weighted normal and Poisson fits are compared with independent R fits; test other families when added. |
| Offsets in model formulas | Available for one spline feature through `fit(..., offset=...)` and prediction methods | The low-count Poisson offset failure has R/Python regression tests; the R fixes are on GitHub at the pinned commit. R's bivariate prediction branch does not apply new-data offsets. |
| Orders 2, 3, and 4 | Available when fitted | Make errors for unavailable orders clearer. |
| `predict(..., type = "response"/"link")` | Available | Direct R parity is tested for weighted normal, Poisson, and bivariate normal fits; add more families. |
| `predict(..., type = "terms")` | Available as `predict_terms()` | The required R term-matrix fix is in the pinned GitHub commit. |
| `coef()`, `knots()`, `deviance()` | Available for selected order | Seeded weighted coefficients match an independent R fit; broaden knot and deviance parity tests. |
| `confint()` and `logLik()` | Available locally through order-specific methods | Confidence interval values and names match R for univariate and bivariate fits; Poisson and bivariate log likelihoods match R. |
| `shapeConstrain()` | Available as `shape_constrain()` for supported normal, GAM, and boosted models | Extend tests to mixed additive terms and combined shape constraints. |
| `NGeDSgam()` and `NGeDSboost()` | Available as `GeDSGAMRegressor` and `GeDSBoostRegressor` | Extend parity tests to additional families, mixed terms, and orders. R remains the sole fitting implementation. |
| `Derive()`, `Integrate()`, `PPolyRep()` | Available as `derive()`, `integrate()`, and `piecewise_polynomial()` for univariate spline fits | Extend tests to generalized fits and alternate spline orders. |
| Cross-validation helper | Missing | Evaluate whether scikit-learn's cross-validation already covers the Python use case. |
| R plotting | Python `plot_fit()` covers univariate fits | Extend plotting only with clear model-specific semantics. |

## Priority for local development

1. Keep the pinned R GitHub commit in CI and installation instructions until
   a later R release contains the same bridge capabilities. The GitHub R
   package remains version 0.3.6; an internal marker distinguishes fixed builds
   from older builds reporting the same version. CRAN review follows its
   separate schedule.
2. Finish parity checks for fitted-model statistics that R already computes.
3. Strengthen parity tests and documentation, including a generalized model
   example and platform-specific installation checks.
4. Review the public API and test clean installation across supported systems
   before selecting the next release version.

The R implementation describes multivariate fits beyond two spline covariates
as experimental. Offset support in the current R prediction method applies to
univariate fits; its bivariate branch does not add a new-data offset. The
Python API should make that boundary explicit.
