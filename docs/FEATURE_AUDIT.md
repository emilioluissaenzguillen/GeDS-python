# GeDS R-to-Python feature audit

This audit compares the current Python wrapper with the R package's exported
interfaces. The R package remains the implementation of fitting and
statistical calculations.

| R capability | Python status | Next step |
| --- | --- | --- |
| `NGeDS()` and `GGeDS()` | Available as `GeDSRegressor` and `GeDSGeneralizedRegressor` | Keep parity tests for representative families and model shapes. |
| Spline and parametric formula components | Available through feature selectors | Reordered prediction columns are tested; the joint-spline versus parametric distinction is documented. |
| Prior weights | Available as `sample_weight` | Seeded weighted normal and Poisson fits are compared with independent R fits; test other families when added. |
| Offsets in model formulas | Available locally for one spline feature through `fit(..., offset=...)` and prediction methods | The low-count Poisson offset failure was fixed in local R source and has R/Python regression tests. Publish both R fit and prediction fixes on GitHub before releasing this Python API. R's bivariate prediction branch does not apply new-data offsets. |
| Orders 2, 3, and 4 | Available when fitted | Make errors for unavailable orders clearer. |
| `predict(..., type = "response"/"link")` | Available | Direct R parity is tested for weighted normal, Poisson, and bivariate normal fits; add more families. |
| `predict(..., type = "terms")` | Available locally as `predict_terms()` | Requires the local R term-matrix fix to be published on GitHub. |
| `coef()`, `knots()`, `deviance()` | Available for selected order | Seeded weighted coefficients match an independent R fit; broaden knot and deviance parity tests. |
| `confint()` and `logLik()` | Available locally through order-specific methods | Confidence interval values and names match R for univariate and bivariate fits; Poisson and bivariate log likelihoods match R. |
| `shapeConstrain()` | Missing | Evaluate a separate post-fit API after testing supported shapes and model types. |
| `NGeDSgam()` and `NGeDSboost()` | Missing | Consider separate estimators after the core model API stabilizes. |
| `Derive()`, `Integrate()`, `PPolyRep()` | Missing | Offer model methods only after checking input/output forms and use cases. |
| Cross-validation helper | Missing | Evaluate whether scikit-learn's cross-validation already covers the Python use case. |
| R plotting | Python `plot_fit()` covers univariate fits | Extend plotting only with clear model-specific semantics. |

## Priority for local development

1. Push the locally tested R prediction fixes to GitHub when ready; a CRAN
   submission can follow its separate review schedule. Before releasing Python
   offsets and terms, test against that exact GitHub commit or tag and tell
   users how to install it. Exposure offsets are especially useful for Poisson
   models.
2. Finish parity checks for fitted-model statistics that R already computes.
3. Strengthen parity tests and documentation, including a generalized model
   example and platform-specific installation checks.
4. Review the public API and test clean installation across supported systems
   before selecting the next release version.

The R implementation describes multivariate fits beyond two spline covariates
as experimental. Offset support in the current R prediction method applies to
univariate fits; its bivariate branch does not add a new-data offset. The
Python API should make that boundary explicit.
