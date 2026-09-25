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
| `shapeConstrain()` | Available as `shape_constrain()` for supported normal, GAM, and boosted models | Mixed additive terms and basic constraints have R parity tests; broaden combined-constraint and family coverage where R permits it. |
| `NGeDSgam()` and `NGeDSboost()` | Available as `GeDSGAMRegressor` and `GeDSBoostRegressor` | Extend parity tests to additional families, mixed terms, and orders. R remains the sole fitting implementation. |
| `N.boost.iter()` and `bl_imp()` | Available as `n_iter_` and `get_base_learner_importance()` | Both are directly compared with R for boosted fits. |
| `Derive()`, `Integrate()`, `PPolyRep()` | Available as `derive()`, `integrate()`, and `piecewise_polynomial()` for univariate spline fits | Direct R parity covers orders 2, 3, and 4; generalized Poisson derivative and output shape are also tested. |
| `crossv_GeDS()` | Available as `cross_validate_geds()` for Gaussian models with R's own grid and summaries; scikit-learn CV also works sequentially | Extend to non-Gaussian models only if R's CV routine supports forwarding the family reliably. |
| `visualize_boosting()` and R plotting | Python `plot_fit()` covers univariate fits; `save_boosting_diagnostics()` saves R's single-learner boosting plots as a multipage PDF | Broader plotting is a separate UX decision; preserve R's one-predictor restriction. |
| Exported low-level fitters (`UnivariateFitter`, `IRLSfit`, etc.) | Not wrapped | These are implementation-level building blocks, not yet part of the proposed Python estimator API. |

## Priority for local development

1. Keep the pinned R GitHub commit in CI and installation instructions until
   a later R release contains the same bridge capabilities. The GitHub R
   package remains version 0.3.6; an internal marker distinguishes fixed builds
   from older builds reporting the same version. CRAN review follows its
   separate schedule.
2. Review the intentional limits of R's specialized cross-validation and
   boosting plots; both now have Python interfaces for supported cases.
3. Review the public API and test clean installation across supported systems
   before selecting the next release version.

The R implementation describes multivariate fits beyond two spline covariates
as experimental. Offset support in the current R prediction method applies to
univariate fits; its bivariate branch does not add a new-data offset. The
Python API should make that boundary explicit.
