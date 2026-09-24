"""Lazy, narrowly scoped access to the R implementation of GeDS."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
from importlib.metadata import version
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Iterator

import numpy as np
import pandas as pd


class BackendUnavailableError(RuntimeError):
    """Raised when R, rpy2, or the GeDS R package is unavailable."""


_DLL_HANDLES: list[Any] = []
MIN_GEDS_VERSION = "0.3.6"
REQUIRED_GEDS_CAPABILITIES = {"offset_glm_v1", "terms_matrix_v1"}


def _version_key(path: Path) -> tuple[int, ...]:
    match = re.search(r"R-(\d+(?:\.\d+)*)$", path.name)
    return tuple(int(part) for part in match.group(1).split(".")) if match else ()


def _discover_r_home() -> Path:
    configured = os.environ.get("R_HOME")
    if configured and (Path(configured) / "bin").is_dir():
        return Path(configured)

    if os.name == "nt":
        roots = [Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "R"]
        candidates = [
            path
            for root in roots
            if root.is_dir()
            for path in root.glob("R-*")
            if (path / "bin" / "Rscript.exe").is_file()
        ]
        if candidates:
            return max(candidates, key=_version_key)

    rscript = shutil.which("Rscript")
    if rscript:
        executable = Path(rscript).resolve()
        try:
            reported_home = subprocess.run(
                [str(executable), "--vanilla", "-e", "cat(R.home())"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            reported_home = ""
        if reported_home and (Path(reported_home) / "bin").is_dir():
            return Path(reported_home)

    raise BackendUnavailableError(
        "R was not found. Install R and either put Rscript on PATH or set R_HOME."
    )


def _configure_r_process() -> Path:
    """Configure the current process before rpy2 imports R's shared library."""
    r_home = _discover_r_home()
    os.environ["R_HOME"] = str(r_home)

    if os.name == "nt":
        r_bin = r_home / "bin" / "x64"
        if not r_bin.is_dir():
            r_bin = r_home / "bin"
        os.environ["PATH"] = os.pathsep.join(
            (str(r_bin), str(r_home / "bin"), os.environ.get("PATH", ""))
        )
        if hasattr(os, "add_dll_directory"):
            _DLL_HANDLES.append(os.add_dll_directory(str(r_bin)))
        # R loads package DLLs itself. On current Windows/Python combinations,
        # registering the directory with the OS is also required for their
        # transitive dependencies (for example stats.dll -> R.dll).
        if not ctypes.windll.kernel32.SetDllDirectoryW(str(r_bin)):
            raise BackendUnavailableError(
                f"Windows could not register R's DLL directory: {r_bin}"
            )
        # Explicitly retain R's core DLLs as well. R 4.6 can otherwise start
        # successfully while later failing to load recommended packages such
        # as stats because their Rblas/Rlapack dependencies are not found.
        for dll_name in ("R.dll", "Rblas.dll", "Rlapack.dll", "Riconv.dll"):
            dll_path = r_bin / dll_name
            if dll_path.is_file():
                try:
                    _DLL_HANDLES.append(ctypes.WinDLL(str(dll_path)))
                except OSError as exc:
                    raise BackendUnavailableError(
                        f"Windows could not load R's core library: {dll_path}"
                    ) from exc

    return r_home


class RBackend:
    """Own the embedded-R session and keep fitted models opaque."""

    def __init__(self) -> None:
        self.r_home = _configure_r_process()
        rpy2_situation = None
        original_get_r_flags = None
        try:
            # rpy2 3.6.x probes ``R CMD config --ldflags`` before loading
            # R.dll. Some Windows R builds return no stdout for that query;
            # rpy2 then raises IndexError instead of taking its normal
            # Windows DLL-directory fallback. Translate only that empty-output
            # case into the exception the fallback already handles.
            if os.name == "nt":
                import rpy2.situation as rpy2_situation

                original_get_r_flags = rpy2_situation.get_r_flags

                def get_r_flags_with_windows_fallback(*args: Any, **kwargs: Any) -> Any:
                    try:
                        return original_get_r_flags(*args, **kwargs)
                    except IndexError as exc:
                        raise subprocess.CalledProcessError(1, "R CMD config") from exc

                rpy2_situation.get_r_flags = get_r_flags_with_windows_fallback

            import rpy2.robjects as ro
            from rpy2.rinterface_lib import openrlib
            from rpy2.robjects import default_converter, pandas2ri
            from rpy2.robjects.packages import importr
        except Exception as exc:  # pragma: no cover - depends on host setup
            raise BackendUnavailableError(
                "rpy2 could not initialize embedded R. Run geds.diagnostics() "
                "and verify that R and rpy2 use compatible versions."
            ) from exc
        finally:
            if rpy2_situation is not None and original_get_r_flags is not None:
                rpy2_situation.get_r_flags = original_get_r_flags

        self.ro = ro
        self._openrlib = openrlib
        self._converter = default_converter + pandas2ri.converter

        extra_library = os.environ.get("GEDS_R_LIBRARY")
        if extra_library:
            current = list(ro.r(".libPaths()"))
            ro.r[".libPaths"](ro.StrVector([extra_library, *current]))

        try:
            self.geds = importr("GeDS")
            self.stats = importr("stats")
        except Exception as exc:
            raise BackendUnavailableError(
                "The GeDS R package could not be loaded. Install GeDS into a "
                "library visible to this R installation."
            ) from exc

        self.geds_version = str(
            self.ro.r("as.character(packageVersion('GeDS'))")[0]
        )
        is_supported = bool(
            self.ro.r(
                "packageVersion('GeDS') >= "
                f"package_version('{MIN_GEDS_VERSION}')"
            )[0]
        )
        if not is_supported:
            raise BackendUnavailableError(
                f"GeDS {self.geds_version} is installed, but geds-python "
                f"requires GeDS >= {MIN_GEDS_VERSION}."
            )
        capabilities = self.ro.r(
            "get0('.GeDS_python_bridge_capabilities', "
            "envir=asNamespace('GeDS'), inherits=FALSE)"
        )
        available = set() if capabilities is self.ro.NULL else set(map(str, capabilities))
        missing = REQUIRED_GEDS_CAPABILITIES - available
        if missing:
            raise BackendUnavailableError(
                "This GeDS R build lacks Python bridge fixes "
                f"({', '.join(sorted(missing))}). Install the GeDS GitHub "
                "commit specified in the geds-python README."
            )

    @contextmanager
    def locked(self) -> Iterator[None]:
        """Serialize access to R's process-global runtime."""
        with self._openrlib.rlock:
            yield

    def dataframe_to_r(self, frame: pd.DataFrame) -> Any:
        with self.locked(), self._converter.context():
            return self.ro.conversion.get_conversion().py2rpy(frame)

    def vector(self, values: Any) -> Any:
        return self.ro.FloatVector(np.asarray(values, dtype=float))

    def formula(self, expression: str) -> Any:
        return self.ro.Formula(expression)

    def family(self, name: str, link: str | None) -> Any:
        normalized = name.lower()
        functions = {
            "gaussian": "gaussian",
            "poisson": "poisson",
            "quasipoisson": "quasipoisson",
            "binomial": "binomial",
            "quasibinomial": "quasibinomial",
            "gamma": "Gamma",
        }
        if normalized not in functions:
            choices = ", ".join(sorted(functions))
            raise ValueError(f"Unsupported family {name!r}; choose one of: {choices}.")
        function = getattr(self.stats, functions[normalized])
        return function() if link is None else function(link=link)

    def boost_family(self, name: str) -> Any:
        """Construct an mboost loss family for GeDS's R boosting algorithm."""
        from rpy2.robjects.packages import importr

        functions = {
            "gaussian": "Gaussian",
            "poisson": "Poisson",
            "binomial": "Binomial",
            "gamma": "GammaReg",
        }
        normalized = name.lower()
        if normalized not in functions:
            choices = ", ".join(sorted(functions))
            raise ValueError(f"Unsupported boosting family {name!r}; choose one of: {choices}.")
        with self.locked():
            return getattr(importr("mboost"), functions[normalized])()

    def fit(self, function: str, formula: Any, data: Any, **kwargs: Any) -> Any:
        with self.locked():
            return getattr(self.geds, function)(formula, data=data, **kwargs)

    def predict(
        self, model: Any, data: Any, order: int, prediction_type: str,
        *, base_learner: str | None = None,
    ) -> np.ndarray | pd.DataFrame:
        with self.locked():
            kwargs: dict[str, Any] = {"newdata": data, "n": order, "type": prediction_type}
            if base_learner is not None:
                kwargs["base_learner"] = base_learner
            result = self.stats.predict(model, **kwargs)
            values = np.array(result, dtype=float, copy=True)
            if prediction_type == "terms":
                names = self.ro.r("colnames")(result)
                if values.ndim != 2 or names is self.ro.NULL:
                    raise BackendUnavailableError(
                        "This GeDS R build does not return named term predictions. "
                        "Install the GeDS GitHub prediction fixes."
                    )
                columns = [str(name) for name in names]
                return pd.DataFrame(values, columns=columns)
        return values

    def coefficients(self, model: Any, order: int) -> Any:
        with self.locked():
            return self.to_python(self.stats.coef(model, n=order))

    def knots(self, model: Any, order: int) -> Any:
        with self.locked():
            return self.to_python(
                self.stats.knots(model, n=order, options="internal")
            )

    def deviance(self, model: Any, order: int) -> float:
        with self.locked():
            return float(self.stats.deviance(model, n=order)[0])

    def log_likelihood(self, model: Any, order: int) -> float:
        with self.locked():
            return float(self.stats.logLik(model, n=order)[0])

    def confidence_intervals(
        self, model: Any, order: int, level: float
    ) -> pd.DataFrame:
        with self.locked():
            intervals = self.stats.confint(model, n=order, level=level)
            names = [str(name) for name in self.ro.r("rownames")(intervals)]
            values = np.array(intervals, dtype=float, copy=True)
        return pd.DataFrame(values, index=names, columns=["lower", "upper"])

    def derive(self, model: Any, x: Any, order: int, spline_order: int) -> np.ndarray:
        with self.locked():
            result = self.geds.Derive(
                model, order=order, x=self.vector(x), n=spline_order
            )
            return np.array(result, dtype=float, copy=True)

    def integrate(self, model: Any, lower: Any, upper: Any, spline_order: int) -> np.ndarray:
        with self.locked():
            result = self.geds.Integrate(
                model, **{"from": self.vector(lower)}, to=self.vector(upper),
                n=spline_order,
            )
            return np.array(result, dtype=float, copy=True)

    def piecewise_polynomial(self, model: Any, spline_order: int) -> tuple[np.ndarray, np.ndarray]:
        with self.locked():
            result = self.geds.PPolyRep(model, n=spline_order)
            knots = np.array(result.rx2("knots"), dtype=float, copy=True)
            coefficients = np.array(result.rx2("coefficients"), dtype=float, copy=True)
            return knots, coefficients

    def shape_constrain(
        self, model: Any, spline_order: int, constraints: list[str],
        eps: float, ridge: float, base_learner: str | None,
    ) -> Any:
        with self.locked():
            kwargs: dict[str, Any] = {
                "n": spline_order,
                "shape_constraint": self.ro.StrVector(constraints),
                "eps": eps,
                "ridge": ridge,
            }
            if base_learner is not None:
                kwargs["base_learner"] = base_learner
            return self.geds.shapeConstrain(model, **kwargs)

    def base_learner_importance(
        self, model: Any, boosting_iter_only: bool
    ) -> pd.Series:
        with self.locked():
            result = self.geds.bl_imp(
                model, boosting_iter_only=boosting_iter_only
            )
            names = [str(name) for name in self.ro.r("names")(result)]
            values = np.array(result, dtype=float, copy=True)
        return pd.Series(values, index=names, name="importance")

    def cross_validate(
        self, model_name: str, formula: str, frame: pd.DataFrame,
        parameters: dict[str, np.ndarray], order: int, n_folds: int,
        n_cores: int, random_state: int | None,
        **fit_kwargs: Any,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Call R's specialized GeDS grid search, retaining its model symbol."""
        if model_name not in {"NGeDS", "GGeDS", "NGeDSgam", "NGeDSboost"}:
            raise ValueError("Unsupported GeDS cross-validation model.")
        with self.locked():
            # crossv_GeDS inspects the *symbol* of model_fun via substitute().
            # Passing an rpy2 function value directly would lose that symbol.
            wrapper = self.ro.r(
                "function(formula, data, parameters, n, n_folds, n_cores, ...) {"
                f" {model_name} <- GeDS::{model_name};"
                " GeDS::crossv_GeDS(formula=formula, data=data,"
                f" model_fun={model_name}, parameters=parameters,"
                " n=n, n_folds=n_folds, n_cores=n_cores, ...)"
                " }"
            )
            r_parameters = self.ro.ListVector({
                name: self.vector(values) for name, values in parameters.items()
            })
            if random_state is not None:
                self.ro.r["set.seed"](int(random_state))
            with self._converter.context():
                r_frame = self.ro.conversion.get_conversion().py2rpy(frame)
            library = os.environ.get("GEDS_R_LIBRARY")
            original_r_libs_user = os.environ.get("R_LIBS_USER")
            if library:
                os.environ["R_LIBS_USER"] = os.pathsep.join(
                    part for part in (library, original_r_libs_user) if part
                )
            try:
                result = wrapper(
                    self.formula(formula), r_frame,
                    parameters=r_parameters, n=order, n_folds=n_folds,
                    n_cores=n_cores, **fit_kwargs,
                )
            finally:
                if library:
                    if original_r_libs_user is None:
                        os.environ.pop("R_LIBS_USER", None)
                    else:
                        os.environ["R_LIBS_USER"] = original_r_libs_user
            with self._converter.context():
                conversion = self.ro.conversion.get_conversion()
                best = conversion.rpy2py(result.rx2("best_params"))
                results = conversion.rpy2py(result.rx2("results"))
            return pd.DataFrame(best).reset_index(drop=True), pd.DataFrame(results).reset_index(drop=True)

    def save_boosting_diagnostics(
        self, model: Any, path: Path, iterations: list[int], final_fits: bool
    ) -> None:
        """Write R ``visualize_boosting`` plots to a multipage PDF."""
        with self.locked():
            self.ro.r["pdf"](file=str(path), width=8, height=6, onefile=True)
            try:
                self.geds.visualize_boosting(
                    model, iters=self.ro.IntVector(iterations),
                    final_fits=final_fits,
                )
            finally:
                self.ro.r["dev.off"]()

    def component(self, model: Any, name: str) -> Any:
        try:
            return self.to_python(model.rx2(name))
        except Exception:
            return None

    def to_python(self, value: Any) -> Any:
        """Convert extracted values without recursively converting a model."""
        vectors = self.ro.vectors
        if value is self.ro.NULL:
            return None
        if isinstance(value, vectors.ListVector):
            names = list(value.names) if value.names is not self.ro.NULL else []
            converted = [self.to_python(item) for item in value]
            if names and len(names) == len(converted) and all(names):
                return dict(zip(names, converted))
            return converted
        if isinstance(value, vectors.StrVector):
            result = np.array(value, dtype=str, copy=True)
        elif isinstance(
            value, (vectors.FloatVector, vectors.IntVector, vectors.BoolVector)
        ):
            result = np.array(value, copy=True)
        else:
            return value
        return result.item() if result.ndim == 0 else result

    def information(self) -> dict[str, str]:
        with self.locked():
            return {
                "r_home": str(self.r_home),
                "r_version": str(self.ro.r("R.version.string")[0]),
                "geds_version": self.geds_version,
                "minimum_geds_version": MIN_GEDS_VERSION,
                "geds_library": str(self.ro.r("find.package('GeDS')")[0]),
                "rpy2_version": version("rpy2"),
                "python": sys.version.split()[0],
            }


_BACKEND: RBackend | None = None


def get_backend() -> RBackend:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = RBackend()
    return _BACKEND


def diagnostics() -> dict[str, str]:
    """Return the versions and locations used by the Python/R bridge."""
    return get_backend().information()
