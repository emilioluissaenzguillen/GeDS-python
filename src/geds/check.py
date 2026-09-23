"""Command-line environment check for the Python/R bridge."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from ._backend import BackendUnavailableError, diagnostics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m geds.check",
        description="Check that Python can initialize R and load the GeDS package.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print machine-readable JSON instead of the human-readable report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the environment check and return a process exit code."""
    args = _parser().parse_args(argv)
    try:
        information = diagnostics()
    except BackendUnavailableError as exc:
        if args.json:
            print(json.dumps({"status": "error", "message": str(exc)}, indent=2))
        else:
            print("GeDS environment check: FAILED", file=sys.stderr)
            print(str(exc), file=sys.stderr)
            print(
                "Set R_HOME if the intended R installation is not discovered, "
                "and set GEDS_R_LIBRARY if GeDS is in a non-default R library.",
                file=sys.stderr,
            )
        return 1

    report = {"status": "ok", **information}
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        labels = {
            "python": "Python",
            "r_version": "R",
            "r_home": "R home",
            "rpy2_version": "rpy2",
            "geds_version": "GeDS",
            "geds_library": "GeDS library",
            "minimum_geds_version": "Minimum GeDS",
        }
        print("GeDS environment check: OK")
        for key, label in labels.items():
            print(f"  {label}: {information[key]}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a module
    raise SystemExit(main())
