# Releasing GeDS for Python

Releases are built by GitHub Actions and published with PyPI Trusted
Publishing. Do not add a long-lived PyPI token to the repository.

## One-time setup

1. Create separate accounts on [PyPI](https://pypi.org/) and
   [TestPyPI](https://test.pypi.org/), if needed.
2. In the GitHub repository settings, create environments named `testpypi`
   and `pypi`. Configure `pypi` to require manual approval.
3. Register a pending GitHub Trusted Publisher on both package indexes with:

   - PyPI project name: `geds-python`
   - GitHub owner: `emilioluissaenzguillen`
   - GitHub repository: `GeDS-python`
   - Workflow filename: `release.yml`
   - Environment: `testpypi` on TestPyPI and `pypi` on PyPI

A pending publisher does not reserve the package name. The first successful
publication creates the project.

## Prepare a release

1. Update the version in `pyproject.toml`, `src/geds/__init__.py`, and
   `CITATION.cff`.
2. Add the release notes and date to `CHANGELOG.md`.
3. Run the test suite and build checks locally.
4. Merge the release commit into `main` and confirm that CI passes.

Published versions are immutable. Never reuse a version that has already been
uploaded to either package index.

## TestPyPI

Run the **Publish Python distribution** workflow manually from the `main`
branch. A manual run publishes only to TestPyPI.

To test without allowing TestPyPI to supply third-party dependencies, create a
clean environment and install in two steps:

```console
python -m pip install numpy pandas "rpy2>=3.6.7,<3.7" "scikit-learn>=1.4"
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps geds-python==0.1.0a1
```

Install GeDS in R, then run `geds.diagnostics()` and a fit/predict smoke test.

## PyPI

After TestPyPI validation:

1. Create an annotated tag matching the version, for example `v0.1.0a1`.
2. Push the tag.
3. Create and publish a GitHub Release from that tag.
4. Approve the protected `pypi` environment deployment.

Publishing the GitHub Release triggers production publication. The release
workflow builds the artifacts once, validates them with Twine, and passes the
same stored artifacts to the publishing job.
