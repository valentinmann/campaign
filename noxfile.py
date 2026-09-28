#!/usr/bin/env -S uv run
# /// script
# dependencies = ["nox>=2024.3.2"]
# ///
"""Task runner. `pip install nox`, then `nox` runs everything CI runs.

The point is that a contributor, or a reviewer, never has to reconstruct the
commands from the CI file:

    nox                    # lint, types and tests on the default interpreter
    nox -s tests           # tests only
    nox -s tests -p 3.11   # tests on one interpreter
    nox -s lint            # ruff check and format --check
    nox -s types           # mypy, strict
    nox -s demo            # run the demo campaign and draw the figure
    nox -s build           # build the wheel and sdist, then check the metadata
"""

from __future__ import annotations

import nox

nox.needs_version = ">=2024.3.2"
nox.options.default_venv_backend = "uv|virtualenv"
nox.options.reuse_existing_virtualenvs = True

PYTHONS = ["3.11", "3.12", "3.13"]


@nox.session
def lint(session: nox.Session) -> None:
    """Run ruff, both the linter and the formatter check."""
    session.install("ruff>=0.9")
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")


@nox.session
def types(session: nox.Session) -> None:
    """Type-check the package and the demo in strict mode."""
    session.install("-e", ".[dev]")
    session.run("mypy")


@nox.session(python=PYTHONS)
def tests(session: nox.Session) -> None:
    """Run the test suite, including the docstring examples."""
    session.install("-e", ".[dev]")
    session.run("pytest", *session.posargs)


@nox.session(default=False)
def coverage(session: nox.Session) -> None:
    """Run the tests and report line and branch coverage."""
    session.install("-e", ".[dev]")
    session.run("pytest", "--cov", "--cov-report=term-missing", *session.posargs)


@nox.session(default=False)
def demo(session: nox.Session) -> None:
    """Run the demo campaign end to end and redraw the figure."""
    session.install("-e", ".[demo]")
    session.run("campaign", "run", "demo/sir.yaml")
    session.run("python", "demo/plot.py")


@nox.session(default=False)
def build(session: nox.Session) -> None:
    """Build the distributions and validate their metadata."""
    session.install("build", "twine")
    session.run("python", "-m", "build")
    session.run("twine", "check", "--strict", "dist/*")


if __name__ == "__main__":
    nox.main()
