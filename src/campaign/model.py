"""The contract between campaign and a model.

A model is a plain function in a plain Python file::

    from campaign import RunOutput

    def simulate(params: dict, seed: int) -> RunOutput:
        ...
        return RunOutput(series={"t": t, "x": x}, metrics={"peak": x.max()})

``series`` are the arrays the invariants are checked against, and they are
stored with the run so that a later check never needs the model again.
``metrics`` are the scalars that become one row of the results table.

The model file is loaded by path, not imported by name, so it does not have
to be installed or sit on ``sys.path``. Its directory is put on ``sys.path``
while it loads, as ``python model.py`` would, so it can import its siblings.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["ModelError", "RunOutput", "file_hash", "load_model"]


class ModelError(Exception):
    """The model cannot be loaded, or returned something that is not a run."""


@dataclass(frozen=True, slots=True)
class RunOutput:
    """What a model returns for one scenario.

    Attributes:
        series: named one-dimensional arrays, checked by the invariants.
        metrics: named scalars, one column each in the results table.
    """

    series: Mapping[str, ArrayLike] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)

    def arrays(self) -> dict[str, NDArray[np.float64]]:
        """The series as float arrays, or :class:`ModelError` if one is not numeric."""
        out: dict[str, NDArray[np.float64]] = {}
        for key, values in self.series.items():
            name = _name("series", key)
            try:
                out[name] = np.asarray(values, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ModelError(f"series {name!r} is not numeric: {exc}") from exc
        return out

    def scalars(self) -> dict[str, float]:
        """The metrics as floats, or :class:`ModelError` if one is not a number."""
        out: dict[str, float] = {}
        for key, value in self.metrics.items():
            name = _name("metric", key)
            try:
                out[name] = float(value)
            except (TypeError, ValueError) as exc:
                raise ModelError(f"metric {name!r} is not a number: {value!r}") from exc
        return out


def _name(kind: str, key: object) -> str:
    # The annotations say str, but a model is ordinary, often untyped, code,
    # and {1: array} is easy to write. Checked at runtime for that reason.
    if not isinstance(key, str) or not key:
        raise ModelError(f"{kind} names must be non-empty strings, got {key!r}")
    return key


ModelFunction = Callable[[dict[str, Any], int], object]


def file_hash(path: Path) -> str:
    """SHA-256 of a file's bytes.

    Recorded with every run, so that editing the model invalidates the runs it
    produced. Only the model file itself is hashed: a change in a module it
    imports is not detected, which the README says.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(path: Path, function: str) -> ModelFunction:
    """Load ``function`` from the Python file at ``path``.

    The module is registered under a name derived from the file's content, so
    two different files, or two versions of one file, never share a module,
    and loading the same unchanged file twice in one process reuses it.

    Raises:
        ModelError: the file cannot be executed or has no such function.
    """
    digest = file_hash(path)
    module_name = f"_campaign_model_{digest[:16]}"
    module = sys.modules.get(module_name)

    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ModelError(f"cannot load {path} as a Python module")
        module = importlib.util.module_from_spec(spec)
        # Registered before execution, as the importlib documentation's recipe
        # does: a dataclass defined in the model looks its module up here.
        sys.modules[module_name] = module
        directory = str(path.parent)
        if directory not in sys.path:
            sys.path.insert(0, directory)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            del sys.modules[module_name]
            raise ModelError(f"{path.name} raised while loading: {exc!r}") from exc

    fn = getattr(module, function, None)
    if not callable(fn):
        raise ModelError(f"{path.name} has no function named {function!r}")
    return cast("ModelFunction", fn)
