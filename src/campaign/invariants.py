"""What every run must satisfy, and the check that says whether it did.

An invariant is a property a correct run cannot violate whatever its
parameters: a conserved total stays constant, a population never goes
negative, a cumulative count never decreases. They are declared in the
campaign file and checked against the series a model returns.

The vocabulary is closed on purpose. There are four checks, ``finite``,
``nonnegative``, ``conserved`` and ``monotone``, and no way to write an
expression. An expression read from a configuration file and evaluated is
code execution, and a campaign file is the kind of thing that gets copied
between machines and people.

Every check is written so that a NaN fails it. That sounds automatic and is
not: ``np.all(x >= 0)`` is False when ``x`` holds a NaN, so that form is safe,
but ``not np.any(x < 0)`` is True, because every comparison with NaN is False.
The two read the same and disagree exactly when it matters, which is why each
check below tests for non-finite values explicitly before it tests anything
else.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast, get_args

import numpy as np
from numpy.typing import NDArray

__all__ = ["CHECKS", "Invariant", "InvariantError", "Violation", "check", "parse_invariant"]

Check = Literal["finite", "nonnegative", "conserved", "monotone"]
Direction = Literal["increasing", "decreasing"]
CHECKS: tuple[str, ...] = get_args(Check)

# Which keys each check accepts, beyond ``check`` and ``name``. Anything else
# is rejected: a misspelt ``rtoll`` silently falling back to a default would
# make the check stricter or looser than the author believes.
_ALLOWED: dict[str, frozenset[str]] = {
    "finite": frozenset({"series"}),
    "nonnegative": frozenset({"series", "atol"}),
    "conserved": frozenset({"series", "rtol", "atol"}),
    "monotone": frozenset({"series", "direction", "atol"}),
}


class InvariantError(ValueError):
    """An invariant is declared incorrectly. Raised while reading the file."""


@dataclass(frozen=True, slots=True)
class Invariant:
    """One declared property of a run.

    Attributes:
        check: which of the four checks this is.
        series: the names of the model output series it applies to.
        name: a label for the report, derived from the check if not given.
        rtol: relative tolerance, used by ``conserved`` only.
        atol: absolute tolerance.
        direction: ``increasing`` or ``decreasing``, for ``monotone`` only.
    """

    check: Check
    series: tuple[str, ...]
    name: str
    rtol: float = 0.0
    atol: float = 0.0
    direction: Direction | None = None


@dataclass(frozen=True, slots=True)
class Violation:
    """One invariant a run failed, and the evidence.

    Attributes:
        invariant: the name of the invariant that failed.
        detail: what was measured, where, and what was allowed.
    """

    invariant: str
    detail: str


def _tolerance(spec: Mapping[str, Any], key: str, where: str) -> float:
    value = spec.get(key, 0.0)
    # bool is an int in Python; `atol: true` is a mistake, not a tolerance.
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvariantError(f"{where}: {key} must be a number, got {value!r}")
    if not np.isfinite(value) or value < 0:
        raise InvariantError(f"{where}: {key} must be finite and non-negative, got {value}")
    return float(value)


def _series(spec: Mapping[str, Any], where: str) -> tuple[str, ...]:
    raw = spec.get("series")
    names = [raw] if isinstance(raw, str) else raw
    if not isinstance(names, list) or not names:
        raise InvariantError(f"{where}: series must be a name or a non-empty list of names")
    if not all(isinstance(n, str) and n for n in names):
        raise InvariantError(f"{where}: every series name must be a non-empty string")
    if len(set(names)) != len(names):
        raise InvariantError(f"{where}: series lists a name twice: {names}")
    return tuple(names)


def parse_invariant(spec: object, where: str) -> Invariant:
    """Build an :class:`Invariant` from one entry of the campaign file.

    Args:
        spec: the mapping read from YAML.
        where: a location for error messages, such as ``"invariants[2]"``.

    Raises:
        InvariantError: the entry is not a valid invariant.

    >>> parse_invariant({"check": "conserved", "series": ["S", "I", "R"],
    ...                  "rtol": 1e-9}, "invariants[0]").name
    'conserved(S+I+R)'
    """
    if not isinstance(spec, Mapping):
        raise InvariantError(f"{where}: expected a mapping, got {type(spec).__name__}")
    raw_kind = spec.get("check")
    if raw_kind not in CHECKS:
        raise InvariantError(
            f"{where}: check must be one of {', '.join(CHECKS)}; got {raw_kind!r}"
        )
    kind = cast("Check", raw_kind)

    unknown = set(spec) - _ALLOWED[kind] - {"check", "name"}
    if unknown:
        allowed = ", ".join(sorted(_ALLOWED[kind] | {"name"}))
        raise InvariantError(
            f"{where}: {kind} does not take {', '.join(sorted(unknown))}; "
            f"it takes {allowed}"
        )

    series = _series(spec, where)
    rtol = _tolerance(spec, "rtol", where)
    atol = _tolerance(spec, "atol", where)

    direction: Direction | None = None
    if kind == "conserved" and "rtol" not in spec and "atol" not in spec:
        raise InvariantError(
            f"{where}: conserved needs rtol or atol; a floating-point sum is never "
            "exactly constant, so a zero tolerance fails every run"
        )
    if kind == "monotone":
        if len(series) != 1:
            raise InvariantError(f"{where}: monotone applies to exactly one series")
        raw_direction = spec.get("direction")
        if raw_direction not in get_args(Direction):
            raise InvariantError(
                f"{where}: monotone needs direction: increasing or decreasing, "
                f"got {raw_direction!r}"
            )
        direction = cast("Direction", raw_direction)

    name = spec.get("name")
    if name is None:
        joined = "+".join(series) if kind == "conserved" else ", ".join(series)
        prefix = f"monotone {direction}" if direction else kind
        name = f"{prefix}({joined})"
    elif not isinstance(name, str) or not name:
        raise InvariantError(f"{where}: name must be a non-empty string")

    return Invariant(
        check=kind, series=series, name=name, rtol=rtol, atol=atol, direction=direction
    )


# Each check receives arrays already known to be finite, non-empty and
# one-dimensional, and returns why the invariant fails, or None if it holds.
_Arrays = Sequence[NDArray[np.float64]]


def _nonfinite(inv: Invariant, arrays: _Arrays) -> str | None:
    for label, x in zip(inv.series, arrays, strict=True):
        bad = np.flatnonzero(~np.isfinite(x))
        if bad.size:
            return (
                f"{label} has {bad.size} non-finite value(s), the first at index {bad[0]}"
            )
    return None


def _finite(inv: Invariant, arrays: _Arrays) -> str | None:  # noqa: ARG001
    # Reaching here means _nonfinite already found nothing.
    return None


def _nonnegative(inv: Invariant, arrays: _Arrays) -> str | None:
    for label, x in zip(inv.series, arrays, strict=True):
        index = int(np.argmin(x))
        if x[index] < -inv.atol:
            bound = "below zero" if inv.atol == 0 else f"more than {inv.atol:g} below zero"
            return f"{label} reaches {x[index]:.6g} at index {index}, {bound}"
    return None


def _conserved(inv: Invariant, arrays: _Arrays) -> str | None:
    lengths = {x.shape[0] for x in arrays}
    if len(lengths) != 1:
        return f"series have different lengths: {sorted(lengths)}"
    total = np.sum(arrays, axis=0)
    drift = np.abs(total - total[0])
    allowed = inv.atol + inv.rtol * abs(total[0])
    index = int(np.argmax(drift))
    if drift[index] <= allowed:
        return None
    return (
        f"{'+'.join(inv.series)} drifts by {drift[index]:.3g} from its initial "
        f"value {total[0]:.6g} at index {index}; allowed {allowed:.3g}"
    )


def _monotone(inv: Invariant, arrays: _Arrays) -> str | None:
    (x,) = arrays
    moves = np.diff(x)
    # Measure every step in the direction that should never be negative.
    steps = moves if inv.direction == "increasing" else -moves
    if steps.size == 0:
        return None
    index = int(np.argmin(steps))
    if steps[index] >= -inv.atol:
        return None
    return (
        f"{inv.series[0]} is not {inv.direction}: it moves by "
        f"{moves[index]:.3g} between index {index} and {index + 1}"
    )


_CHECK = {
    "finite": _finite,
    "nonnegative": _nonnegative,
    "conserved": _conserved,
    "monotone": _monotone,
}


def _check_one(inv: Invariant, arrays: _Arrays) -> str | None:
    """Return why ``inv`` fails on ``arrays``, or None if it holds.

    Non-finite values are ruled out first, for every check, so no check can
    wave a NaN through by way of a comparison that is quietly False.
    """
    return _nonfinite(inv, arrays) or _CHECK[inv.check](inv, arrays)


def check(invariants: Sequence[Invariant], series: Mapping[str, object]) -> list[Violation]:
    """Check every invariant against a run's output series.

    Every invariant is checked, not only up to the first failure, so the
    report shows everything a run got wrong at once.

    Args:
        invariants: the declared invariants.
        series: the model's output, name to one-dimensional array.

    Returns:
        One :class:`Violation` per failed invariant, in declaration order.
        An empty list means the run satisfies all of them.

    >>> inv = parse_invariant({"check": "nonnegative", "series": "x"}, "doc")
    >>> check([inv], {"x": [1.0, 0.5, -0.25]})
    [Violation(invariant='nonnegative(x)', detail='x reaches -0.25 at index 2, below zero')]
    """
    violations: list[Violation] = []
    for inv in invariants:
        missing = [s for s in inv.series if s not in series]
        if missing:
            available = ", ".join(sorted(series)) or "none"
            detail = (
                f"no series named {', '.join(missing)} in the output (have: {available})"
            )
            violations.append(Violation(inv.name, detail))
            continue

        arrays = [np.asarray(series[s], dtype=np.float64) for s in inv.series]
        shapes = [a.shape for a in arrays]
        if any(len(shape) != 1 or shape[0] == 0 for shape in shapes):
            detail = f"series must be non-empty and one-dimensional, got shapes {shapes}"
            violations.append(Violation(inv.name, detail))
            continue

        reason = _check_one(inv, arrays)
        if reason is not None:
            violations.append(Violation(inv.name, reason))
    return violations
