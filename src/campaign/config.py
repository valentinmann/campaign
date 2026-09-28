"""Reading a campaign file.

A campaign file is YAML with a small, closed schema: a name, a model, a seed,
an output directory, either a parameter grid or an explicit list of scenarios,
and the invariants every run must satisfy. Unknown keys are errors rather
than warnings, because in a file that decides what gets computed, a typo that
is silently ignored is a wrong result that looks right.

Every scenario gets an identifier derived from its content, the model
reference and its parameters, and never from its position. Resuming depends
on it: if identifiers were positions, inserting one value at the front of a
grid would shift every index by one and a resumed campaign would skip the
wrong scenarios without a word.

Paths in the file, the model and the output directory, are resolved against
the file's own directory, so a campaign runs the same from any working
directory.
"""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from campaign.invariants import Invariant, InvariantError, parse_invariant

__all__ = [
    "DEFAULT_MAX_ARRAY_SIZE",
    "Campaign",
    "ConfigError",
    "Scalar",
    "Scenario",
    "SlurmSettings",
    "canonical_json",
    "load",
    "parse",
    "scenario_id",
]

Scalar = bool | int | float | str
Backend = Literal["local", "slurm"]
Format = Literal["csv", "parquet"]

# SLURM's default MaxArraySize. Array indices run from 0 to one less than it,
# so this is also the most tasks one array can hold on a default install.
DEFAULT_MAX_ARRAY_SIZE = 1001

_TOP_LEVEL = frozenset(
    {
        "name",
        "model",
        "seed",
        "output",
        "grid",
        "scenarios",
        "constants",
        "invariants",
        "backend",
        "workers",
        "slurm",
        "format",
    }
)
_REQUIRED = ("name", "model", "seed", "output")
_SLURM_KEYS = frozenset({"options", "throttle", "max_array_size", "sbatch"})
# Options campaign writes itself. Letting the file set them too would produce
# a job script with two conflicting values and no error from sbatch.
_SLURM_RESERVED = frozenset({"array", "output", "error", "job-name", "parsable", "wait"})
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SLURM_OPTION = re.compile(r"^[A-Za-z][A-Za-z0-9-]*$")


class ConfigError(ValueError):
    """The campaign file is malformed. The message says where and why."""


class _Loader(yaml.SafeLoader):
    """A SafeLoader that reads ``1e-9`` as a number.

    PyYAML implements YAML 1.1, whose float pattern requires a decimal point,
    so ``rtol: 1e-9`` loads as the *string* ``"1e-9"``. Nobody writing a
    tolerance means a string. The resolver added below is the exponent form
    from the YAML 1.2 core schema. This is still a SafeLoader: no tag can
    construct an arbitrary Python object.
    """


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:[0-9][0-9_]*)(?:\.[0-9_]*)?[eE][-+]?[0-9]+$"),
    list("-+0123456789"),
)


@dataclass(frozen=True, slots=True)
class Scenario:
    """One set of parameters to run.

    Attributes:
        id: 16 hex characters, a hash of the model reference and the
            parameters. Stable across runs, machines and grid reorderings.
        params: every parameter the model receives, swept and constant.
    """

    id: str
    params: dict[str, Scalar]


@dataclass(frozen=True, slots=True)
class SlurmSettings:
    """How to submit to SLURM.

    Attributes:
        options: written verbatim as ``#SBATCH --key=value`` lines. There are
            no defaults: partitions, accounts and limits belong to a cluster,
            not to this package.
        throttle: the most array tasks allowed to run at once, or None.
        max_array_size: the cluster's MaxArraySize. Larger campaigns are
            split into several arrays.
        sbatch: the submission command. Tests point it at a stub.
    """

    options: dict[str, str] = field(default_factory=dict)
    throttle: int | None = None
    max_array_size: int = DEFAULT_MAX_ARRAY_SIZE
    sbatch: tuple[str, ...] = ("sbatch",)


@dataclass(frozen=True, slots=True)
class Campaign:
    """A parsed, validated campaign file.

    Attributes:
        name: the campaign's name.
        source: the campaign file.
        model_ref: the model reference as written, ``path.py:function``.
        model_file: the model file, resolved.
        model_function: the function in it that runs one scenario.
        seed: the campaign seed, from which every run's seed is derived.
        output: the output directory, resolved.
        swept: the parameter names that vary, in the order they are declared.
        constants: parameters passed unchanged to every scenario.
        scenarios: every scenario, in declaration order.
        invariants: what every run must satisfy.
        backend: ``local`` or ``slurm``.
        workers: processes for the local backend.
        slurm: settings for the SLURM backend.
        format: ``csv`` or ``parquet`` for the results table.
        config_hash: a hash of everything that determines the results. How
            they are executed (backend, workers, SLURM settings) is excluded,
            because it does not change them.
    """

    name: str
    source: Path
    model_ref: str
    model_file: Path
    model_function: str
    seed: int
    output: Path
    swept: tuple[str, ...]
    constants: dict[str, Scalar]
    scenarios: tuple[Scenario, ...]
    invariants: tuple[Invariant, ...]
    backend: Backend
    workers: int
    slurm: SlurmSettings
    format: Format
    config_hash: str


def canonical_json(obj: object) -> str:
    """Serialise ``obj`` so that equal content always gives equal text.

    Keys are sorted, whitespace is fixed and floats use Python's shortest
    round-trip representation, so ``0.3`` and ``0.30`` in a YAML file hash
    the same while ``0.3`` and ``0.30000000000000004`` do not.

    >>> canonical_json({"b": 0.30, "a": [1, "x"]})
    '{"a":[1,"x"],"b":0.3}'
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def scenario_id(model_ref: str, params: Mapping[str, Scalar]) -> str:
    """The content identifier of a scenario.

    >>> scenario_id("m.py:f", {"a": 1, "b": 2}) == scenario_id("m.py:f", {"b": 2, "a": 1})
    True
    """
    payload = canonical_json({"model": model_ref, "params": dict(params)})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _tidy(value: float) -> float:
    """Round a generated value to 12 significant digits.

    ``np.linspace(0.15, 0.6, 10)`` produces ``0.30000000000000004``. That is
    the correct double, but it is not the number anybody meant, and it is the
    number that would appear in the results table.
    """
    return float(f"{value:.12g}")


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _scalar(value: object, where: str) -> Scalar:
    if isinstance(value, str | bool | int | float):
        if isinstance(value, float) and not np.isfinite(value):
            raise ConfigError(f"{where}: parameter values must be finite, got {value}")
        return value
    raise ConfigError(
        f"{where}: parameter values must be numbers, strings or booleans, "
        f"got {type(value).__name__}"
    )


def _axis(name: str, spec: object, where: str) -> list[Scalar]:
    """The values one grid dimension takes."""
    if isinstance(spec, list):
        if not spec:
            raise ConfigError(f"{where}: {name} has no values")
        values = [_scalar(v, f"{where}[{i}]") for i, v in enumerate(spec)]
        if len({canonical_json(v) for v in values}) != len(values):
            raise ConfigError(f"{where}: {name} lists a value twice")
        return values

    if isinstance(spec, Mapping):
        if set(spec) != {"start", "stop", "num"}:
            raise ConfigError(
                f"{where}: a range needs exactly start, stop and num; got {sorted(spec)}"
            )
        start, stop, num = spec["start"], spec["stop"], spec["num"]
        if not (_is_number(start) and _is_number(stop)):
            raise ConfigError(f"{where}: start and stop must be numbers")
        if not (np.isfinite(start) and np.isfinite(stop)):
            raise ConfigError(f"{where}: start and stop must be finite")
        if not isinstance(num, int) or isinstance(num, bool) or num < 1:
            raise ConfigError(f"{where}: num must be a positive integer, got {num!r}")
        values = [_tidy(float(v)) for v in np.linspace(start, stop, num)]
        if len(set(values)) != len(values):
            raise ConfigError(f"{where}: start equals stop, so the range repeats one value")
        return list(values)

    raise ConfigError(f"{where}: expected a list of values or a start/stop/num range")


def _swept_from_grid(grid: object) -> tuple[tuple[str, ...], list[dict[str, Scalar]]]:
    if not isinstance(grid, Mapping) or not grid:
        raise ConfigError("grid: expected a mapping of parameter name to values")
    names = tuple(str(n) for n in grid)
    axes = [_axis(n, grid[n], f"grid.{n}") for n in names]
    # itertools.product varies the last axis fastest, like nested loops written
    # in the order the axes are declared.
    rows = [dict(zip(names, combo, strict=True)) for combo in itertools.product(*axes)]
    return names, rows


def _swept_from_list(scenarios: object) -> tuple[tuple[str, ...], list[dict[str, Scalar]]]:
    if not isinstance(scenarios, list) or not scenarios:
        raise ConfigError("scenarios: expected a non-empty list of mappings")
    rows: list[dict[str, Scalar]] = []
    names: tuple[str, ...] = ()
    for i, entry in enumerate(scenarios):
        where = f"scenarios[{i}]"
        if not isinstance(entry, Mapping) or not entry:
            raise ConfigError(f"{where}: expected a non-empty mapping of parameters")
        keys = tuple(str(k) for k in entry)
        if i == 0:
            names = keys
        elif set(keys) != set(names):
            # Almost always a typo, and the table would have a hole in it.
            raise ConfigError(
                f"{where}: has parameters {sorted(keys)}, but scenarios[0] has "
                f"{sorted(names)}; every scenario must set the same parameters"
            )
        rows.append({k: _scalar(entry[k], f"{where}.{k}") for k in names})
    return names, rows


def _constants(raw: object) -> dict[str, Scalar]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError("constants: expected a mapping of parameter name to value")
    return {str(k): _scalar(v, f"constants.{k}") for k, v in raw.items()}


def _model(raw: object, base: Path) -> tuple[str, Path, str]:
    if not isinstance(raw, str) or ":" not in raw:
        raise ConfigError("model: expected 'path/to/file.py:function'")
    # The last colon: a Windows path such as C:\models\sir.py has one too.
    file_part, function = raw.rsplit(":", 1)
    if not function.isidentifier():
        raise ConfigError(f"model: {function!r} is not a valid function name")
    path = (base / file_part).resolve()
    if path.suffix != ".py" or not path.is_file():
        raise ConfigError(f"model: {path} is not an existing .py file")
    return raw, path, function


def _positive_int(value: object, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ConfigError(f"{where}: expected a positive integer, got {value!r}")
    return value


def _slurm(raw: object) -> SlurmSettings:
    if raw is None:
        return SlurmSettings()
    if not isinstance(raw, Mapping):
        raise ConfigError("slurm: expected a mapping")
    unknown = set(raw) - _SLURM_KEYS
    if unknown:
        raise ConfigError(
            f"slurm: unknown key(s) {', '.join(sorted(map(str, unknown)))}; "
            f"allowed: {', '.join(sorted(_SLURM_KEYS))}"
        )

    options: dict[str, str] = {}
    raw_options = raw.get("options") or {}
    if not isinstance(raw_options, Mapping):
        raise ConfigError("slurm.options: expected a mapping of option to value")
    for raw_key, value in raw_options.items():
        key = str(raw_key)
        if not _SLURM_OPTION.match(key):
            raise ConfigError(f"slurm.options: {key!r} is not a valid option name")
        if key in _SLURM_RESERVED:
            raise ConfigError(f"slurm.options: {key} is set by campaign itself")
        text = str(_scalar(value, f"slurm.options.{key}"))
        # Each option becomes one #SBATCH line; a newline would start another.
        if "\n" in text or "\r" in text:
            raise ConfigError(f"slurm.options.{key}: value contains a line break")
        options[key] = text

    throttle = raw.get("throttle")
    if throttle is not None:
        throttle = _positive_int(throttle, "slurm.throttle")

    max_size = _positive_int(
        raw.get("max_array_size", DEFAULT_MAX_ARRAY_SIZE), "slurm.max_array_size"
    )

    sbatch_raw = raw.get("sbatch", ["sbatch"])
    sbatch = [sbatch_raw] if isinstance(sbatch_raw, str) else sbatch_raw
    if not isinstance(sbatch, list) or not sbatch:
        raise ConfigError("slurm.sbatch: expected a command or a list of arguments")
    if not all(isinstance(part, str) and part for part in sbatch):
        raise ConfigError("slurm.sbatch: every argument must be a non-empty string")

    return SlurmSettings(
        options=options, throttle=throttle, max_array_size=max_size, sbatch=tuple(sbatch)
    )


def _invariants(raw: object) -> tuple[Invariant, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ConfigError("invariants: expected a list")
    try:
        parsed = tuple(
            parse_invariant(spec, f"invariants[{i}]") for i, spec in enumerate(raw)
        )
    except InvariantError as exc:
        raise ConfigError(str(exc)) from exc
    names = [inv.name for inv in parsed]
    duplicated = sorted({n for n in names if names.count(n) > 1})
    if duplicated:
        raise ConfigError(
            f"invariants: {', '.join(duplicated)} declared twice; "
            "give one of them a distinct name"
        )
    return parsed


def _check_keys(raw: Mapping[str, Any]) -> None:
    unknown = set(raw) - _TOP_LEVEL
    if unknown:
        raise ConfigError(
            f"unknown key(s) {', '.join(sorted(map(str, unknown)))}; "
            f"allowed: {', '.join(sorted(_TOP_LEVEL))}"
        )
    missing = [key for key in _REQUIRED if key not in raw]
    if missing:
        raise ConfigError(f"missing required key(s): {', '.join(missing)}")


def _name(raw: object) -> str:
    if not isinstance(raw, str) or not _NAME.match(raw):
        raise ConfigError(
            "name: letters, digits, dots, dashes and underscores only, "
            "starting with a letter or digit"
        )
    return raw


def _seed(raw: object) -> int:
    if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
        raise ConfigError(f"seed: expected a non-negative integer, got {raw!r}")
    return raw


def _output(raw: object, base: Path) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ConfigError("output: expected a directory path")
    return (base / raw).resolve()


def _scenarios(
    raw: Mapping[str, Any], model_ref: str
) -> tuple[tuple[str, ...], dict[str, Scalar], tuple[Scenario, ...]]:
    """The swept names, the constants, and every scenario with its id."""
    if ("grid" in raw) == ("scenarios" in raw):
        raise ConfigError("give exactly one of grid or scenarios")
    if "grid" in raw:
        swept, rows = _swept_from_grid(raw["grid"])
    else:
        swept, rows = _swept_from_list(raw["scenarios"])

    constants = _constants(raw.get("constants"))
    clash = sorted(set(constants) & set(swept))
    if clash:
        raise ConfigError(f"constants: {', '.join(clash)} is also swept")

    scenarios: list[Scenario] = []
    seen: dict[str, int] = {}
    for i, row in enumerate(rows):
        params = {**row, **constants}
        sid = scenario_id(model_ref, params)
        if sid in seen:
            raise ConfigError(f"scenarios[{i}] repeats scenarios[{seen[sid]}]: {row}")
        seen[sid] = i
        scenarios.append(Scenario(id=sid, params=params))
    return swept, constants, tuple(scenarios)


def _backend(raw: object) -> Backend:
    if raw not in ("local", "slurm"):
        raise ConfigError(f"backend: expected local or slurm, got {raw!r}")
    return raw


def _format(raw: object) -> Format:
    if raw not in ("csv", "parquet"):
        raise ConfigError(f"format: expected csv or parquet, got {raw!r}")
    if raw == "parquet" and importlib.util.find_spec("pyarrow") is None:
        # Found now rather than after every scenario has run.
        raise ConfigError("format: parquet needs pyarrow: pip install 'campaign[parquet]'")
    return raw


def parse(raw: object, *, base: Path, source: Path | None = None) -> Campaign:
    """Validate a campaign already loaded from YAML.

    Args:
        raw: the loaded document.
        base: the directory relative paths are resolved against.
        source: the file it came from, for the record.

    Raises:
        ConfigError: anything is missing, unknown or malformed.
    """
    if not isinstance(raw, Mapping):
        raise ConfigError("a campaign file must be a mapping at the top level")
    _check_keys(raw)

    name = _name(raw["name"])
    model_ref, model_file, model_function = _model(raw["model"], base)
    seed = _seed(raw["seed"])
    swept, constants, scenarios = _scenarios(raw, model_ref)
    invariants = _invariants(raw.get("invariants"))

    config_hash = hashlib.sha256(
        canonical_json(
            {
                "name": name,
                "model": model_ref,
                "seed": seed,
                "scenarios": [s.id for s in scenarios],
                "invariants": [asdict(inv) for inv in invariants],
            }
        ).encode("utf-8")
    ).hexdigest()

    return Campaign(
        name=name,
        source=(source or base).resolve(),
        model_ref=model_ref,
        model_file=model_file,
        model_function=model_function,
        seed=seed,
        output=_output(raw["output"], base),
        swept=swept,
        constants=constants,
        scenarios=scenarios,
        invariants=invariants,
        backend=_backend(raw.get("backend", "local")),
        workers=_positive_int(raw.get("workers", 1), "workers"),
        slurm=_slurm(raw.get("slurm")),
        format=_format(raw.get("format", "csv")),
        config_hash=config_hash,
    )


def load(path: str | Path) -> Campaign:
    """Read and validate a campaign file.

    Raises:
        ConfigError: the file cannot be read, is not YAML, or is not a valid
            campaign.
    """
    source = Path(path).resolve()
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read {source}: {exc}") from exc

    loader = _Loader(text)
    try:
        document: Any = loader.get_single_data()
    except yaml.YAMLError as exc:
        raise ConfigError(f"{source.name}: {exc}") from exc
    finally:
        loader.dispose()
    return parse(document, base=source.parent, source=source)
