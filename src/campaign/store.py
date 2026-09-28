"""Where runs are kept, and how a run is known to have finished.

Layout of the output directory::

    runs/<scenario id>/series.npz    the raw series the invariants are checked against
    runs/<scenario id>/record.json   written last; its presence means the run finished
    results.csv | results.parquet    one row per scenario
    manifest.json                    code version, config hash, seeds, wall times
    report.md                        what failed, and why

Every file is written under a temporary name in its final directory and moved
into place with ``os.replace``, which is atomic on POSIX and on Windows. A run
killed halfway therefore leaves a stray temporary file, never a truncated
record, and can only ever look unfinished. The record is written after the
series for the same reason, and a rerun deletes the old record before it
starts, so no record ever sits next to a series it did not produce.

``record.json`` is read by this package only. It uses Python's JSON dialect,
which writes NaN and infinity as bare tokens, because a model is allowed to
return a NaN metric and the record should keep it rather than turn it into
something else.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from campaign.config import Scalar

__all__ = [
    "RECORD",
    "SERIES",
    "RunRecord",
    "Status",
    "load_finished",
    "matches",
    "read_record",
    "read_series",
    "run_dir",
    "write_atomic",
    "write_run",
]

RECORD = "record.json"
SERIES = "series.npz"

Status = Literal["completed", "error"]


@dataclass(frozen=True, slots=True)
class RunRecord:
    """What is known about one finished run.

    Attributes:
        scenario_id: the scenario's content identifier.
        params: the parameters it ran with.
        run_seed: the seed the model received.
        model_hash: SHA-256 of the model file that actually ran.
        status: ``completed`` if the model returned, ``error`` if it raised.
            Whether it satisfies the invariants is not stored here: that is
            decided afresh every time the campaign is aggregated.
        metrics: the scalars the model returned.
        error: for an error, the exception in one line.
        traceback: for an error, the full traceback.
        wall_time_s: seconds from loading the model to storing the result.
        finished_at: UTC timestamp, ISO 8601.
        host: the machine it ran on.
    """

    scenario_id: str
    params: dict[str, Scalar]
    run_seed: int
    model_hash: str
    status: Status
    metrics: dict[str, float]
    error: str | None
    traceback: str | None
    wall_time_s: float
    finished_at: str
    host: str


def run_dir(output: Path, scenario_id: str) -> Path:
    """The directory holding one scenario's files."""
    return output / "runs" / scenario_id


def write_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` so that ``path`` is either absent, old, or complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_run(
    directory: Path, record: RunRecord, series: Mapping[str, NDArray[np.float64]] | None
) -> None:
    """Store a finished run: the series first, the record last."""
    series_path = directory / SERIES
    if series is None:
        series_path.unlink(missing_ok=True)
    else:
        # Two traps in np.savez. It appends ".npz" to a file name that lacks
        # one, which would turn the temporary name into a different file, so
        # it writes to a buffer. And it takes arrays as keyword arguments next
        # to its own `file` and `allow_pickle`, so a series named either would
        # collide with them; the arrays go in positionally and their names in
        # a separate array.
        names = list(series)
        buffer = io.BytesIO()
        np.savez(buffer, *(series[n] for n in names), names=np.array(names, dtype=str))
        write_atomic(series_path, buffer.getvalue())
    text = json.dumps(asdict(record), indent=2, sort_keys=True)
    write_atomic(directory / RECORD, (text + "\n").encode("utf-8"))


def read_record(directory: Path) -> RunRecord | None:
    """The run's record, or None if it is missing, truncated or unreadable.

    An unreadable record is treated exactly like a missing one: the run is
    not finished, and resuming reruns it.
    """
    try:
        data = json.loads((directory / RECORD).read_text(encoding="utf-8"))
        return RunRecord(**data)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return None


def read_series(directory: Path) -> dict[str, NDArray[np.float64]] | None:
    """The run's stored series, or None if they cannot be read.

    Loaded with ``allow_pickle=False``: a results directory can come from
    anywhere, and a pickle inside an ``.npz`` file is code.

    The file is opened here rather than by ``np.load``. Given a path, np.load
    opens the file itself, and when the archive turns out to be truncated it
    raises before handing the handle to anything that would close it. On
    Windows a handle left open blocks the ``os.replace`` that rewrites the
    file when the run is redone; the test suite, which treats warnings as
    errors, caught it as an unclosed-file warning.
    """
    try:
        with (directory / SERIES).open("rb") as f, np.load(f, allow_pickle=False) as data:
            names = [str(n) for n in data["names"]]
            return {
                name: np.asarray(data[f"arr_{i}"], dtype=np.float64)
                for i, name in enumerate(names)
            }
    # A truncated archive raises BadZipFile, which is neither an OSError nor a
    # ValueError; a truncated member raises EOFError; a file this package did
    # not write has no "names" and raises KeyError.
    except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile):
        return None


def matches(
    record: RunRecord | None,
    *,
    params: Mapping[str, Scalar],
    run_seed: int,
    model_hash: str,
) -> bool:
    """Whether ``record`` was produced by exactly these inputs, whatever its status."""
    return (
        record is not None
        and record.params == dict(params)
        and record.run_seed == run_seed
        and record.model_hash == model_hash
    )


def load_finished(
    directory: Path,
    *,
    params: Mapping[str, Scalar],
    run_seed: int,
    model_hash: str,
) -> tuple[RunRecord, dict[str, NDArray[np.float64]]] | None:
    """The record and series of a run that completed with these inputs.

    None means the scenario has to run again: there is no record, the model
    raised, the record came from other parameters, another seed or another
    version of the model file, or the series can no longer be read. The last
    case matters as much as the others. A sound record next to a truncated
    series is a run that cannot be re-checked, and treating it as finished
    would leave it unchecked for good.
    """
    record = read_record(directory)
    if record is None or record.status != "completed":
        return None
    if not matches(record, params=params, run_seed=run_seed, model_hash=model_hash):
        return None
    series = read_series(directory)
    if series is None:
        return None
    return record, series
