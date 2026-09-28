"""Gathering every run into one table and one manifest.

Aggregation reads what the runs left on disk and checks every completed run
against the invariants as they are declared *now*. The verdict is never
stored with the run. That is the point of keeping the raw series: resume
skips simulations, never checks. Tighten a tolerance and run the campaign
again, and every run that passed before is re-examined against the new
tolerance without being simulated again.

The table holds only what is determined by the campaign: identifiers,
parameters, outcomes, metrics. Wall times, host names and timestamps go in the
manifest. So aggregating the same runs twice produces the same table, byte
for byte, which is how a reader can check that nothing moved.
"""

from __future__ import annotations

import csv
import io
import json
import platform
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from campaign import __version__
from campaign.config import Campaign, Scalar, Scenario
from campaign.invariants import Violation, check
from campaign.model import file_hash
from campaign.runner import derive_seed
from campaign.store import (
    RunRecord,
    matches,
    read_record,
    read_series,
    run_dir,
    write_atomic,
)

__all__ = [
    "OUTCOMES",
    "CampaignResult",
    "Outcome",
    "ResultsError",
    "RunResult",
    "collect",
    "manifest",
    "table",
    "write_manifest",
    "write_table",
]

Outcome = Literal["succeeded", "failed", "error", "pending"]
OUTCOMES: tuple[Outcome, ...] = ("succeeded", "failed", "error", "pending")

_LEADING = ("scenario_id", "status")
_TRAILING = ("failed_invariants", "error")

Cell = Scalar | None


class ResultsError(ValueError):
    """The runs cannot be put in one table as they stand."""


@dataclass(frozen=True, slots=True)
class RunResult:
    """One scenario's outcome, as of this aggregation.

    Attributes:
        scenario: the scenario.
        run_seed: the seed it runs with.
        outcome: ``succeeded`` (ran, every invariant holds), ``failed`` (ran,
            some invariant does not), ``error`` (the model raised) or
            ``pending`` (no usable run yet).
        record: the stored run, or None if pending.
        violations: the invariants it fails, empty unless ``failed``.
    """

    scenario: Scenario
    run_seed: int
    outcome: Outcome
    record: RunRecord | None
    violations: tuple[Violation, ...]


@dataclass(frozen=True, slots=True)
class CampaignResult:
    """Every scenario's outcome, in declaration order."""

    campaign: Campaign
    model_hash: str
    runs: tuple[RunResult, ...]

    @property
    def counts(self) -> dict[str, int]:
        """How many runs have each outcome, plus the total attempted."""
        counts: dict[str, int] = {"attempted": len(self.runs)}
        for outcome in OUTCOMES:
            counts[outcome] = sum(run.outcome == outcome for run in self.runs)
        return counts


def collect(campaign: Campaign) -> CampaignResult:
    """Read every run from disk and check it against the current invariants."""
    model_hash = file_hash(campaign.model_file)
    runs: list[RunResult] = []
    for scenario in campaign.scenarios:
        seed = derive_seed(campaign.seed, scenario.id)
        directory = run_dir(campaign.output, scenario.id)
        record = read_record(directory)
        current = record is not None and matches(
            record, params=scenario.params, run_seed=seed, model_hash=model_hash
        )
        if record is None or not current:
            runs.append(RunResult(scenario, seed, "pending", None, ()))
            continue
        if record.status == "error":
            runs.append(RunResult(scenario, seed, "error", record, ()))
            continue
        series = read_series(directory)
        if series is None:
            runs.append(RunResult(scenario, seed, "pending", None, ()))
            continue
        violations = tuple(check(campaign.invariants, series))
        outcome: Outcome = "failed" if violations else "succeeded"
        runs.append(RunResult(scenario, seed, outcome, record, violations))
    return CampaignResult(campaign=campaign, model_hash=model_hash, runs=tuple(runs))


def _harmonise(values: Sequence[Cell]) -> list[Cell]:
    """Give a column one type, so a Parquet writer and a reader agree on it.

    A column mixing integers and floats becomes floats; one mixing numbers,
    booleans and strings becomes strings. Missing values stay missing.
    """
    present = [v for v in values if v is not None]
    kinds = {type(v) for v in present}
    if kinds <= {int} or kinds <= {float} or kinds <= {bool} or kinds <= {str}:
        return list(values)
    if kinds <= {int, float}:
        return [None if v is None else float(v) for v in values]
    return [None if v is None else str(v) for v in values]


def table(result: CampaignResult) -> tuple[list[str], list[list[Cell]]]:
    """The results as a header and rows, one row per scenario.

    Columns: ``scenario_id``, ``status``, each swept parameter, each metric
    any run returned (sorted), then ``failed_invariants`` and ``error``.
    Constants are the same in every row and are recorded in the manifest.

    Raises:
        ResultsError: a metric has the name of a column campaign writes.
    """
    swept = list(result.campaign.swept)
    metrics = sorted(
        {
            name
            for run in result.runs
            if run.record is not None
            for name in run.record.metrics
        }
    )
    clash = sorted(set(metrics) & {*_LEADING, *swept, *_TRAILING})
    if clash:
        raise ResultsError(
            f"metric(s) {', '.join(clash)} share a name with a column of the results "
            "table; rename them in the model"
        )

    header = [*_LEADING, *swept, *metrics, *_TRAILING]
    rows: list[list[Cell]] = []
    for run in result.runs:
        values = run.record.metrics if run.record is not None else {}
        error = (
            run.record.error if run.record is not None and run.outcome == "error" else None
        )
        rows.append(
            [
                run.scenario.id,
                run.outcome,
                *(run.scenario.params[name] for name in swept),
                *(values.get(name) for name in metrics),
                "; ".join(v.invariant for v in run.violations),
                error or "",
            ]
        )

    columns = [_harmonise([row[i] for row in rows]) for i in range(len(header))]
    rows = [list(row) for row in zip(*columns, strict=True)] if rows else []
    return header, rows


def _csv_bytes(header: Sequence[str], rows: Sequence[Sequence[Cell]]) -> bytes:
    buffer = io.StringIO()
    # "\n" rather than csv's default "\r\n", so the file matches the rest of
    # the repository's line endings on every platform.
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _parquet_bytes(header: Sequence[str], rows: Sequence[Sequence[Cell]]) -> bytes:
    import pyarrow as pa  # noqa: PLC0415 - optional dependency, checked at load time
    import pyarrow.parquet as pq  # noqa: PLC0415

    columns = {name: [row[i] for row in rows] for i, name in enumerate(header)}
    sink = pa.BufferOutputStream()
    pq.write_table(pa.table(columns), sink)
    return bytes(sink.getvalue().to_pybytes())


def write_table(result: CampaignResult) -> Path:
    """Write the results table in the campaign's format. Returns its path."""
    header, rows = table(result)
    campaign = result.campaign
    if campaign.format == "parquet":
        path, data = campaign.output / "results.parquet", _parquet_bytes(header, rows)
    else:
        path, data = campaign.output / "results.csv", _csv_bytes(header, rows)
    write_atomic(path, data)
    return path


def _git_state(directory: Path) -> dict[str, Any] | None:
    """The commit of the repository holding the model, if there is one.

    Measured when the manifest is written, not when the runs happened, so it
    describes the code only if nothing was committed in between. What ran is
    pinned by the model file's hash, which each run records for itself.
    """
    git = shutil.which("git")
    if git is None:
        return None
    try:
        head = subprocess.run(  # noqa: S603 - fixed arguments, no shell
            [git, "-C", str(directory), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        changes = subprocess.run(  # noqa: S603 - fixed arguments, no shell
            [git, "-C", str(directory), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return {"commit": head, "uncommitted_changes": bool(changes.strip())}


def manifest(result: CampaignResult) -> dict[str, Any]:
    """Everything needed to say what produced the table.

    The code version (this package's version, the model file's hash, and the
    git commit when there is one), the config hash, the campaign seed, and
    for every run its own seed, outcome and wall time.
    """
    campaign = result.campaign
    return {
        "campaign": campaign.name,
        "config_hash": campaign.config_hash,
        "seed": campaign.seed,
        "code": {
            "campaign_version": __version__,
            "model": campaign.model_ref,
            "model_sha256": result.model_hash,
            "git": _git_state(campaign.model_file.parent),
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "constants": campaign.constants,
        "invariants": [inv.name for inv in campaign.invariants],
        "counts": result.counts,
        "runs": [
            {
                "scenario_id": run.scenario.id,
                "run_seed": run.run_seed,
                "outcome": run.outcome,
                "wall_time_s": None if run.record is None else run.record.wall_time_s,
                "finished_at": None if run.record is None else run.record.finished_at,
                "host": None if run.record is None else run.record.host,
            }
            for run in result.runs
        ],
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def write_manifest(result: CampaignResult) -> Path:
    """Write ``manifest.json``. Returns its path."""
    path = result.campaign.output / "manifest.json"
    text = json.dumps(manifest(result), indent=2, allow_nan=False)
    write_atomic(path, (text + "\n").encode("utf-8"))
    return path
