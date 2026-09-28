"""Running one scenario: the unit both backends execute.

A local worker calls :func:`execute` once per scenario; a SLURM array task
calls it once, for its own index. Either way the result lands in the same
place in the same form, so everything downstream is indifferent to where a
run happened.

A model that raises does not stop the campaign. The exception is recorded
with the run, the run is marked as an error, and the next scenario starts.
Only ``KeyboardInterrupt`` and ``SystemExit`` get through, so Ctrl-C still
stops a campaign.
"""

from __future__ import annotations

import socket
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from campaign.config import Campaign, Scalar
from campaign.model import ModelError, RunOutput, file_hash, load_model
from campaign.store import RunRecord, is_current, read_record, run_dir, write_run

__all__ = ["Task", "derive_seed", "execute", "pending", "tasks"]


@dataclass(frozen=True, slots=True)
class Task:
    """Everything needed to run one scenario, without the campaign file.

    The SLURM backend writes these to disk when it submits, and each array
    task reads its own back. Editing the campaign file while jobs wait in the
    queue therefore cannot change what they run.

    Attributes:
        scenario_id: the scenario's content identifier.
        params: the parameters, swept and constant.
        run_seed: the seed the model receives.
        model_file: absolute path of the model file.
        model_function: the function to call in it.
        run_dir: where the run's files go.
    """

    scenario_id: str
    params: dict[str, Scalar]
    run_seed: int
    model_file: str
    model_function: str
    run_dir: str

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable form, for the SLURM task file."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        """The inverse of :meth:`to_dict`."""
        return cls(**data)


def derive_seed(campaign_seed: int, scenario_id: str) -> int:
    """The seed one scenario's run receives.

    Derived from the scenario's identity rather than its position, so a
    scenario gets the same seed whatever order scenarios run in, however many
    workers share them, and whether or not other scenarios are added. It is a
    32-bit integer because every seeding interface accepts one, including
    numpy's legacy ``RandomState``, which rejects anything larger.

    >>> derive_seed(1, "00000000000000aa") == derive_seed(1, "00000000000000aa")
    True
    >>> derive_seed(1, "00000000000000aa") != derive_seed(2, "00000000000000aa")
    True
    """
    sequence = np.random.SeedSequence([campaign_seed, int(scenario_id, 16)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def tasks(campaign: Campaign) -> list[Task]:
    """Every scenario of the campaign as a task, in declaration order."""
    return [
        Task(
            scenario_id=s.id,
            params=dict(s.params),
            run_seed=derive_seed(campaign.seed, s.id),
            model_file=str(campaign.model_file),
            model_function=campaign.model_function,
            run_dir=str(run_dir(campaign.output, s.id)),
        )
        for s in campaign.scenarios
    ]


def pending(campaign: Campaign) -> list[Task]:
    """The tasks that still need to run: no current record for them.

    A record is current if the run completed with the same parameters, seed
    and model file. So a run that raised is retried, and editing the model
    file reruns everything it produced.
    """
    model_hash = file_hash(campaign.model_file)
    return [
        task
        for task in tasks(campaign)
        if not is_current(
            read_record(Path(task.run_dir)),
            params=task.params,
            run_seed=task.run_seed,
            model_hash=model_hash,
        )
    ]


def _run_model(task: Task) -> tuple[dict[str, NDArray[np.float64]], dict[str, float]]:
    """Call the model and return its series and metrics, or raise."""
    fn = load_model(Path(task.model_file), task.model_function)
    output = fn(dict(task.params), task.run_seed)
    if not isinstance(output, RunOutput):
        raise ModelError(
            f"{task.model_function} returned {type(output).__name__}, "
            "expected campaign.RunOutput"
        )
    return output.arrays(), output.scalars()


def execute(task: Task) -> RunRecord:
    """Run one task and store its result. Never raises for a model failure."""
    directory = Path(task.run_dir)
    # Remove any earlier record first, so that a crash from here on leaves the
    # scenario looking unfinished rather than looking like the old result.
    (directory / "record.json").unlink(missing_ok=True)

    start = time.perf_counter()
    model_hash = ""
    series: dict[str, NDArray[np.float64]] | None = None
    metrics: dict[str, float] = {}
    error = tb = None
    try:
        model_hash = file_hash(Path(task.model_file))
        series, metrics = _run_model(task)
    except Exception as exc:  # noqa: BLE001 - a failing model is a result, not a crash
        series, metrics = None, {}
        error = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc()

    record = RunRecord(
        scenario_id=task.scenario_id,
        params=dict(task.params),
        run_seed=task.run_seed,
        model_hash=model_hash,
        status="completed" if error is None else "error",
        metrics=metrics,
        error=error,
        traceback=tb,
        wall_time_s=time.perf_counter() - start,
        finished_at=datetime.now(UTC).isoformat(timespec="seconds"),
        host=socket.gethostname(),
    )
    write_run(directory, record, series)
    return record
