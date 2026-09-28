"""Running the pending tasks: here, or as a SLURM job array.

Both backends take the same tasks and leave the same files behind, so nothing
downstream can tell which one ran. The local backend runs them in a pool of
processes and returns when they are done. The SLURM backend freezes the tasks
into a file, writes a job script, submits it with ``sbatch``, and either
returns the job ids at once or, with ``wait``, blocks until the jobs end.

What the SLURM backend assumes about ``sbatch`` comes from its documentation,
not from experience on a cluster, and the tests exercise it against a stub
built from the same documentation:

* ``--parsable`` prints the job id, followed by ``;cluster`` on a
  multi-cluster installation;
* array indices run from 0 to MaxArraySize - 1, and MaxArraySize defaults to
  1001, so a larger campaign is split into several arrays;
* with ``--wait``, sbatch exits with the highest exit code of any array task.
"""

from __future__ import annotations

import json
import multiprocessing
import re
import shlex
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from campaign import __version__
from campaign.config import Campaign
from campaign.runner import Task, execute
from campaign.store import write_atomic

__all__ = [
    "ExecutionError",
    "Submission",
    "SubmissionError",
    "array_chunks",
    "job_script",
    "parse_job_id",
    "run_local",
    "run_slurm",
    "run_task_file",
    "sbatch_arguments",
]

_JOB_ID = re.compile(r"^\d+$")


class ExecutionError(RuntimeError):
    """The local backend could not finish: a worker process died."""


class SubmissionError(RuntimeError):
    """sbatch could not be run, or did not accept the job."""


@dataclass(frozen=True, slots=True)
class Submission:
    """What a backend did.

    Attributes:
        backend: ``local`` or ``slurm``.
        tasks: how many tasks were run or submitted.
        job_ids: the SLURM job ids, one per array; empty for local runs.
        finished: whether the tasks have ended when this is returned.
        exit_code: with SLURM and ``wait``, the highest exit code of any
            array task. Non-zero means some task died before it could record
            a result; those scenarios show up as pending.
    """

    backend: str
    tasks: int
    job_ids: tuple[str, ...] = ()
    finished: bool = True
    exit_code: int = 0


# ---------------------------------------------------------------------------
# local
# ---------------------------------------------------------------------------


def run_local(tasks: Sequence[Task], *, workers: int) -> Submission:
    """Run every task on this machine and return when all have finished.

    With one worker the tasks run in this process, which keeps a debugger and
    a traceback usable. With more, they run in a pool of processes started
    with ``spawn`` on every platform. Linux defaults to ``fork`` before Python
    3.14, and forking a process that holds threads, numpy's BLAS for one, can
    deadlock; Python 3.12 warns about exactly that. Spawn behaves the same on
    Linux, macOS and Windows, which also makes their results comparable.

    Raises:
        ExecutionError: a worker process died. Runs that finished are kept,
            so running the campaign again resumes from there.
    """
    if not tasks:
        return Submission(backend="local", tasks=0)
    if workers == 1 or len(tasks) == 1:
        for task in tasks:
            execute(task)
        return Submission(backend="local", tasks=len(tasks))

    context = multiprocessing.get_context("spawn")
    try:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)), mp_context=context
        ) as pool:
            for _ in pool.map(execute, tasks):
                pass
    except BrokenProcessPool as exc:
        raise ExecutionError(
            "a worker process died, most likely killed by the operating system or by "
            "the model exiting the interpreter; runs that finished are kept, and "
            "running the campaign again resumes from them"
        ) from exc
    return Submission(backend="local", tasks=len(tasks))


# ---------------------------------------------------------------------------
# SLURM
# ---------------------------------------------------------------------------


def array_chunks(n_tasks: int, max_array_size: int) -> list[tuple[int, int]]:
    """Split ``n_tasks`` into arrays SLURM will accept.

    Returns ``(offset, size)`` pairs. Each array uses indices ``0`` to
    ``size - 1``, and task ``offset + index`` of the task file.

    >>> array_chunks(2500, 1001)
    [(0, 1001), (1001, 1001), (2002, 498)]
    """
    return [
        (start, min(max_array_size, n_tasks - start))
        for start in range(0, n_tasks, max_array_size)
    ]


def parse_job_id(stdout: str) -> str:
    r"""The job id from ``sbatch --parsable`` output.

    >>> parse_job_id("4242\n"), parse_job_id("4242;west\n")
    ('4242', '4242')

    Raises:
        SubmissionError: no job id in the output.
    """
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    job_id = lines[-1].split(";", 1)[0].strip() if lines else ""
    if not _JOB_ID.match(job_id):
        raise SubmissionError(f"could not read a job id from sbatch output: {stdout!r}")
    return job_id


def sbatch_arguments(campaign: Campaign, log_dir: Path, size: int) -> list[str]:
    """The options campaign passes to sbatch for one array.

    Passed as arguments rather than written as ``#SBATCH`` lines, because
    sbatch parses those lines with quoting rules of its own, and a path with
    a space in it, which is not unusual on a laptop, has to survive them.
    """
    array = f"0-{size - 1}"
    if campaign.slurm.throttle is not None:
        array += f"%{campaign.slurm.throttle}"
    return [
        f"--job-name={campaign.name}",
        f"--array={array}",
        f"--output={log_dir.as_posix()}/%A_%a.out",
        *(f"--{key}={value}" for key, value in campaign.slurm.options.items()),
    ]


def job_script(task_file: Path, offset: int, arguments: Sequence[str]) -> str:
    """The batch script for one array: run the task this index points at.

    Paths are written in POSIX form and quoted for the shell, so a space in
    them is harmless. The sbatch options are repeated as comments for whoever
    opens the script to see how it was submitted.
    """
    python = shlex.quote(Path(sys.executable).as_posix())
    tasks = shlex.quote(task_file.as_posix())
    submitted = "\n".join(f"#   {arg}" for arg in arguments)
    return (
        "#!/bin/bash\n"
        f"# Written by campaign {__version__}. Submitted with these sbatch options:\n"
        f"{submitted}\n"
        "# Each array task runs one scenario: entry OFFSET + SLURM_ARRAY_TASK_ID\n"
        "# of the task file, which was frozen when the array was submitted.\n"
        "set -euo pipefail\n"
        f"exec {python} -m campaign run-one {tasks} "
        f'--index "$(({offset} + SLURM_ARRAY_TASK_ID))"\n'
    )


def _sbatch(
    command: Sequence[str], arguments: Sequence[str], script: Path, *, wait: bool
) -> tuple[str, int]:
    """Run sbatch for one script. Returns the job id and sbatch's exit code."""
    argv = [*command, "--parsable", *(["--wait"] if wait else []), *arguments, str(script)]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, check=False)  # noqa: S603 - no shell; the command is the user's own sbatch
    except FileNotFoundError as exc:
        raise SubmissionError(
            f"{command[0]!r} not found; the SLURM backend has to run where sbatch is "
            "installed, usually a cluster's login node"
        ) from exc

    if proc.returncode != 0 and not wait:
        raise SubmissionError(
            f"sbatch exited with code {proc.returncode}: "
            f"{proc.stderr.strip() or proc.stdout.strip() or 'no message'}"
        )
    # With --wait a non-zero code can mean the job ran and some task failed,
    # in which case the id was printed at submission. No id means sbatch
    # rejected the job outright.
    try:
        job_id = parse_job_id(proc.stdout)
    except SubmissionError as exc:
        raise SubmissionError(
            f"sbatch exited with code {proc.returncode}: "
            f"{proc.stderr.strip() or 'no job id and no message'}"
        ) from exc
    return job_id, proc.returncode


def run_slurm(
    campaign: Campaign, tasks: Sequence[Task], *, wait: bool = False
) -> Submission:
    """Submit the tasks as one or more SLURM job arrays.

    The tasks are written to a file first and each array task reads its own
    entry back, so editing the campaign file while the jobs queue changes
    nothing they run.

    Raises:
        SubmissionError: sbatch is missing, or rejected a job.
    """
    if not tasks:
        return Submission(backend="slurm", tasks=0)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    batch = campaign.output / "slurm" / stamp
    task_file = batch / "tasks.json"
    payload = {"campaign": campaign.name, "tasks": [task.to_dict() for task in tasks]}
    write_atomic(task_file, (json.dumps(payload, indent=1) + "\n").encode("utf-8"))

    submissions: list[tuple[list[str], Path]] = []
    for k, (offset, size) in enumerate(
        array_chunks(len(tasks), campaign.slurm.max_array_size)
    ):
        arguments = sbatch_arguments(campaign, batch / "logs", size)
        script = batch / f"job_{k:03d}.sh"
        write_atomic(script, job_script(task_file, offset, arguments).encode("utf-8"))
        submissions.append((arguments, script))
    (batch / "logs").mkdir(exist_ok=True)

    def submit(submission: tuple[list[str], Path]) -> tuple[str, int]:
        arguments, script = submission
        return _sbatch(campaign.slurm.sbatch, arguments, script, wait=wait)

    # One thread per array, so that with --wait they run side by side rather
    # than each waiting for the one before it to finish.
    with ThreadPoolExecutor(max_workers=len(submissions)) as pool:
        outcomes = list(pool.map(submit, submissions))
    return Submission(
        backend="slurm",
        tasks=len(tasks),
        job_ids=tuple(job_id for job_id, _ in outcomes),
        finished=wait,
        exit_code=max(code for _, code in outcomes),
    )


def run_task_file(task_file: Path, index: int) -> int:
    """Run one entry of a frozen task file. What each array task calls.

    Returns 0 once the run is recorded, whether the model succeeded or
    raised: a model error is a result, and it is in the record. Anything
    else, a missing file or an index past the end, is a failure of the job
    itself and returns 2.

    It prints one line saying which scenario ran and how it ended, because
    that line is what lands in the array task's log file, and an empty log
    is no help to whoever opens it on the cluster.
    """
    try:
        entries = json.loads(task_file.read_text(encoding="utf-8"))["tasks"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"campaign run-one: cannot read {task_file}: {exc}", file=sys.stderr)  # noqa: T201
        return 2
    if not 0 <= index < len(entries):
        print(  # noqa: T201
            f"campaign run-one: index {index} is outside the {len(entries)} tasks "
            f"in {task_file}",
            file=sys.stderr,
        )
        return 2
    record = execute(Task.from_dict(entries[index]))
    outcome = "completed" if record.error is None else f"raised {record.error}"
    print(  # noqa: T201
        f"campaign run-one: task {index}, scenario {record.scenario_id}, "
        f"{outcome} in {record.wall_time_s:.3g} s"
    )
    return 0
