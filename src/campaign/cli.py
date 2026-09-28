"""The ``campaign`` command.

    campaign run CAMPAIGN.yaml [--backend local|slurm] [--workers N] [--wait]
    campaign report CAMPAIGN.yaml
    campaign run-one TASKS.json --index N

``run`` runs whatever is not already done and writes the table, manifest and
report. ``report`` writes them from the runs on disk without running
anything, which is what to do once submitted SLURM jobs have finished.
``run-one`` is what each SLURM array task calls; it is not meant to be typed.

Exit codes, so the command can sit in a script or a CI job:

    0    every scenario succeeded
    1    the campaign finished, but some run failed an invariant, raised,
         or never produced a result
    2    the campaign file or the arguments are invalid
    3    running or submitting failed: a worker died, sbatch refused
    130  interrupted; finished runs are kept and a rerun resumes
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from campaign import __version__
from campaign.api import run, write_outputs
from campaign.config import Campaign, ConfigError, load
from campaign.executors import ExecutionError, SubmissionError, run_task_file
from campaign.runner import pending

__all__ = ["main"]

OK, FAILURES, INVALID, EXECUTION, INTERRUPTED = 0, 1, 2, 3, 130


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="campaign",
        description="Run reproducible simulation campaigns, locally or on SLURM.",
    )
    parser.add_argument("--version", action="version", version=f"campaign {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    run_cmd = commands.add_parser(
        "run", help="run what is not done yet, then write results"
    )
    run_cmd.add_argument("config", type=Path, help="the campaign file")
    run_cmd.add_argument(
        "--backend", choices=["local", "slurm"], help="override the campaign file"
    )
    run_cmd.add_argument(
        "--workers", type=int, help="processes for the local backend (overrides the file)"
    )
    run_cmd.add_argument(
        "--wait", action="store_true", help="with SLURM, block until the jobs end"
    )

    report_cmd = commands.add_parser(
        "report", help="write results from the runs on disk, running nothing"
    )
    report_cmd.add_argument("config", type=Path, help="the campaign file")

    one = commands.add_parser("run-one", help="run one frozen task (used by SLURM arrays)")
    one.add_argument("tasks", type=Path, help="the task file written at submission")
    one.add_argument("--index", type=int, required=True, help="which task to run")
    return parser


def _summarise(campaign: Campaign) -> int:
    """Write the outputs, print where they are, and return the exit code."""
    outputs = write_outputs(campaign)
    counts = outputs.result.counts
    print(
        f"{counts['succeeded']} succeeded, {counts['failed']} failed an invariant, "
        f"{counts['error']} raised, {counts['pending']} without a result "
        f"(of {counts['attempted']})"
    )
    for label, path in (
        ("table", outputs.table),
        ("manifest", outputs.manifest),
        ("report", outputs.report),
    ):
        print(f"  {label:8} {path}")
    return OK if counts["succeeded"] == counts["attempted"] else FAILURES


def _run(args: argparse.Namespace) -> int:
    campaign = load(args.config)
    if args.workers is not None and args.workers < 1:
        raise ConfigError("--workers must be at least 1")
    backend = args.backend or campaign.backend
    todo = pending(campaign)
    total = len(campaign.scenarios)
    print(
        f"campaign {campaign.name}: {total} scenarios, {total - len(todo)} already done, "
        f"{len(todo)} to run ({backend})"
    )

    submission = run(
        campaign, backend=args.backend, workers=args.workers, wait=args.wait, todo=todo
    )
    if not submission.finished:
        print(
            f"submitted {submission.tasks} task(s) as SLURM job(s) "
            f"{', '.join(submission.job_ids)}.\n"
            f"When they have finished: campaign report {args.config}"
        )
        return OK
    if submission.exit_code != 0:
        print(
            f"warning: an array task exited with code {submission.exit_code}; the "
            "scenarios it did not record are listed as pending in the report",
            file=sys.stderr,
        )
    return _summarise(campaign)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``campaign`` command. Returns the exit code."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "run-one":
            return run_task_file(args.tasks, args.index)
        if args.command == "report":
            return _summarise(load(args.config))
        return _run(args)
    except ConfigError as exc:
        print(f"campaign: {exc}", file=sys.stderr)
        return INVALID
    except (ExecutionError, SubmissionError) as exc:
        print(f"campaign: {exc}", file=sys.stderr)
        return EXECUTION
    except KeyboardInterrupt:
        print(
            "campaign: interrupted; finished runs are kept, and running again resumes",
            file=sys.stderr,
        )
        return INTERRUPTED
