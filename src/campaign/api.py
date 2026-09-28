"""The two things a campaign does: run what is pending, and write up the result.

These are what the command line calls, and what a script or a notebook can
call instead::

    from campaign.api import run, write_outputs
    from campaign.config import load

    campaign = load("demo/sir.yaml")
    run(campaign)
    outputs = write_outputs(campaign)
    print(outputs.result.counts)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from campaign.config import Backend, Campaign
from campaign.executors import Submission, run_local, run_slurm
from campaign.report import write_report
from campaign.results import CampaignResult, collect, write_manifest, write_table
from campaign.runner import Task, pending

__all__ = ["Outputs", "run", "write_outputs"]


@dataclass(frozen=True, slots=True)
class Outputs:
    """What :func:`write_outputs` produced."""

    result: CampaignResult
    table: Path
    manifest: Path
    report: Path


def run(
    campaign: Campaign,
    *,
    backend: Backend | None = None,
    workers: int | None = None,
    wait: bool = False,
    todo: Sequence[Task] | None = None,
) -> Submission:
    """Run every scenario that is not already done.

    Args:
        campaign: the loaded campaign.
        backend: overrides the campaign file's ``backend``.
        workers: overrides the campaign file's ``workers``, for the local
            backend.
        wait: for SLURM, block until the jobs have ended.
        todo: the pending tasks, if already computed. Working them out reads
            every stored series back, so a caller that has done it once
            should not make this do it again.
    """
    if todo is None:
        todo = pending(campaign)
    if (backend or campaign.backend) == "slurm":
        return run_slurm(campaign, todo, wait=wait)
    return run_local(todo, workers=workers or campaign.workers)


def write_outputs(campaign: Campaign) -> Outputs:
    """Check every stored run and write the table, the manifest and the report."""
    result = collect(campaign)
    return Outputs(
        result=result,
        table=write_table(result),
        manifest=write_manifest(result),
        report=write_report(result),
    )
