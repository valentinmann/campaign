"""The command line, and the SLURM path from submission to report.

The end-to-end SLURM tests are the closest this package gets to a cluster:
the stub sbatch runs the generated job script through bash, once per array
index, and the script calls ``python -m campaign run-one``. Everything
between the campaign file and the report is the real code; only the
scheduler is pretended. The output directory has a space in its name on
purpose.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from campaign import __version__
from campaign.cli import main
from tests.conftest import usable_bash

CAMPAIGN = """
    name: toy
    model: model.py:simulate
    seed: 21
    output: 'out dir'
    grid:
      a: [1.0, 2.0, 3.0]
    invariants:
      - {check: conserved, series: [x, y], rtol: 1e-9}
      - {check: nonnegative, series: [x, y]}
"""

needs_bash = pytest.mark.skipif(
    usable_bash() is None, reason="the stub runs the job script through bash"
)


def _slurm(fake_sbatch, body: str = CAMPAIGN) -> str:
    return body + f"    backend: slurm\n    slurm:\n      {fake_sbatch.yaml}\n"


# --------------------------------------------------------------------------
# local runs
# --------------------------------------------------------------------------


def test_a_clean_run_succeeds_and_says_where_everything_is(
    write_campaign, capsys, tmp_path
):
    assert main(["run", str(write_campaign(CAMPAIGN))]) == 0
    out = capsys.readouterr().out
    assert "3 scenarios, 0 already done, 3 to run (local)" in out
    assert "3 succeeded, 0 failed an invariant, 0 raised" in out
    for name in ("results.csv", "manifest.json", "report.md"):
        assert (tmp_path / "out dir" / name).exists()


def test_running_again_simulates_nothing_and_changes_nothing(
    write_campaign, capsys, tmp_path
):
    path = str(write_campaign(CAMPAIGN))
    main(["run", path])
    table = (tmp_path / "out dir" / "results.csv").read_bytes()
    capsys.readouterr()
    assert main(["run", path]) == 0
    assert "3 already done, 0 to run" in capsys.readouterr().out
    assert (tmp_path / "out dir" / "results.csv").read_bytes() == table


def test_a_failed_invariant_finishes_the_campaign_and_exits_1(write_campaign, capsys):
    body = CAMPAIGN.replace(
        "a: [1.0, 2.0, 3.0]", "a: [1.0, 2.0, 3.0]\n      leak: [0.0, 0.5]"
    )
    assert main(["run", str(write_campaign(body))]) == 1
    assert "3 succeeded, 3 failed an invariant" in capsys.readouterr().out


def test_an_invalid_campaign_exits_2_with_a_message_not_a_traceback(write_campaign, capsys):
    assert main(["run", str(write_campaign(CAMPAIGN.replace("seed: 21", "seed: -1")))]) == 2
    err = capsys.readouterr().err
    assert err.startswith("campaign: seed:")
    assert "Traceback" not in err


def test_the_backend_flag_overrides_the_file(write_campaign, fake_sbatch):
    path = str(write_campaign(_slurm(fake_sbatch)))
    assert main(["run", path, "--backend", "local", "--workers", "2"]) == 0
    assert fake_sbatch.calls() == []


def test_zero_workers_is_refused(write_campaign, capsys):
    assert main(["run", str(write_campaign(CAMPAIGN)), "--workers", "0"]) == 2
    assert "--workers" in capsys.readouterr().err


def test_report_runs_nothing_and_counts_what_is_missing(write_campaign, capsys, tmp_path):
    assert main(["report", str(write_campaign(CAMPAIGN))]) == 1
    assert "0 succeeded, 0 failed an invariant, 0 raised, 3 without a result" in (
        capsys.readouterr().out
    )
    assert not (tmp_path / "out dir" / "runs").exists()


def test_the_version_and_the_module_entry_point(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert capsys.readouterr().out.strip() == f"campaign {__version__}"
    proc = subprocess.run(
        [sys.executable, "-m", "campaign", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == f"campaign {__version__}"


# --------------------------------------------------------------------------
# SLURM, end to end
# --------------------------------------------------------------------------


def test_submitting_without_waiting_returns_at_once_with_the_job_id(
    write_campaign, fake_sbatch, capsys, tmp_path
):
    assert main(["run", str(write_campaign(_slurm(fake_sbatch)))]) == 0
    out = capsys.readouterr().out
    assert "as SLURM job(s) 1000" in out
    assert "campaign report" in out
    assert not (tmp_path / "out dir" / "results.csv").exists()


@needs_bash
def test_a_slurm_campaign_runs_from_submission_to_report(
    write_campaign, fake_sbatch, capsys, tmp_path
):
    """The stub runs the real job script, which runs the real run-one."""
    path = str(write_campaign(_slurm(fake_sbatch)))
    assert main(["run", path, "--wait"]) == 0
    out = capsys.readouterr().out
    assert "3 succeeded" in out
    assert "--wait" in fake_sbatch.calls()[0]["argv"]
    logs = sorted((tmp_path / "out dir" / "slurm").rglob("logs/1000_*.out"))
    assert len(logs) == 3
    # Each array task says what it ran, so a log opened on a cluster explains itself.
    assert "task 1, scenario" in logs[1].read_text(encoding="utf-8")
    assert "completed in" in logs[1].read_text(encoding="utf-8")

    # Everything is done, so a second submission sends nothing to sbatch.
    assert main(["run", path, "--wait"]) == 0
    assert len(fake_sbatch.calls()) == 1


def test_submitted_jobs_are_collected_by_report_once_they_finish(
    write_campaign, fake_sbatch, capsys, tmp_path
):
    path = str(write_campaign(_slurm(fake_sbatch)))
    main(["run", path])
    (task_file,) = (tmp_path / "out dir" / "slurm").rglob("tasks.json")
    for index in range(3):  # what the scheduler would do later
        assert main(["run-one", str(task_file), "--index", str(index)]) == 0
    capsys.readouterr()
    assert main(["report", path]) == 0
    assert "3 succeeded" in capsys.readouterr().out


@needs_bash
def test_an_array_task_that_dies_is_reported_as_pending(
    write_campaign, fake_sbatch, capsys
):
    body = CAMPAIGN.replace("a: [1.0, 2.0, 3.0]", "a: [1.0, 2.0]\n      exit: [0, 9]")
    assert main(["run", str(write_campaign(_slurm(fake_sbatch, body))), "--wait"]) == 1
    captured = capsys.readouterr()
    assert "exited with code 9" in captured.err
    assert (
        "2 succeeded, 0 failed an invariant, 0 raised, 2 without a result" in captured.out
    )


def test_a_rejected_submission_exits_3(write_campaign, fake_sbatch, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_SBATCH_MODE", "reject")
    assert main(["run", str(write_campaign(_slurm(fake_sbatch)))]) == 3
    assert "Invalid partition" in capsys.readouterr().err


def test_run_one_with_a_bad_index_fails_the_array_task(
    write_campaign, fake_sbatch, tmp_path
):
    main(["run", str(write_campaign(_slurm(fake_sbatch)))])
    (task_file,) = (tmp_path / "out dir" / "slurm").rglob("tasks.json")
    assert main(["run-one", str(task_file), "--index", "7"]) == 2
