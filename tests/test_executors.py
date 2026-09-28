"""The two backends: a local process pool, and SLURM job arrays through sbatch.

No cluster is available, so the SLURM path runs against ``fake_sbatch.py``, a
stub written from the sbatch manual page. These tests check what campaign
hands to sbatch and how it reads the answer back; the end-to-end test that
lets the stub run the generated script lives with the command-line tests,
since the script calls the command line.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from campaign.config import load
from campaign.executors import (
    ExecutionError,
    SubmissionError,
    array_chunks,
    parse_job_id,
    run_local,
    run_slurm,
    run_task_file,
)
from campaign.runner import pending, tasks
from campaign.store import read_record

CAMPAIGN = """
    name: toy
    model: model.py:simulate
    seed: 9
    output: out
    grid:
      a: [1.0, 2.0, 3.0]
"""


def _draws(campaign) -> dict[str, float]:
    records = {t.scenario_id: read_record(Path(t.run_dir)) for t in tasks(campaign)}
    return {sid: r.metrics["draw"] for sid, r in records.items() if r is not None}


# --------------------------------------------------------------------------
# local
# --------------------------------------------------------------------------


def test_the_local_backend_runs_every_pending_task(write_campaign):
    campaign = load(write_campaign(CAMPAIGN))
    submission = run_local(pending(campaign), workers=1)
    assert (submission.backend, submission.tasks, submission.finished) == ("local", 3, True)
    assert pending(campaign) == []


def test_a_process_pool_gives_exactly_what_one_process_gives(write_campaign):
    """Seeds follow scenarios, not workers, so the pool changes nothing."""
    one = load(write_campaign(CAMPAIGN))
    run_local(pending(one), workers=1)
    pooled = load(write_campaign(CAMPAIGN.replace("output: out", "output: pooled")))
    run_local(pending(pooled), workers=3)
    assert _draws(pooled) == _draws(one)
    assert len(_draws(one)) == 3


def test_a_worker_that_dies_stops_the_pool_and_keeps_what_finished(write_campaign):
    body = CAMPAIGN.replace("a: [1.0, 2.0, 3.0]", "a: [1.0, 2.0, 3.0]\n      exit: [0, 7]")
    campaign = load(write_campaign(body))
    with pytest.raises(ExecutionError, match="resumes"):
        run_local(pending(campaign), workers=2)
    left = {t.params["exit"] for t in pending(campaign)}
    assert 7 in left  # the ones that killed their worker are still to do


def test_nothing_to_do_does_nothing(write_campaign):
    campaign = load(write_campaign(CAMPAIGN))
    assert run_local([], workers=4).tasks == 0
    assert run_slurm(campaign, []).tasks == 0
    assert not (campaign.output / "slurm").exists()


# --------------------------------------------------------------------------
# what sbatch receives
# --------------------------------------------------------------------------


def _slurm_campaign(write_campaign, fake_sbatch, extra: str = "", output: str = "out"):
    body = CAMPAIGN.replace("output: out", f"output: '{output}'") + (
        "    backend: slurm\n"
        "    slurm:\n"
        "      options: {partition: short, time: '00:05:00'}\n"
        f"      {fake_sbatch.yaml}\n"
        f"{extra}"
    )
    return load(write_campaign(body))


def test_the_submission_carries_the_array_the_logs_and_the_users_options(
    write_campaign, fake_sbatch
):
    campaign = _slurm_campaign(write_campaign, fake_sbatch, extra="      throttle: 2\n")
    submission = run_slurm(campaign, pending(campaign))
    assert submission.job_ids == ("1000",)
    assert submission.finished is False

    (call,) = fake_sbatch.calls()
    argv = call["argv"]
    assert argv[0] == "--parsable"
    assert "--wait" not in argv
    assert "--job-name=toy" in argv
    assert "--array=0-2%2" in argv
    assert "--partition=short" in argv
    assert "--time=00:05:00" in argv
    (output,) = [a for a in argv if a.startswith("--output=")]
    assert output.endswith("/logs/%A_%a.out")


def test_the_job_script_runs_one_frozen_task_per_index(write_campaign, fake_sbatch):
    campaign = _slurm_campaign(write_campaign, fake_sbatch, output="out with spaces")
    run_slurm(campaign, pending(campaign))
    script = fake_sbatch.calls()[0]["script"]
    assert script.startswith("#!/bin/bash\n")
    assert "set -euo pipefail" in script
    assert "-m campaign run-one" in script
    assert '--index "$((0 + SLURM_ARRAY_TASK_ID))"' in script
    assert "'" in script.split("run-one", 1)[1]  # the spaced path is quoted
    assert "#   --array=0-2" in script  # how it was submitted, for the reader


def test_only_pending_tasks_are_frozen_into_the_task_file(write_campaign, fake_sbatch):
    campaign = _slurm_campaign(write_campaign, fake_sbatch)
    run_local(pending(campaign)[:1], workers=1)
    run_slurm(campaign, pending(campaign))
    (task_file,) = (campaign.output / "slurm").rglob("tasks.json")
    frozen = json.loads(task_file.read_text(encoding="utf-8"))["tasks"]
    assert [t["params"]["a"] for t in frozen] == [2.0, 3.0]
    assert "--array=0-1" in fake_sbatch.calls()[0]["argv"]


def test_a_campaign_larger_than_max_array_size_becomes_several_arrays(
    write_campaign, fake_sbatch
):
    body = CAMPAIGN.replace("[1.0, 2.0, 3.0]", "[1.0, 2.0, 3.0, 4.0, 5.0]") + (
        "    backend: slurm\n"
        "    slurm:\n"
        "      max_array_size: 2\n"
        f"      {fake_sbatch.yaml}\n"
    )
    campaign = load(write_campaign(body))
    submission = run_slurm(campaign, pending(campaign))
    assert sorted(submission.job_ids) == ["1000", "1001", "1002"]
    calls = sorted(fake_sbatch.calls(), key=lambda c: c["argv"][-1])
    arrays = [next(a for a in c["argv"] if a.startswith("--array=")) for c in calls]
    offsets = [c["script"].split('"$((', 1)[1].split(" +", 1)[0] for c in calls]
    assert arrays == ["--array=0-1", "--array=0-1", "--array=0-0"]
    assert offsets == ["0", "2", "4"]


@pytest.mark.parametrize(
    ("n", "size", "chunks"),
    [
        (1, 1001, [(0, 1)]),
        (1001, 1001, [(0, 1001)]),
        (2002, 1001, [(0, 1001), (1001, 1001)]),
    ],
)
def test_array_chunks_at_the_edges(n, size, chunks):
    assert array_chunks(n, size) == chunks


# --------------------------------------------------------------------------
# what sbatch answers
# --------------------------------------------------------------------------


@pytest.mark.parametrize("stdout", ["4242", "4242\n", "4242;cluster-a\n", "\n4242\n"])
def test_the_job_id_is_read_with_or_without_a_cluster_name(stdout):
    assert parse_job_id(stdout) == "4242"


@pytest.mark.parametrize("stdout", ["", "Submitted batch job 4242", "4242_7"])
def test_output_without_a_bare_job_id_is_an_error(stdout):
    with pytest.raises(SubmissionError, match="job id"):
        parse_job_id(stdout)


def test_a_multi_cluster_answer_is_understood_end_to_end(
    write_campaign, fake_sbatch, monkeypatch
):
    monkeypatch.setenv("FAKE_SBATCH_MODE", "cluster")
    campaign = _slurm_campaign(write_campaign, fake_sbatch)
    assert run_slurm(campaign, pending(campaign)).job_ids == ("1000",)


def test_a_rejected_job_raises_with_sbatchs_own_message(
    write_campaign, fake_sbatch, monkeypatch
):
    monkeypatch.setenv("FAKE_SBATCH_MODE", "reject")
    campaign = _slurm_campaign(write_campaign, fake_sbatch)
    with pytest.raises(SubmissionError, match="Invalid partition name specified"):
        run_slurm(campaign, pending(campaign))


def test_a_missing_sbatch_says_where_the_backend_has_to_run(write_campaign):
    body = CAMPAIGN + "    backend: slurm\n    slurm: {sbatch: no-such-sbatch-anywhere}\n"
    campaign = load(write_campaign(body))
    with pytest.raises(SubmissionError, match="login node"):
        run_slurm(campaign, pending(campaign))


# --------------------------------------------------------------------------
# the array task's side
# --------------------------------------------------------------------------


def test_editing_the_campaign_after_submission_changes_nothing_queued(
    write_campaign, fake_sbatch
):
    campaign = _slurm_campaign(write_campaign, fake_sbatch)
    run_slurm(campaign, pending(campaign))
    (task_file,) = (campaign.output / "slurm").rglob("tasks.json")

    write_campaign(CAMPAIGN.replace("[1.0, 2.0, 3.0]", "[10.0, 20.0, 30.0]"))
    assert run_task_file(task_file, 1) == 0
    ran = [r for r in map(read_record, (campaign.output / "runs").iterdir()) if r]
    assert [r.params["a"] for r in ran] == [2.0]


def test_a_model_error_is_a_recorded_result_not_a_failed_job(write_campaign, fake_sbatch):
    body = CAMPAIGN.replace("a: [1.0, 2.0, 3.0]", "a: [1.0]\n      crash: [true]")
    campaign = load(write_campaign(body + f"    slurm:\n      {fake_sbatch.yaml}\n"))
    run_slurm(campaign, pending(campaign))
    (task_file,) = (campaign.output / "slurm").rglob("tasks.json")
    assert run_task_file(task_file, 0) == 0
    (record,) = [r for r in map(read_record, (campaign.output / "runs").iterdir()) if r]
    assert record.status == "error"


def test_a_bad_index_or_task_file_fails_the_job(
    write_campaign, fake_sbatch, tmp_path, capsys
):
    campaign = _slurm_campaign(write_campaign, fake_sbatch)
    run_slurm(campaign, pending(campaign))
    (task_file,) = (campaign.output / "slurm").rglob("tasks.json")
    assert run_task_file(task_file, 3) == 2
    assert "outside the 3 tasks" in capsys.readouterr().err
    assert run_task_file(tmp_path / "missing.json", 0) == 2
