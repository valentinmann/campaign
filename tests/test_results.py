"""Aggregation: every run re-checked, one table, one manifest.

The central test here is the one about tolerances. A verdict is never stored
with a run; it is recomputed from the stored series every time. So tightening
a tolerance must fail a run that passed before, without that run being
simulated again, and the test checks both halves.
"""

from __future__ import annotations

import csv
import io
import json
import shutil
import subprocess

import pytest

from campaign.config import load
from campaign.results import (
    ResultsError,
    collect,
    manifest,
    table,
    write_manifest,
    write_table,
)
from campaign.runner import execute, pending, tasks

MIXED = """
    name: toy
    model: model.py:simulate
    seed: 3
    output: out
    scenarios:
      - {a: 1.0, leak: 0.0, crash: false}
      - {a: 2.0, leak: 1.0e-3, crash: false}
      - {a: 3.0, leak: 0.0, crash: true}
      - {a: 4.0, leak: 0.0, crash: false}
    invariants:
      - {check: conserved, series: [x, y], rtol: 1e-9}
      - {check: nonnegative, series: [x, y]}
"""


def _run(campaign, *, skip_last: bool = False) -> None:
    todo = pending(campaign)
    for task in todo[:-1] if skip_last else todo:
        execute(task)


@pytest.fixture
def mixed(write_campaign):
    campaign = load(write_campaign(MIXED))
    _run(campaign, skip_last=True)
    return campaign


# --------------------------------------------------------------------------
# outcomes
# --------------------------------------------------------------------------


def test_each_outcome_is_told_apart(mixed):
    result = collect(mixed)
    assert [run.outcome for run in result.runs] == [
        "succeeded",
        "failed",
        "error",
        "pending",
    ]
    assert result.counts == {
        "attempted": 4,
        "succeeded": 1,
        "failed": 1,
        "error": 1,
        "pending": 1,
    }
    (violation,) = result.runs[1].violations
    assert violation.invariant == "conserved(x+y)"


def test_tightening_a_tolerance_fails_a_passing_run_without_rerunning_it(write_campaign):
    loose = MIXED.replace("rtol: 1e-9", "rtol: 1e-2")
    campaign = load(write_campaign(loose))
    _run(campaign)
    assert collect(campaign).runs[1].outcome == "succeeded"

    stamps = {p: p.stat().st_mtime_ns for p in (campaign.output / "runs").rglob("*")}
    strict = load(write_campaign(MIXED))
    assert [t.params["crash"] for t in pending(strict)] == [True]  # only the model error
    assert collect(strict).runs[1].outcome == "failed"
    assert {p: p.stat().st_mtime_ns for p in stamps} == stamps  # nothing was rewritten


def test_a_run_from_an_edited_model_is_pending_not_reused(mixed, tmp_path):
    model = tmp_path / "model.py"
    model.write_text(model.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    assert {run.outcome for run in collect(mixed).runs} == {"pending"}


# --------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------


def test_the_table_has_one_row_per_scenario_in_declaration_order(mixed):
    header, rows = table(collect(mixed))
    # Swept parameters in declaration order, then metrics sorted by name.
    assert header == [
        "scenario_id",
        "status",
        "a",
        "leak",
        "crash",
        "draw",
        "total",
        "failed_invariants",
        "error",
    ]
    by_status = {row[1]: row for row in rows}
    assert [row[2] for row in rows] == [1.0, 2.0, 3.0, 4.0]
    assert by_status["failed"][-2] == "conserved(x+y)"
    assert by_status["error"][-1] == "RuntimeError: the model gave up"
    assert by_status["pending"][5:7] == [None, None]
    assert by_status["succeeded"][6] == 1.0


def test_the_csv_is_byte_identical_when_nothing_changed(mixed):
    first = write_table(collect(mixed)).read_bytes()
    second = write_table(collect(mixed)).read_bytes()
    assert first == second
    text = first.decode("utf-8")
    assert "\r" not in text
    for volatile in ("wall_time", "host", "finished_at"):
        assert volatile not in text


def test_the_csv_reads_back(mixed):
    path = write_table(collect(mixed))
    rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))
    assert [r["status"] for r in rows] == ["succeeded", "failed", "error", "pending"]
    assert float(rows[0]["total"]) == 1.0
    assert rows[3]["total"] == ""


def test_runs_left_over_from_an_earlier_grid_are_not_in_the_table(write_campaign):
    everything = load(write_campaign(MIXED))
    _run(everything)
    fewer = load(
        write_campaign(MIXED.replace("      - {a: 4.0, leak: 0.0, crash: false}\n", ""))
    )
    assert len(list((fewer.output / "runs").iterdir())) == 4  # the orphan is still on disk
    _, rows = table(collect(fewer))
    assert [row[2] for row in rows] == [1.0, 2.0, 3.0]


def test_a_metric_named_like_a_column_is_refused(write_campaign):
    model = (
        "from campaign import RunOutput\n\n"
        "def simulate(params, seed):\n"
        "    return RunOutput(metrics={'status': 1.0})\n"
    )
    campaign = load(write_campaign(MIXED.partition("    invariants:")[0], model=model))
    _run(campaign)
    with pytest.raises(ResultsError, match="status"):
        table(collect(campaign))


def test_parquet_round_trips_with_one_type_per_column(write_campaign):
    pq = pytest.importorskip("pyarrow.parquet")
    body = MIXED.replace("{a: 1.0,", "{a: 1,") + "    format: parquet\n"
    campaign = load(write_campaign(body))
    _run(campaign)
    back = pq.read_table(write_table(collect(campaign))).to_pydict()
    assert back["a"] == [1.0, 2.0, 3.0, 4.0]  # an int among floats became a float
    assert back["status"] == ["succeeded", "failed", "error", "succeeded"]
    assert back["total"][2] is None


# --------------------------------------------------------------------------
# the manifest
# --------------------------------------------------------------------------


def test_the_manifest_says_what_produced_the_table(mixed):
    data = json.loads(write_manifest(collect(mixed)).read_text(encoding="utf-8"))
    assert data["config_hash"] == mixed.config_hash
    assert data["seed"] == 3
    assert data["code"]["campaign_version"]
    assert len(data["code"]["model_sha256"]) == 64
    assert data["counts"]["pending"] == 1
    by_id = {run["scenario_id"]: run for run in data["runs"]}
    for task in tasks(mixed):
        assert by_id[task.scenario_id]["run_seed"] == task.run_seed
    wall = [run["wall_time_s"] for run in data["runs"]]
    assert all(w >= 0.0 for w in wall[:3])
    assert wall[3] is None


def test_the_manifest_records_no_git_state_outside_a_repository(mixed):
    assert manifest(collect(mixed))["code"]["git"] is None


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_the_manifest_records_the_commit_and_whether_it_is_dirty(mixed, tmp_path):
    git_exe = shutil.which("git")
    assert git_exe is not None

    def git(*args: str) -> None:
        subprocess.run(
            [git_exe, "-C", str(tmp_path), *args],
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    git("add", "model.py")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "model")
    state = manifest(collect(mixed))["code"]["git"]
    assert len(state["commit"]) == 40
    assert state["uncommitted_changes"] is False

    (tmp_path / "model.py").write_text("# changed\n", encoding="utf-8")
    assert manifest(collect(mixed))["code"]["git"]["uncommitted_changes"] is True
