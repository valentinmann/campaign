"""Running one scenario, storing it, and knowing afterwards whether it finished.

Resume is only as good as the answer to "did this run finish?", so most of
these tests break a run in some specific way, a crash, a truncated file, a
stray temporary, an edited model, and check the answer stays honest.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from campaign import runner
from campaign.config import load
from campaign.model import ModelError, load_model
from campaign.runner import derive_seed, execute, pending, tasks
from campaign.store import SERIES, read_record, read_series, write_atomic

CAMPAIGN = """
    name: toy
    model: model.py:simulate
    seed: 11
    output: out
    grid:
      a: [1.0, 2.0, 3.0]
"""


@pytest.fixture
def campaign(write_campaign):
    return load(write_campaign(CAMPAIGN))


def _run_all(campaign) -> None:
    for task in tasks(campaign):
        execute(task)


# --------------------------------------------------------------------------
# one run
# --------------------------------------------------------------------------


def test_a_completed_run_stores_its_series_and_metrics(campaign):
    task = tasks(campaign)[1]
    record = execute(task)
    assert record.status == "completed"
    assert record.metrics["total"] == 2.0
    assert record.params == {"a": 2.0}
    assert record.wall_time_s >= 0.0

    stored = read_series(Path(task.run_dir))
    assert stored is not None
    np.testing.assert_allclose(stored["x"] + stored["y"], 2.0)
    assert read_record(Path(task.run_dir)) == record


def test_a_model_that_raises_is_recorded_and_the_error_kept(write_campaign):
    c = load(write_campaign(CAMPAIGN + "    constants: {crash: true}\n"))
    record = execute(tasks(c)[0])
    assert record.status == "error"
    assert record.error == "RuntimeError: the model gave up"
    assert record.traceback is not None
    assert "simulate" in record.traceback
    assert not (Path(tasks(c)[0].run_dir) / SERIES).exists()


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("return {'x': [1.0]}", "returned dict, expected campaign.RunOutput"),
        ("return RunOutput(series={'x': ['a', 'b']})", "series 'x' is not numeric"),
        ("return RunOutput(metrics={'m': 'high'})", "metric 'm' is not a number"),
    ],
)
def test_a_model_returning_something_unusable_is_an_error(write_campaign, body, message):
    model = f"from campaign import RunOutput\n\ndef simulate(params, seed):\n    {body}\n"
    c = load(write_campaign(CAMPAIGN, model=model))
    record = execute(tasks(c)[0])
    assert record.status == "error"
    assert message in (record.error or "")


def test_series_named_like_numpys_own_arguments_survive_storage(write_campaign):
    """np.savez takes arrays as keywords next to its own `file` and `allow_pickle`."""
    model = (
        "import numpy as np\nfrom campaign import RunOutput\n\n"
        "def simulate(params, seed):\n"
        "    return RunOutput(series={'file': np.arange(3.0), 'allow_pickle': np.ones(2),\n"
        "                             'names': np.zeros(4)})\n"
    )
    c = load(write_campaign(CAMPAIGN, model=model))
    task = tasks(c)[0]
    assert execute(task).status == "completed"
    stored = read_series(Path(task.run_dir))
    assert stored is not None
    assert set(stored) == {"file", "allow_pickle", "names"}
    np.testing.assert_array_equal(stored["file"], [0.0, 1.0, 2.0])


def test_a_model_can_import_a_module_next_to_it(write_campaign, tmp_path):
    (tmp_path / "helper_for_campaign_test.py").write_text(
        "SCALE = 10.0\n", encoding="utf-8"
    )
    model = (
        "import numpy as np\nfrom campaign import RunOutput\n"
        "from helper_for_campaign_test import SCALE\n\n"
        "def simulate(params, seed):\n"
        "    return RunOutput(metrics={'scaled': SCALE * params['a']})\n"
    )
    c = load(write_campaign(CAMPAIGN, model=model))
    assert execute(tasks(c)[0]).metrics == {"scaled": 10.0}


def test_loading_a_broken_model_raises_a_model_error_and_leaves_no_trace(tmp_path):
    broken = tmp_path / "broken.py"
    broken.write_text("raise ImportError('missing dependency')\n", encoding="utf-8")
    before = set(sys.modules)
    with pytest.raises(ModelError, match="missing dependency"):
        load_model(broken, "simulate")
    assert set(sys.modules) == before

    fine = tmp_path / "fine.py"
    fine.write_text("def other(p, s): ...\n", encoding="utf-8")
    with pytest.raises(ModelError, match="no function named 'simulate'"):
        load_model(fine, "simulate")


# --------------------------------------------------------------------------
# seeds
# --------------------------------------------------------------------------


def test_a_scenario_keeps_its_seed_whatever_the_order_or_the_neighbours(write_campaign):
    """Seeds follow identity, not position: reorder and extend, nothing moves."""
    first = {t.scenario_id: t.run_seed for t in tasks(load(write_campaign(CAMPAIGN)))}
    reordered = CAMPAIGN.replace("[1.0, 2.0, 3.0]", "[3.0, 0.5, 1.0, 2.0]")
    second = {t.scenario_id: t.run_seed for t in tasks(load(write_campaign(reordered)))}
    assert first.items() <= second.items()


def test_seeds_differ_between_scenarios_and_between_campaign_seeds(campaign):
    seeds = [t.run_seed for t in tasks(campaign)]
    assert len(set(seeds)) == len(seeds)
    sid = tasks(campaign)[0].scenario_id
    assert derive_seed(11, sid) != derive_seed(12, sid)


def test_the_seed_reaches_the_model_and_fits_the_legacy_numpy_interface(campaign):
    task = tasks(campaign)[0]
    record = execute(task)
    expected = float(np.random.default_rng(task.run_seed).uniform())
    assert record.metrics["draw"] == expected
    np.random.RandomState(task.run_seed)  # raises above 2**32 - 1


# --------------------------------------------------------------------------
# what is left to do
# --------------------------------------------------------------------------


def test_nothing_is_pending_after_every_run_completed(campaign):
    assert len(pending(campaign)) == 3
    _run_all(campaign)
    assert pending(campaign) == []


def test_a_run_that_raised_is_pending_again(write_campaign):
    body = CAMPAIGN.replace(
        "a: [1.0, 2.0, 3.0]", "a: [1.0, 2.0]\n      crash: [false, true]"
    )
    c = load(write_campaign(body))
    _run_all(c)
    left = pending(c)
    assert [t.params["crash"] for t in left] == [True, True]


def test_editing_the_model_makes_every_run_pending(tmp_path, campaign):
    _run_all(campaign)
    model = tmp_path / "model.py"
    model.write_text(
        model.read_text(encoding="utf-8") + "\n# fixed a bug\n", encoding="utf-8"
    )
    assert len(pending(campaign)) == 3


def test_changing_the_campaign_seed_makes_every_run_pending(write_campaign, campaign):
    _run_all(campaign)
    reseeded = load(write_campaign(CAMPAIGN.replace("seed: 11", "seed: 12")))
    assert len(pending(reseeded)) == 3


@pytest.mark.parametrize("damage", ["truncate", "garbage", "wrong_schema"])
def test_a_damaged_record_counts_as_unfinished(campaign, damage):
    _run_all(campaign)
    record_path = Path(tasks(campaign)[0].run_dir) / "record.json"
    text = record_path.read_text(encoding="utf-8")
    record_path.write_text(
        {
            "truncate": text[: len(text) // 2],
            "garbage": "\x00\x01not json",
            "wrong_schema": json.dumps({"scenario_id": "x"}),
        }[damage],
        encoding="utf-8",
    )
    assert [t.scenario_id for t in pending(campaign)] == [tasks(campaign)[0].scenario_id]


def test_a_truncated_series_file_reads_as_missing(campaign):
    task = tasks(campaign)[0]
    execute(task)
    path = Path(task.run_dir) / SERIES
    path.write_bytes(path.read_bytes()[:50])
    assert read_series(Path(task.run_dir)) is None


@pytest.mark.parametrize("damage", ["truncate", "delete"])
def test_a_sound_record_with_unreadable_series_is_redone(campaign, damage):
    """The record alone said "finished"; the series it vouches for were gone.

    An earlier version decided a run was finished from its record only, so
    this run was skipped on resume and could never be re-checked.
    """
    _run_all(campaign)
    task = tasks(campaign)[2]
    path = Path(task.run_dir) / SERIES
    if damage == "truncate":
        path.write_bytes(path.read_bytes()[:50])
    else:
        path.unlink()
    assert read_record(Path(task.run_dir)) is not None
    assert [t.scenario_id for t in pending(campaign)] == [task.scenario_id]


def test_an_interrupted_rerun_does_not_leave_the_old_record_in_place(campaign, monkeypatch):
    """Ctrl-C during a rerun: the scenario must look unfinished, not done."""
    _run_all(campaign)
    task = tasks(campaign)[0]
    # A crash between deleting the old record and writing the new one.
    monkeypatch.setattr(
        runner, "load_model", lambda *_: (_ for _ in ()).throw(KeyboardInterrupt)
    )
    with pytest.raises(KeyboardInterrupt):
        execute(task)
    assert read_record(Path(task.run_dir)) is None
    assert task.scenario_id in {t.scenario_id for t in pending(campaign)}


# --------------------------------------------------------------------------
# atomic writes
# --------------------------------------------------------------------------


def test_an_atomic_write_replaces_the_file_and_leaves_nothing_behind(tmp_path):
    target = tmp_path / "deep" / "record.json"
    write_atomic(target, b"first")
    write_atomic(target, b"second")
    assert target.read_bytes() == b"second"
    assert [p.name for p in target.parent.iterdir()] == ["record.json"]


def test_a_failed_atomic_write_keeps_the_old_file(tmp_path, monkeypatch):
    target = tmp_path / "record.json"
    write_atomic(target, b"old")

    def refuse(*_: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", refuse)
    with pytest.raises(OSError, match="disk full"):
        write_atomic(target, b"new")
    assert target.read_bytes() == b"old"
    assert [p.name for p in tmp_path.iterdir()] == ["record.json"]
