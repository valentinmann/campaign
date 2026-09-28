"""The validation report: every failure, with its invariant and its parameters."""

from __future__ import annotations

import pytest

from campaign.config import load
from campaign.report import render, write_report
from campaign.results import collect
from campaign.runner import execute, pending

CAMPAIGN = """
    name: toy
    model: model.py:simulate
    seed: 5
    output: out
    scenarios:
      - {a: 1.0, leak: 0.0, crash: false}
      - {a: 2.0, leak: 1.0e-3, crash: false}
      - {a: 3.0, leak: 0.0, crash: true}
      - {a: 4.0, leak: 0.0, crash: false}
    invariants:
      - {check: conserved, series: [x, y], rtol: 1e-9}
      - {check: nonnegative, series: x}
"""


@pytest.fixture
def report(write_campaign) -> str:
    campaign = load(write_campaign(CAMPAIGN))
    for task in pending(campaign)[:-1]:
        execute(task)
    return render(collect(campaign))


def test_the_counts_are_stated(report):
    for line in (
        "| attempted | 4 |",
        "| succeeded | 1 |",
        "| failed | 1 |",
        "| error | 1 |",
        "| pending | 1 |",
    ):
        assert line in report


def test_every_failure_names_its_invariant_its_evidence_and_its_parameters(report):
    failures = report.split("## Failures")[1].split("## Errors")[0]
    assert "a=2.0, leak=0.001, crash=False" in failures
    assert "**conserved(x+y)**" in failures
    assert "drifts by" in failures
    assert "nonnegative" not in failures  # it held; only what broke is listed


def test_each_invariant_says_how_many_runs_break_it(report):
    assert "| conserved(x+y) | 1 |" in report
    assert "| nonnegative(x) | 0 |" in report


def test_errors_and_pending_runs_are_listed(report):
    errors = report.split("## Errors")[1].split("## Pending")[0]
    assert "a=3.0" in errors
    assert "RuntimeError: the model gave up" in errors
    assert "retried on resume" in errors
    assert "a=4.0" in report.split("## Pending")[1]


def test_a_clean_campaign_says_so(write_campaign):
    clean = CAMPAIGN.replace("leak: 1.0e-3", "leak: 0.0").replace(
        "crash: true", "crash: false"
    )
    campaign = load(write_campaign(clean))
    for task in pending(campaign):
        execute(task)
    text = render(collect(campaign))
    assert "No run failed an invariant." in text
    assert "## Errors" not in text
    assert "## Pending" not in text


def test_text_from_the_campaign_cannot_break_the_markdown(write_campaign):
    """A pipe would split a table cell; a < would open an HTML tag."""
    nasty = CAMPAIGN.replace("name: toy", "name: toy").replace(
        "- {check: nonnegative, series: x}",
        "- {check: nonnegative, series: x, name: 'x | <b>never</b> *below* 0'}",
    )
    nasty = nasty.replace(
        "{a: 2.0, leak: 1.0e-3, crash: false}", "{a: 2.0, leak: 1.0e-3, crash: 'no|<i>'}"
    )
    campaign = load(write_campaign(nasty))
    for task in pending(campaign):
        execute(task)
    text = render(collect(campaign))
    assert "<b>" not in text
    assert "<i>" not in text
    assert r"x \| \<b\>never\</b\> \*below\* 0" in text
    assert r"crash=no\|\<i\>" in text


def test_the_report_is_written_next_to_the_table(write_campaign):
    campaign = load(write_campaign(CAMPAIGN))
    path = write_report(collect(campaign))
    assert path == campaign.output / "report.md"
    assert path.read_text(encoding="utf-8").startswith("# Validation report: toy")


def test_snake_case_names_are_left_readable(write_campaign):
    """An underscore inside a word never opens emphasis, so it stays as written."""
    body = """
        name: toy
        model: model.py:simulate
        seed: 1
        output: out
        grid:
          a: [1.0]
          leak_rate: [0.5]
    """
    text = render(collect(load(write_campaign(body))))  # nothing ran: listed as pending
    assert "leak_rate=0.5" in text
    assert r"leak\_rate" not in text
