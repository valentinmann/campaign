"""The SIR demo: checked against closed forms, and its invariants against a real bug.

The invariants in ``demo/sir.yaml`` allow the integrator's absolute tolerance,
one millionth of a person, because the first run failed them on noise of that
size late in the epidemic. A tolerance loosened until the runs pass would be
worthless, so the last test here hands the same invariants a model with a
genuine bug and checks they still refuse it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from campaign.api import write_outputs
from campaign.config import load
from campaign.invariants import check
from campaign.results import collect
from campaign.runner import execute, pending
from demo.plot import draw, read
from demo.sir import ATOL, peak_closed_form, simulate

DEMO = Path(__file__).resolve().parents[1] / "demo"
BASE = {
    "gamma": 0.1,
    "reduction": 0.6,
    "population": 1.0e6,
    "initial_infected": 10.0,
    "t_end": 365.0,
}


def _run(beta: float, t_intervention: float, **overrides: float):
    return simulate(
        {**BASE, **overrides, "beta": beta, "t_intervention": t_intervention}, 0
    )


def _closed_form(beta: float) -> float:
    n, i0 = BASE["population"], BASE["initial_infected"]
    return peak_closed_form(beta, BASE["gamma"], n, n - i0, i0)


# --------------------------------------------------------------------------
# the peak, against what it must be
# --------------------------------------------------------------------------


@pytest.mark.parametrize("beta", [0.15, 0.3, 0.45, 0.6])
def test_without_intervention_the_peak_matches_the_closed_form(beta):
    out = _run(beta, t_intervention=1.0e9)
    assert out.metrics["peak_infected"] == pytest.approx(_closed_form(beta), rel=1e-8)


def test_an_intervention_after_the_peak_changes_nothing_about_the_peak():
    free = _run(0.45, t_intervention=1.0e9)
    late = _run(0.45, t_intervention=free.metrics["peak_day"] + 5.0)
    assert late.metrics["peak_infected"] == pytest.approx(
        free.metrics["peak_infected"], rel=1e-9
    )
    assert late.metrics["peak_day"] == pytest.approx(free.metrics["peak_day"], abs=1e-6)


def test_an_intervention_that_stops_growth_at_once_puts_the_peak_on_that_day():
    """dI/dt jumps from positive to negative there, and no event fires.

    A peak read off the solver's events alone would miss this and report an
    earlier, lower value.
    """
    # On day 60, with beta = 0.3, S/N is about 0.38: infection still grows
    # (0.3 x 0.38 = 0.114 > gamma = 0.1), and cutting 60% of contacts takes
    # that to 0.045 in one step. The premise is asserted, not assumed: the
    # first version of this test used day 40, where the epidemic has barely
    # started and growth survives the cut, and this line is what said so.
    out = _run(0.3, t_intervention=60.0)
    t, infected = out.series["t"], out.series["I"]
    s_at_cut = np.interp(60.0, t, out.series["S"]) / BASE["population"]
    assert s_at_cut * 0.3 * 0.4 < BASE["gamma"] < s_at_cut * 0.3  # the premise
    assert out.metrics["peak_day"] == 60.0
    assert out.metrics["peak_infected"] == pytest.approx(np.interp(60.0, t, infected))


def test_after_an_early_cut_the_peak_matches_the_closed_form_from_that_day():
    """Growth continues more slowly; SIR's conserved quantity still fixes the peak."""
    beta, cut, reduction = 0.6, 10.0, 0.6
    out = _run(beta, t_intervention=cut)
    t, s, infected = out.series["t"], out.series["S"], out.series["I"]
    k = int(np.flatnonzero(t == cut)[0])
    expected = peak_closed_form(
        beta * (1 - reduction), BASE["gamma"], BASE["population"], s[k], infected[k]
    )
    assert out.metrics["peak_infected"] == pytest.approx(expected, rel=1e-7)
    assert out.metrics["peak_day"] > cut


@pytest.mark.parametrize("beta", [0.15, 0.3, 0.45, 0.6])
@pytest.mark.parametrize("cut", [5.0, 10.0, 30.0, 90.0, 1.0e9])
def test_the_reported_peak_is_never_below_any_sample(beta, cut):
    """The peak is a maximum, so no sample of I may exceed it.

    An earlier version took the maximum over the start, the dI/dt = 0 events
    and the intervention day, and forgot the end of the horizon. An epidemic
    slowed but still growing on day 365 then reported its size on the day of
    the intervention as its peak: 27 people, where it had reached 13,751 by
    the last day. The series were right, so no invariant could see it; the
    first table of results did.
    """
    out = _run(beta, t_intervention=cut)
    assert out.metrics["peak_infected"] >= out.series["I"].max() * (1 - 1e-12)


def test_an_epidemic_still_growing_at_the_horizon_says_so():
    out = _run(0.3, t_intervention=5.0)  # R0 = 3, then 1.2: slow growth all year
    assert out.metrics["peak_day"] == BASE["t_end"]
    assert out.metrics["peaked"] == 0.0
    assert _run(0.3, t_intervention=10.0).metrics["peaked"] == 1.0


def test_the_intervention_day_is_sampled_once_and_time_only_moves_forward():
    t = _run(0.4, t_intervention=20.0).series["t"]
    assert np.count_nonzero(t == 20.0) == 1
    assert np.all(np.diff(t) > 0)


def test_the_model_ignores_its_seed():
    a = simulate({**BASE, "beta": 0.4, "t_intervention": 20.0}, 1)
    b = simulate({**BASE, "beta": 0.4, "t_intervention": 20.0}, 99)
    np.testing.assert_array_equal(a.series["I"], b.series["I"])


# --------------------------------------------------------------------------
# the campaign itself
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def demo_campaign(tmp_path_factory):
    """The shipped demo, copied so its output lands in a temporary directory."""
    directory = tmp_path_factory.mktemp("demo")
    shutil.copy(DEMO / "sir.py", directory / "sir.py")
    text = (DEMO / "sir.yaml").read_text(encoding="utf-8")
    (directory / "sir.yaml").write_text(
        text.replace("output: ../results/sir", "output: out"), encoding="utf-8"
    )
    return load(directory / "sir.yaml")


@pytest.fixture(scope="module")
def demo_ran(demo_campaign):
    """The demo with all 80 runs done, once for the whole module."""
    for task in pending(demo_campaign):
        execute(task)
    return demo_campaign


def test_the_demo_sweeps_at_least_fifty_scenarios(demo_campaign):
    assert len(demo_campaign.scenarios) == 80
    assert demo_campaign.swept == ("beta", "t_intervention")


def test_every_demo_run_satisfies_every_invariant(demo_ran):
    result = collect(demo_ran)
    assert result.counts["succeeded"] == 80, [
        (run.scenario.params, run.violations) for run in result.runs if run.violations
    ]


def test_the_invariants_allow_the_integrators_precision_and_no_more(demo_campaign):
    """If someone loosens the campaign file, this fails."""
    tolerances = {inv.name: inv.atol for inv in demo_campaign.invariants if inv.atol}
    assert tolerances == {
        "nonnegative(S, I, R)": ATOL,
        "monotone decreasing(S)": ATOL,
        "monotone increasing(R)": ATOL,
    }


def _one_person_lost(series):
    series["R"][len(series["R"]) // 2 :] -= 1.0


def _one_person_below_zero(series):
    series["I"][-1] = -1.0


def _one_recovery_undone(series):
    # At the last sample R grows by 0.06 persons per sample, so dropping it by
    # one person makes it fall. Halfway through it grows by about 150 per
    # sample, the same drop is hidden, and only conservation would see it.
    series["R"][-1] -= 1.0


@pytest.mark.parametrize(
    ("bug", "caught_by"),
    [
        (_one_person_lost, "conserved(S+I+R)"),
        (_one_person_below_zero, "nonnegative(S, I, R)"),
        (_one_recovery_undone, "monotone increasing(R)"),
    ],
)
def test_a_one_person_bug_is_still_caught(demo_campaign, bug, caught_by):
    """A single person is a million times the tolerance the checks allow."""
    series = {name: values.copy() for name, values in _run(0.4, 20.0).series.items()}
    assert check(demo_campaign.invariants, series) == []
    bug(series)
    assert caught_by in [v.invariant for v in check(demo_campaign.invariants, series)]


def test_the_figure_is_drawn_from_the_results_byte_for_byte_reproducibly(
    demo_ran, tmp_path
):
    write_outputs(demo_ran)
    grid = read(demo_ran.output)
    assert grid.peak_pct.shape == (10, 8)
    # R0 = 3 cut on day 5 is still growing on day 365: a lower bound, labelled so.
    assert grid.still_rising.sum() == 1

    first = draw(grid, tmp_path / "a.png").read_bytes()
    second = draw(grid, tmp_path / "b.png").read_bytes()
    assert first == second
    assert b"tEXtSoftware\x00matplotlib" in first
    for volatile in (b"Creation Time", b"matplotlib v", b"Matplotlib version"):
        assert volatile not in first
