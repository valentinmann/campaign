"""The invariant vocabulary: what it accepts, and what it refuses to let pass.

The NaN tests are the ones that matter most. Every comparison with NaN is
False, so a check written as "no value is below zero" passes an array full of
NaN, while one written as "every value is at or above zero" fails it. The two
read the same; these tests pin which one each check actually is.
"""

from __future__ import annotations

import numpy as np
import pytest

from campaign.invariants import (
    CHECKS,
    Invariant,
    InvariantError,
    check,
    parse_invariant,
)


def _inv(**spec: object) -> Invariant:
    return parse_invariant(spec, "test")


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


def test_every_check_parses_with_a_derived_name():
    names = {
        _inv(check="finite", series=["a", "b"]).name,
        _inv(check="nonnegative", series="a").name,
        _inv(check="conserved", series=["a", "b"], rtol=1e-9).name,
        _inv(check="monotone", series="a", direction="decreasing").name,
    }
    assert names == {
        "finite(a, b)",
        "nonnegative(a)",
        "conserved(a+b)",
        "monotone decreasing(a)",
    }


def test_an_explicit_name_is_kept():
    assert (
        _inv(check="finite", series="a", name="no NaN anywhere").name == "no NaN anywhere"
    )


def test_unknown_check_lists_the_valid_ones():
    with pytest.raises(InvariantError, match="must be one of") as info:
        _inv(check="positive", series="a")
    for name in CHECKS:
        assert name in str(info.value)


def test_a_misspelt_tolerance_is_rejected_rather_than_ignored():
    """`rtoll` falling back to the default would make the check silently exact."""
    with pytest.raises(InvariantError, match="rtoll"):
        _inv(check="conserved", series=["a"], rtoll=1e-9)


def test_a_key_that_belongs_to_another_check_is_rejected():
    with pytest.raises(InvariantError, match="finite does not take rtol"):
        _inv(check="finite", series="a", rtol=1e-3)


def test_conserved_demands_a_tolerance():
    with pytest.raises(InvariantError, match="never exactly constant"):
        _inv(check="conserved", series=["a", "b"])


@pytest.mark.parametrize("bad", [-1e-9, float("nan"), float("inf"), "1e-9", True])
def test_tolerances_must_be_finite_nonnegative_numbers(bad):
    with pytest.raises(InvariantError, match="atol"):
        _inv(check="nonnegative", series="a", atol=bad)


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"check": "monotone", "series": "a"}, "direction"),
        ({"check": "monotone", "series": "a", "direction": "up"}, "direction"),
        (
            {"check": "monotone", "series": ["a", "b"], "direction": "increasing"},
            "exactly one",
        ),
        ({"check": "finite", "series": []}, "non-empty"),
        ({"check": "finite", "series": ["a", "a"]}, "twice"),
        ({"check": "finite", "series": ["a", 3]}, "non-empty string"),
        ({"check": "finite"}, "series"),
        ({"check": "finite", "series": "a", "name": ""}, "name"),
    ],
)
def test_malformed_declarations_are_rejected(spec, message):
    with pytest.raises(InvariantError, match=message):
        parse_invariant(spec, "test")


def test_a_non_mapping_is_rejected():
    with pytest.raises(InvariantError, match="expected a mapping"):
        parse_invariant(["finite", "a"], "test")


# --------------------------------------------------------------------------
# NaN cannot slip through
# --------------------------------------------------------------------------

_EVERY_CHECK = [
    {"check": "finite", "series": "x"},
    {"check": "nonnegative", "series": "x"},
    {"check": "conserved", "series": ["x"], "rtol": 1.0},
    {"check": "monotone", "series": "x", "direction": "increasing"},
    {"check": "monotone", "series": "x", "direction": "decreasing"},
]


@pytest.mark.parametrize("spec", _EVERY_CHECK, ids=lambda s: str(s["check"]))
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_no_check_passes_a_non_finite_value(spec, bad):
    """Including the lenient ones: rtol=1 would otherwise accept anything."""
    series = {"x": np.array([1.0, 1.0, bad, 1.0])}
    (violation,) = check([parse_invariant(spec, "test")], series)
    assert "non-finite" in violation.detail
    assert "index 2" in violation.detail


def test_an_all_nan_series_fails_nonnegative():
    """The array that `not any(x < 0)` would wave through."""
    assert check([_inv(check="nonnegative", series="x")], {"x": np.full(5, np.nan)})


# --------------------------------------------------------------------------
# each check against a known answer
# --------------------------------------------------------------------------


def test_nonnegative_reports_the_worst_value_and_where():
    (v,) = check(
        [_inv(check="nonnegative", series="x")], {"x": np.array([0.0, -1e-3, -2e-3, 1.0])}
    )
    assert "-0.002" in v.detail
    assert "index 2" in v.detail


def test_nonnegative_tolerance_admits_round_off_only():
    inv = _inv(check="nonnegative", series="x", atol=1e-12)
    assert check([inv], {"x": np.array([1.0, -5e-13])}) == []
    assert check([inv], {"x": np.array([1.0, -5e-12])})


def test_conserved_holds_within_tolerance_and_fails_outside_it():
    t = np.linspace(0.0, 1.0, 101)
    a, b = 1.0 - t, t
    inv = _inv(check="conserved", series=["a", "b"], rtol=1e-12)
    assert check([inv], {"a": a, "b": b}) == []

    b_leaky = b.copy()
    b_leaky[60] += 1e-6
    (v,) = check([inv], {"a": a, "b": b_leaky})
    assert "index 60" in v.detail
    assert "a+b" in v.detail


def test_conserved_rejects_series_of_different_lengths():
    inv = _inv(check="conserved", series=["a", "b"], atol=1.0)
    (v,) = check([inv], {"a": np.ones(3), "b": np.ones(4)})
    assert "different lengths" in v.detail


@pytest.mark.parametrize(
    ("direction", "values", "holds"),
    [
        ("increasing", [0.0, 1.0, 1.0, 2.0], True),  # flat steps are allowed
        ("increasing", [0.0, 1.0, 0.5, 2.0], False),
        ("decreasing", [3.0, 2.0, 2.0, 0.0], True),
        ("decreasing", [3.0, 2.0, 2.5, 0.0], False),
        ("increasing", [4.0], True),  # a single point is trivially monotone
    ],
)
def test_monotone(direction, values, holds):
    inv = _inv(check="monotone", series="x", direction=direction)
    assert (check([inv], {"x": np.array(values)}) == []) is holds


def test_monotone_names_the_offending_step():
    inv = _inv(check="monotone", series="x", direction="increasing")
    (v,) = check([inv], {"x": np.array([0.0, 1.0, 0.25, 2.0])})
    assert "between index 1 and 2" in v.detail


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def test_every_failed_invariant_is_reported_not_just_the_first():
    invariants = [
        _inv(check="nonnegative", series="x"),
        _inv(check="finite", series="y"),
        _inv(check="monotone", series="x", direction="increasing"),
    ]
    got = check(invariants, {"x": np.array([1.0, -1.0]), "y": np.array([np.nan])})
    assert [v.invariant for v in got] == [
        "nonnegative(x)",
        "finite(y)",
        "monotone increasing(x)",
    ]


def test_a_missing_series_is_a_violation_not_a_crash():
    """A typo in the campaign file must show up in the report, not abort the run."""
    (v,) = check([_inv(check="finite", series="Infected")], {"I": np.ones(3)})
    assert "Infected" in v.detail
    assert "have: I" in v.detail


@pytest.mark.parametrize("bad", [np.ones((2, 2)), np.array([]), np.float64(1.0)])
def test_series_must_be_one_dimensional_and_non_empty(bad):
    (v,) = check([_inv(check="finite", series="x")], {"x": bad})
    assert "one-dimensional" in v.detail
