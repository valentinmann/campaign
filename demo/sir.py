"""An SIR epidemic with one intervention, written as a campaign model.

    dS/dt = -b(t) S I / N
    dI/dt =  b(t) S I / N - gamma I
    dR/dt =  gamma I

where the transmission rate b(t) is ``beta`` until day ``t_intervention`` and
``beta * (1 - reduction)`` from then on: contacts are cut on a given day. The
campaign sweeps ``beta`` and ``t_intervention``.

Three numerical points, each pinned by a test in ``tests/test_demo.py``:

1. The right-hand side jumps at ``t_intervention``. An adaptive integrator
   stepping across a discontinuity spends its error budget there without
   knowing why, so the integration is split into two segments that meet
   exactly at the intervention.

2. The peak is located where dI/dt = 0, that is where b S/N = gamma, by an
   event function on the solver, not by taking the largest sample. With an
   intervention there is a second way to peak: if cutting contacts takes b S/N
   below gamma in one step, dI/dt jumps from positive to negative at
   ``t_intervention`` without passing through zero, no event fires, and the
   peak is the intervention itself. And an epidemic slowed but not stopped may
   still be growing on the last day, so the end of the horizon is a
   candidate as well, and the run says whether it actually peaked. All three
   are candidates; forgetting the last one is a bug this file once had.

3. Without an intervention, or with one that comes after the peak, the peak
   has a closed form. SIR conserves I + S - (N/R0) ln S, which at the peak,
   where S = N/R0, gives

       I_max = I0 + S0 - N/R0 - (N/R0) ln(S0 R0 / N),     R0 = beta / gamma.

   The model is checked against it.

The model is deterministic. It accepts the run's seed, as every campaign
model must, and does nothing with it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from campaign import RunOutput

# The population is in persons. ATOL was 1e-6 at first, and 11 of the 80
# demo runs failed non-negativity and monotonicity. Once infection decayed
# below 1e-6 the solver stopped controlling it and took steps weeks long, and
# the samples, read off the interpolant between those steps, wobbled by a few
# units in the last place of S and R and by 2e-8 persons in I. At the solver's
# own steps every series was monotone and positive. At 1e-12 the tail is
# resolved, every sample is too, and a run costs about half as much again.
RTOL = 1e-10
ATOL = 1e-12
SAMPLES_PER_DAY = 4


def peak_closed_form(
    beta: float, gamma: float, population: float, s0: float, i0: float
) -> float:
    """The SIR peak of infection, without intervention.

    >>> round(peak_closed_form(0.3, 0.1, 1.0, 0.999, 0.001), 6)
    0.300796
    """
    n_over_r0 = population * gamma / beta
    if s0 <= n_over_r0:
        return i0  # already past the peak: infection only falls
    return float(i0 + s0 - n_over_r0 - n_over_r0 * np.log(s0 / n_over_r0))


def _segment(
    rate: float,
    gamma: float,
    population: float,
    y0: NDArray[np.float64],
    span: tuple[float, float],
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[tuple[float, float]]]:
    """Integrate one segment of constant transmission rate over ``span``.

    Returns the sample times, the states there, and every (time, infected)
    point at which dI/dt crossed zero from above.
    """

    def rhs(_t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        s, i, _r = y
        infection = rate * s * i / population
        recovery = gamma * i
        return np.array([-infection, infection - recovery, recovery])

    def growth_stops(_t: float, y: NDArray[np.float64]) -> float:
        return float(rate * y[0] / population - gamma)

    growth_stops.direction = -1  # type: ignore[attr-defined]

    t0, t1 = span
    n = max(2, round((t1 - t0) * SAMPLES_PER_DAY) + 1)
    sol = solve_ivp(
        rhs,
        (t0, t1),
        y0,
        method="DOP853",
        t_eval=np.linspace(t0, t1, n),
        events=growth_stops,
        rtol=RTOL,
        atol=ATOL,
    )
    if not sol.success:
        raise RuntimeError(f"integration failed on [{t0}, {t1}]: {sol.message}")
    peaks = [
        (float(t), float(y[1]))
        for t, y in zip(sol.t_events[0], sol.y_events[0], strict=True)
    ]
    return sol.t, sol.y, peaks


def simulate(params: dict[str, Any], _seed: int) -> RunOutput:
    """Run one scenario. The seed is accepted and unused: the model is deterministic."""
    beta = float(params["beta"])
    gamma = float(params["gamma"])
    reduction = float(params["reduction"])
    population = float(params["population"])
    i0 = float(params["initial_infected"])
    t_end = float(params["t_end"])
    t_cut = min(float(params["t_intervention"]), t_end)

    y0 = np.array([population - i0, i0, 0.0])
    t_a, y_a, peaks = _segment(beta, gamma, population, y0, (0.0, t_cut))
    candidates = [(0.0, i0), *peaks, (t_cut, float(y_a[1, -1]))]
    t, y = t_a, y_a

    if t_cut < t_end:
        t_b, y_b, peaks_b = _segment(
            beta * (1.0 - reduction), gamma, population, y_a[:, -1], (t_cut, t_end)
        )
        candidates += peaks_b
        # The two segments share the intervention instant; keep it once.
        t = np.concatenate([t_a, t_b[1:]])
        y = np.concatenate([y_a, y_b[:, 1:]], axis=1)

    # The end of the horizon is a candidate too: an epidemic slowed but not
    # stopped can still be growing on the last day.
    candidates.append((t_end, float(y[1, -1])))
    peak_day, peak = max(candidates, key=lambda c: c[1])
    s, i, r = y
    return RunOutput(
        series={"t": t, "S": s, "I": i, "R": r},
        metrics={
            "peak_infected": peak,
            "peak_fraction": peak / population,
            "peak_day": peak_day,
            # 0 when infection is still rising at t_end: the "peak" is then
            # only the largest value within the horizon, a lower bound.
            "peaked": float(peak_day < t_end),
            "attack_rate": float(r[-1] / population),
        },
    )
