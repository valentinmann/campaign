"""Fixtures that write real campaign files to disk.

Tests go through the real loader rather than handing dictionaries to the
parser, because some of what they check only exists at load time: YAML reading
``1e-9`` as a string is a property of the loader, not of the schema.
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable
from pathlib import Path

import pytest

# A deliberately small model for the library's own tests: two series that sum
# to a constant, a monotone one, and a metric drawn from the run's seed so
# that seed handling is observable. Parameters steer it into failure on demand:
# `leak` breaks conservation, `crash` raises, `nan` poisons a series.
TOY_MODEL = """\
import numpy as np

from campaign import RunOutput


def simulate(params, seed):
    if params.get("crash"):
        raise RuntimeError("the model gave up")
    a = float(params["a"])
    t = np.linspace(0.0, 1.0, 11)
    x = a * (1.0 - t)
    y = a * t
    y[5] += float(params.get("leak", 0.0))
    if params.get("nan"):
        x[3] = np.nan
    draw = float(np.random.default_rng(seed).uniform())
    return RunOutput(
        series={"t": t, "x": x, "y": y},
        metrics={"total": float(x[0] + y[0]), "draw": draw},
    )
"""

WriteCampaign = Callable[..., Path]


@pytest.fixture
def write_campaign(tmp_path: Path) -> WriteCampaign:
    """Write ``model.py`` and a campaign file into ``tmp_path``.

    Returns a function taking the YAML body (dedented) and returning the path
    of the campaign file.
    """

    def _write(
        body: str, *, model: str = TOY_MODEL, filename: str = "campaign.yaml"
    ) -> Path:
        (tmp_path / "model.py").write_text(model, encoding="utf-8")
        path = tmp_path / filename
        path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
        return path

    return _write
