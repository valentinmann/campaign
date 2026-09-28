"""Fixtures that write real campaign files to disk.

Tests go through the real loader rather than handing dictionaries to the
parser, because some of what they check only exists at load time: YAML reading
``1e-9`` as a string is a property of the loader, not of the schema.
"""

from __future__ import annotations

import functools
import json
import shutil
import subprocess
import sys
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

# A deliberately small model for the library's own tests: two series that sum
# to a constant, a monotone one, and a metric drawn from the run's seed so
# that seed handling is observable. Parameters steer it into failure on demand:
# `leak` breaks conservation, `crash` raises, `nan` poisons a series, and
# `exit` kills the interpreter outright, as the OOM killer or a SLURM time
# limit would.
TOY_MODEL = """\
import os

import numpy as np

from campaign import RunOutput


def simulate(params, seed):
    if params.get("exit"):
        os._exit(int(params["exit"]))
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


# --------------------------------------------------------------------------
# a stand-in for sbatch
# --------------------------------------------------------------------------

FAKE_SBATCH = Path(__file__).with_name("fake_sbatch.py")


@functools.cache
def usable_bash() -> str | None:
    """A bash that actually runs, or None.

    On Windows, ``bash`` on the PATH can be the WSL launcher with no Linux
    installed behind it, so Git's own bash is tried first, and every
    candidate has to answer before it is trusted.
    """
    candidates = []
    if sys.platform == "win32":
        candidates.append(r"C:\Program Files\Git\bin\bash.exe")
    if found := shutil.which("bash"):
        candidates.append(found)
    for candidate in candidates:
        try:
            probe = subprocess.run(
                [candidate, "-c", "echo ok"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if probe.stdout.strip() == "ok":
            return candidate
    return None


@dataclass
class FakeSbatch:
    """The stub's command, and the calls it has seen."""

    command: list[str]
    log: Path

    @property
    def yaml(self) -> str:
        """The campaign file line that points the SLURM backend at the stub."""
        quoted = ", ".join(f"'{Path(part).as_posix()}'" for part in self.command)
        return f"sbatch: [{quoted}]"

    def calls(self) -> list[dict[str, Any]]:
        """Every call the stub received, oldest first."""
        if not self.log.exists():
            return []
        files = sorted(self.log.glob("*.json"))
        return [json.loads(f.read_text(encoding="utf-8")) for f in files]


@pytest.fixture
def fake_sbatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeSbatch:
    log = tmp_path / "sbatch_calls"
    monkeypatch.setenv("FAKE_SBATCH_LOG", str(log))
    monkeypatch.setenv("FAKE_SBATCH_IDS", str(tmp_path / "sbatch_job_ids"))
    monkeypatch.delenv("FAKE_SBATCH_MODE", raising=False)
    if bash := usable_bash():
        monkeypatch.setenv("FAKE_SBATCH_BASH", bash)
    return FakeSbatch(command=[sys.executable, str(FAKE_SBATCH)], log=log)
