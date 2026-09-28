"""The validation report: what ran, what held, what did not, and why.

One Markdown file, written next to the results table. Every failure is
listed with the invariant it broke, the evidence, and the parameters that
produced it, so the report can be read without opening anything else.

Everything that comes from the campaign file or the model, parameter values,
invariant names, error messages, is escaped before it goes in. A ``|`` in a
parameter would otherwise split a table cell, and a ``<`` would open a tag.
"""

from __future__ import annotations

import re
from pathlib import Path

from campaign.results import CampaignResult, RunResult
from campaign.store import write_atomic

__all__ = ["render", "write_report"]

# Only what is active in the middle of a line: text from the campaign never
# starts one, so heading and list markers are harmless, and with brackets
# escaped a parenthesis cannot complete a link. Escaping more than this would
# keep the rendered page correct and make the raw file unreadable.
_SPECIAL = re.compile(r"([\\`*_\[\]<>|~])")


def _escape(text: object) -> str:
    r"""Escape Markdown and HTML punctuation, and fold the text onto one line.

    >>> _escape("a|b*c")
    'a\\|b\\*c'
    """
    one_line = " ".join(str(text).split())
    return _SPECIAL.sub(r"\\\1", one_line)


def _label(run: RunResult, swept: tuple[str, ...]) -> str:
    return ", ".join(
        f"{_escape(name)}={_escape(run.scenario.params[name])}" for name in swept
    )


def _first_line(text: str | None) -> str:
    return (text or "").strip().splitlines()[0] if (text or "").strip() else ""


def render(result: CampaignResult) -> str:
    """The report as Markdown text."""
    campaign = result.campaign
    counts = result.counts
    swept = campaign.swept
    failed = [run for run in result.runs if run.outcome == "failed"]
    errors = [run for run in result.runs if run.outcome == "error"]
    waiting = [run for run in result.runs if run.outcome == "pending"]

    lines = [
        f"# Validation report: {_escape(campaign.name)}",
        "",
        (
            f"Model `{_escape(campaign.model_ref)}` "
            f"(sha256 `{result.model_hash[:12]}`), campaign seed {campaign.seed}, "
            f"config hash `{campaign.config_hash[:12]}`."
        ),
        "",
        "| runs | count |",
        "|---|---:|",
        *(f"| {name} | {value} |" for name, value in counts.items()),
        "",
        "## Invariants",
        "",
    ]

    if campaign.invariants:
        lines += ["| invariant | runs failing it |", "|---|---:|"]
        for inv in campaign.invariants:
            n = sum(any(v.invariant == inv.name for v in run.violations) for run in failed)
            lines.append(f"| {_escape(inv.name)} | {n} |")
    else:
        lines.append("None declared, so every run that completed counts as succeeded.")
    lines.append("")

    lines += ["## Failures", ""]
    if not failed:
        lines += ["No run failed an invariant.", ""]
    for run in failed:
        lines += [f"### `{run.scenario.id}` {_label(run, swept)}", ""]
        lines += [
            f"- **{_escape(v.invariant)}**: {_escape(v.detail)}" for v in run.violations
        ]
        lines.append("")

    if errors:
        lines += [
            "## Errors",
            "",
            "The model raised; these runs are retried on resume.",
            "",
        ]
        for run in errors:
            message = _first_line(run.record.error if run.record else None)
            lines.append(f"- `{run.scenario.id}` {_label(run, swept)}: {_escape(message)}")
        lines.append("")

    if waiting:
        lines += [
            "## Pending",
            "",
            (
                f"{len(waiting)} scenario(s) have no usable run yet. Run the campaign "
                "again, or wait for submitted jobs to finish."
            ),
            "",
        ]
        lines += [f"- `{run.scenario.id}` {_label(run, swept)}" for run in waiting]
        lines.append("")

    return "\n".join(lines)


def write_report(result: CampaignResult) -> Path:
    """Write ``report.md`` next to the results. Returns its path."""
    path = result.campaign.output / "report.md"
    write_atomic(path, render(result).encode("utf-8"))
    return path
