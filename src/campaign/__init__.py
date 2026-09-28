"""Reproducible simulation campaigns.

A campaign is a model, a set of scenarios and a list of invariants every run
must satisfy. This package expands the scenarios, runs them locally or as a
SLURM job array, checks every result against the invariants, and writes one
table, one manifest and one validation report. Re-running a campaign skips the
simulations that already completed and re-checks all of them.
"""

__version__ = "0.1.0"
