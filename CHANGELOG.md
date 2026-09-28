# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-09-28

First release.

### Added

- `config`: campaign files in YAML with a closed schema, a parameter grid or
  an explicit scenario list, and scenario identifiers derived from content
  rather than position. Exponent notation such as `1e-9` is read as a number,
  which PyYAML's YAML 1.1 does not do.
- `invariants`: four checks, `finite`, `nonnegative`, `conserved` and
  `monotone`, declared as data, each failing on any non-finite value.
- `runner` and `store`: one scenario run and stored atomically, with a seed
  derived from the scenario's identity. A run counts as finished only if it
  matches the current parameters, seed and model file and its series still
  read back.
- `results` and `report`: every stored run re-checked against the current
  invariants on each aggregation, one results table in CSV or Parquet, a JSON
  manifest with the code version, config hash, seeds and wall times, and a
  Markdown validation report listing every failure.
- `executors`: a local process pool, and SLURM job arrays through `sbatch`,
  split into several arrays above `MaxArraySize`.
- The `campaign` command: `run`, `report` and `run-one`, with exit codes meant
  for scripts.
- A demo sweeping an SIR epidemic over 80 scenarios, with its figure.
- 220 tests at 96% line and branch coverage, including the SLURM path run end
  to end against a stub `sbatch`.
- CI on Python 3.11 to 3.13 on Linux, macOS and Windows, plus jobs that lint,
  type-check in strict mode, build and validate the distributions, and run the
  demo twice to check the second run simulates nothing.
