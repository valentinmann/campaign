# Contributing

This is a small research package maintained by one person. Issues and pull
requests are welcome; please open an issue before a large change so we can
agree on the shape of it first.

## Getting set up

```bash
git clone https://github.com/valentinmann/campaign.git
cd campaign
pip install -e ".[dev]"
pre-commit install
```

## Running the checks

Everything CI runs is available locally through [nox](https://nox.thea.codes):

```bash
nox              # lint, types and tests
nox -s tests     # tests on every installed interpreter
nox -s lint      # ruff check and ruff format --check
nox -s types     # mypy, strict mode, over the package and the demo
nox -s coverage  # tests with a coverage report
nox -s demo      # run the demo campaign and redraw the figure
nox -s build     # build the distributions and validate the metadata
```

Or directly, if you prefer:

```bash
pytest
ruff check . && ruff format --check .
mypy
```

The SLURM end-to-end tests need `bash`. On Windows the tests look for Git's
own bash first, since `bash` on the path can be the WSL launcher with nothing
installed behind it, and skip those tests if no working bash is found.

## What a change needs

- **A test that fails before it and passes after.** Several of this
  package's tests exist because something produced output that looked right
  and was not; that is the standard to hold.
- **Nothing that makes an unfinished run look finished.** Anything touching
  how runs are stored, identified or resumed needs a test that breaks a run
  in the relevant way and checks it is picked up again.
- **Measured claims stay measured.** Docstrings, the demo's campaign file
  and the README quote numbers from real runs. If a change moves them,
  measure again and update the text rather than the other way round.
- `ruff`, `ruff format` and `mypy --strict` clean. `pre-commit` enforces all
  three before the commit lands.

## Reporting a problem

The most useful report is the campaign file, the command that was run, and
the `report.md` or the error it produced. For a SLURM problem, the job
script and one array task's log from `output/slurm/` help most.
