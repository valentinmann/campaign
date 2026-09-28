"""Reading campaign files: what gets computed is decided here.

The tests that matter most are the ones about identity. Resume skips a
scenario because its identifier says it already ran, so an identifier that
depends on a scenario's position, on how a number was typed, or on the order
of keys in the file would make resume skip the wrong work.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from campaign.config import ConfigError, load, scenario_id

BASIC = """
    name: toy
    model: model.py:simulate
    seed: 7
    output: out
    grid:
      a: [1, 2, 3]
      b: {start: 0.15, stop: 0.6, num: 10}
"""


def _load(write_campaign, body: str):
    return load(write_campaign(body))


# --------------------------------------------------------------------------
# what gets computed
# --------------------------------------------------------------------------


def test_the_grid_is_every_combination_with_the_last_axis_fastest(write_campaign):
    c = _load(write_campaign, BASIC)
    assert len(c.scenarios) == 30
    assert c.swept == ("a", "b")
    first, second, eleventh = c.scenarios[0], c.scenarios[1], c.scenarios[10]
    assert (first.params["a"], first.params["b"]) == (1, 0.15)
    assert (second.params["a"], second.params["b"]) == (1, 0.2)
    assert (eleventh.params["a"], eleventh.params["b"]) == (2, 0.15)


def test_generated_values_are_the_numbers_meant(write_campaign):
    """A linspace yields 0.30000000000000004; the table should say 0.3."""
    c = _load(write_campaign, BASIC)
    b_values = sorted({s.params["b"] for s in c.scenarios})
    assert b_values == [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]


def test_exponent_notation_is_read_as_a_number(write_campaign):
    """PyYAML's YAML 1.1 reads `1e-9` as a string. This loader must not."""
    c = _load(
        write_campaign,
        """
        name: toy
        model: model.py:simulate
        seed: 1
        output: out
        grid:
          a: [1e3, 2.5e3, -4E-2]
        invariants:
          - {check: conserved, series: [x, y], rtol: 1e-9}
        """,
    )
    assert [s.params["a"] for s in c.scenarios] == [1000.0, 2500.0, -0.04]
    assert all(isinstance(s.params["a"], float) for s in c.scenarios)
    assert c.invariants[0].rtol == 1e-9


def test_an_explicit_list_is_run_as_given(write_campaign):
    c = _load(
        write_campaign,
        """
        name: toy
        model: model.py:simulate
        seed: 1
        output: out
        scenarios:
          - {a: 5, mode: fast}
          - {a: 2, mode: slow}
        """,
    )
    assert [s.params for s in c.scenarios] == [
        {"a": 5, "mode": "fast"},
        {"a": 2, "mode": "slow"},
    ]


def test_constants_reach_every_scenario_without_being_swept(write_campaign):
    c = _load(write_campaign, BASIC + "    constants: {gamma: 0.1, label: base}\n")
    assert c.swept == ("a", "b")
    assert all(
        s.params["gamma"] == 0.1 and s.params["label"] == "base" for s in c.scenarios
    )


def test_paths_resolve_against_the_file_not_the_working_directory(
    write_campaign, tmp_path, monkeypatch
):
    path = write_campaign(BASIC)
    elsewhere = tmp_path / "somewhere" / "else"
    elsewhere.mkdir(parents=True)
    monkeypatch.chdir(elsewhere)
    c = load(path)
    assert c.output == (tmp_path / "out").resolve()
    assert c.model_file == (tmp_path / "model.py").resolve()


def test_a_model_in_a_subdirectory_and_a_colon_in_the_path(write_campaign, tmp_path):
    """The function is after the last colon, so a drive letter in a path is harmless."""
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "m.py").write_text("def run(p, s): ...\n", encoding="utf-8")
    body = BASIC.replace("model.py:simulate", f"{(tmp_path / 'models' / 'm.py')}:run")
    c = _load(write_campaign, body)
    assert c.model_function == "run"
    assert c.model_file.name == "m.py"


# --------------------------------------------------------------------------
# identity: what resume relies on
# --------------------------------------------------------------------------


def test_identifiers_follow_content_not_position(write_campaign):
    """Insert a value at the front of an axis: every old scenario keeps its id."""
    before = _load(write_campaign, BASIC)
    after = _load(write_campaign, BASIC.replace("a: [1, 2, 3]", "a: [0, 1, 2, 3]"))
    assert {s.id for s in before.scenarios} < {s.id for s in after.scenarios}
    assert before.scenarios[0].id == after.scenarios[10].id


def test_identifiers_ignore_how_a_number_is_written_and_key_order():
    assert scenario_id("m.py:f", {"a": 0.3, "b": 1}) == scenario_id(
        "m.py:f", {"b": 1, "a": 0.30}
    )
    assert scenario_id("m.py:f", {"a": 0.3}) == scenario_id("m.py:f", {"a": 3.0e-1})
    assert scenario_id("m.py:f", {"a": 0.3}) != scenario_id("m.py:g", {"a": 0.3})
    assert scenario_id("m.py:f", {"a": 0.3}) != scenario_id(
        "m.py:f", {"a": 0.30000000000000004}
    )


def test_the_config_hash_tracks_results_not_formatting_or_execution(write_campaign):
    base = _load(write_campaign, BASIC).config_hash
    reformatted = BASIC.replace("seed: 7", "seed:    7   # the seed")
    assert _load(write_campaign, reformatted).config_hash == base
    executed_elsewhere = BASIC + "    backend: slurm\n    workers: 8\n"
    assert _load(write_campaign, executed_elsewhere).config_hash == base

    assert _load(write_campaign, BASIC.replace("seed: 7", "seed: 8")).config_hash != base
    with_invariant = BASIC + "    invariants:\n      - {check: finite, series: x}\n"
    assert _load(write_campaign, with_invariant).config_hash != base


# --------------------------------------------------------------------------
# refusing bad files, with a message that says where
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (("seed: 7", "seed: 7\n    ouput: x"), "unknown key.*ouput"),
        (("    seed: 7\n", ""), "missing required key.*seed"),
        (("name: toy", "name: my campaign"), "name: letters"),
        (("seed: 7", "seed: -1"), "seed: expected a non-negative integer"),
        (("seed: 7", "seed: true"), "seed: expected a non-negative integer"),
        (("model.py:simulate", "model.py"), "path/to/file.py:function"),
        (("model.py:simulate", "missing.py:simulate"), "not an existing .py file"),
        (("model.py:simulate", "model.py:not-a-name"), "not a valid function name"),
        (("a: [1, 2, 3]", "a: []"), "a has no values"),
        (("a: [1, 2, 3]", "a: [1, 2, 1]"), "lists a value twice"),
        (("a: [1, 2, 3]", "a: [1, .nan]"), "must be finite"),
        (("a: [1, 2, 3]", "a: [1, [2]]"), "numbers, strings or booleans"),
        (("num: 10", "num: 0"), "num must be a positive integer"),
        (("stop: 0.6", "stop: 0.15"), "start equals stop"),
        (("start: 0.15", "start: low"), "start and stop must be numbers"),
        (("{start: 0.15, stop: 0.6, num: 10}", "{start: 0, stop: 1}"), "exactly start"),
    ],
)
def test_malformed_files_are_rejected(write_campaign, edit, message):
    old, new = edit
    assert old in BASIC
    with pytest.raises(ConfigError, match=message):
        _load(write_campaign, BASIC.replace(old, new, 1))


def test_grid_and_scenarios_are_mutually_exclusive(write_campaign):
    with pytest.raises(ConfigError, match="exactly one of grid or scenarios"):
        _load(write_campaign, BASIC + "    scenarios:\n      - {a: 1}\n")
    neither = "\n".join(BASIC.splitlines()[:5]) + "\n"
    with pytest.raises(ConfigError, match="exactly one of grid or scenarios"):
        _load(write_campaign, neither)


@pytest.mark.parametrize(
    ("scenarios", "message"),
    [
        ("- {a: 1, beta: 2}\n      - {a: 1, betta: 2}", "every scenario must set the same"),
        ("- {a: 1}\n      - {a: 1}", r"scenarios\[1\] repeats scenarios\[0\]"),
        ("- 5", "non-empty mapping"),
    ],
)
def test_explicit_scenarios_are_checked(write_campaign, scenarios, message):
    body = "\n".join(BASIC.splitlines()[:5]) + f"\n    scenarios:\n      {scenarios}\n"
    with pytest.raises(ConfigError, match=message):
        _load(write_campaign, body)


def test_a_constant_cannot_also_be_swept(write_campaign):
    with pytest.raises(ConfigError, match="a is also swept"):
        _load(write_campaign, BASIC + "    constants: {a: 4}\n")


def test_yaml_tags_that_build_python_objects_are_refused(write_campaign):
    """A campaign file is data. It must never be able to run code."""
    with pytest.raises(ConfigError, match="python/object"):
        _load(
            write_campaign,
            BASIC.replace("seed: 7", "seed: !!python/object/apply:os.getpid []"),
        )


def test_invariant_errors_say_which_entry(write_campaign):
    body = (
        BASIC
        + "    invariants:\n      - {check: finite, series: x}\n      - {check: odd}\n"
    )
    with pytest.raises(ConfigError, match=r"invariants\[1\]"):
        _load(write_campaign, body)


def test_two_invariants_with_one_name_are_rejected(write_campaign):
    body = BASIC + (
        "    invariants:\n"
        "      - {check: finite, series: x}\n"
        "      - {check: finite, series: x}\n"
    )
    with pytest.raises(ConfigError, match="declared twice"):
        _load(write_campaign, body)


def test_parquet_without_pyarrow_fails_before_anything_runs(write_campaign, monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *_: None if name == "pyarrow" else real(name),
    )
    with pytest.raises(ConfigError, match="pip install 'campaign\\[parquet\\]'"):
        _load(write_campaign, BASIC + "    format: parquet\n")


# --------------------------------------------------------------------------
# SLURM settings: passed through, never invented
# --------------------------------------------------------------------------


def test_slurm_settings_pass_through_and_default_to_nothing(write_campaign):
    plain = _load(write_campaign, BASIC)
    assert plain.slurm.options == {}
    assert plain.slurm.max_array_size == 1001

    body = BASIC + (
        "    slurm:\n"
        "      options:\n"
        "        partition: short\n"
        "        time: '00:10:00'\n"
        "        mem-per-cpu: 2G\n"
        "        cpus-per-task: 1\n"
        "      throttle: 20\n"
        "      sbatch: /opt/slurm/bin/sbatch\n"
    )
    c = _load(write_campaign, body)
    assert c.slurm.options == {
        "partition": "short",
        "time": "00:10:00",
        "mem-per-cpu": "2G",
        "cpus-per-task": "1",
    }
    assert c.slurm.throttle == 20
    assert c.slurm.sbatch == ("/opt/slurm/bin/sbatch",)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ("{array: 0-9}", "set by campaign itself"),
        ("{output: log.txt}", "set by campaign itself"),
        ('{comment: "a\\nb"}', "line break"),
        ("{'bad name': x}", "not a valid option name"),
    ],
)
def test_slurm_options_cannot_rewrite_the_job_script(write_campaign, options, message):
    with pytest.raises(ConfigError, match=message):
        _load(write_campaign, BASIC + f"    slurm:\n      options: {options}\n")


def test_unknown_slurm_keys_are_rejected(write_campaign):
    with pytest.raises(ConfigError, match=r"slurm: unknown key.*partition"):
        _load(write_campaign, BASIC + "    slurm: {partition: short}\n")


def test_a_missing_file_is_a_config_error(tmp_path: Path):
    with pytest.raises(ConfigError, match="cannot read"):
        load(tmp_path / "nope.yaml")
