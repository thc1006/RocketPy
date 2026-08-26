import json

import pytest

from rocketpy.simulation.monte_carlo import MonteCarlo, _SIMULATION_ROOT_KEY


def _study(tmp_path, stem, models):
    environment, rocket, flight = models
    return MonteCarlo(
        filename=str(tmp_path / stem),
        environment=environment,
        rocket=rocket,
        flight=flight,
    )


def _drawn(analysis):
    with open(analysis.input_file, "r", encoding="utf-8") as written:
        rows = [json.loads(line) for line in written if line.strip()]
    return {row["index"]: row.get("mass") for row in rows}


@pytest.fixture(name="models")
def _models(stochastic_environment, stochastic_calisto, stochastic_flight):
    return stochastic_environment, stochastic_calisto, stochastic_flight


def test_a_row_says_which_root_drew_it(models, tmp_path):
    """Every input row carries the root, so the log describes itself."""
    analysis = _study(tmp_path, "study", models)

    analysis.simulate(2, append=False, random_seed=42)

    with open(analysis.input_file, "r", encoding="utf-8") as written:
        rows = [json.loads(line) for line in written if line.strip()]
    assert all(row[_SIMULATION_ROOT_KEY]["entropy"] == 42 for row in rows)


def test_an_append_with_no_seed_carries_on_from_the_rows(models, tmp_path):
    """The ordinary resume: no seed given, the stream continues anyway."""
    whole = _study(tmp_path, "whole", models)
    whole.simulate(4, append=False, random_seed=42)

    part = _study(tmp_path, "part", models)
    part.simulate(2, append=False, random_seed=42)
    part.simulate(4, append=True)

    assert _drawn(part) == _drawn(whole)


def test_a_fresh_object_can_continue_a_study(models, tmp_path):
    """A notebook restart is a new object over the same files."""
    first = _study(tmp_path, "study", models)
    first.simulate(2, append=False, random_seed=42)

    second = _study(tmp_path, "study", models)
    second.simulate(4, append=True)

    assert sorted(_drawn(second)) == [0, 1, 2, 3]


def test_appending_with_another_seed_is_refused(models, tmp_path):
    """Two roots in one file is the thing this exists to prevent."""
    analysis = _study(tmp_path, "study", models)
    analysis.simulate(2, append=False, random_seed=42)

    with pytest.raises(ValueError, match="different root"):
        analysis.simulate(4, append=True, random_seed=7)


def test_appending_with_the_same_seed_is_allowed(models, tmp_path):
    """The control. Saying the seed again is not a mismatch."""
    analysis = _study(tmp_path, "study", models)
    analysis.simulate(2, append=False, random_seed=42)

    analysis.simulate(4, append=True, random_seed=42)

    assert sorted(_drawn(analysis)) == [0, 1, 2, 3]


def test_a_log_holding_two_studies_is_refused(models, tmp_path):
    """Rows that disagree are two studies, and neither is safe to continue."""
    analysis = _study(tmp_path, "study", models)
    analysis.simulate(2, append=False, random_seed=42)
    with open(analysis.input_file, "r", encoding="utf-8") as written:
        rows = [json.loads(line) for line in written if line.strip()]
    rows[1][_SIMULATION_ROOT_KEY]["entropy"] = 999
    with open(analysis.input_file, "w", encoding="utf-8") as rewritten:
        for row in rows:
            rewritten.write(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="more than one study"):
        analysis.simulate(4, append=True)


def test_a_log_whose_rows_carry_no_root_is_refused(models, tmp_path):
    """A study from before this release cannot be shown to be one study."""
    # Measured before this was refused: the append read the log as empty,
    # started a root of its own, and left two lineages in the one file.
    analysis = _study(tmp_path, "study", models)
    analysis.simulate(2, append=False, random_seed=42)
    with open(analysis.input_file, "r", encoding="utf-8") as written:
        rows = [json.loads(line) for line in written if line.strip()]
    for row in rows:
        row.pop(_SIMULATION_ROOT_KEY)
    with open(analysis.input_file, "w", encoding="utf-8") as rewritten:
        for row in rows:
            rewritten.write(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="does not say which root"):
        analysis.simulate(4, append=True)


def test_an_empty_log_is_not_a_log_that_cannot_be_checked(models, tmp_path):
    """The control. Nothing recorded yet is a fresh start, not a refusal."""
    analysis = _study(tmp_path, "study", models)

    analysis.simulate(2, append=True, random_seed=42)

    assert sorted(_drawn(analysis)) == [0, 1]


def test_a_seed_given_as_a_sequence_is_recorded_as_one(models, tmp_path):
    """A sequence is a documented seed, so the row has to carry it whole."""
    analysis = _study(tmp_path, "study", models)

    analysis.simulate(2, append=False, random_seed=[1, 2, 3])

    with open(analysis.input_file, "r", encoding="utf-8") as written:
        rows = [json.loads(line) for line in written if line.strip()]
    assert all(row[_SIMULATION_ROOT_KEY]["entropy"] == [1, 2, 3] for row in rows)


def test_a_sequence_seed_still_continues_the_same_study(models, tmp_path):
    """And reads back, since a root that cannot be compared refuses the append."""
    whole = _study(tmp_path, "whole", models)
    whole.simulate(4, append=False, random_seed=[1, 2, 3])

    part = _study(tmp_path, "part", models)
    part.simulate(2, append=False, random_seed=[1, 2, 3])
    part.simulate(4, append=True)

    assert _drawn(part) == _drawn(whole)


def test_blank_lines_between_rows_do_not_hide_the_root(models, tmp_path):
    """A gap in the file is not a row, and not a study without a root either."""
    analysis = _study(tmp_path, "study", models)
    analysis.simulate(2, append=False, random_seed=42)
    with open(analysis.input_file, "r", encoding="utf-8") as written:
        rows = written.read().splitlines()
    with open(analysis.input_file, "w", encoding="utf-8") as spaced:
        for row in rows:
            spaced.write(row + "\n\n")

    analysis.simulate(4, append=True)

    assert sorted(_drawn(analysis)) == [0, 1, 2, 3]


def test_a_row_that_cannot_be_read_is_refused(models, tmp_path):
    """A row that will not parse leaves nothing to check the root against."""
    analysis = _study(tmp_path, "study", models)
    analysis.simulate(2, append=False, random_seed=42)
    with open(analysis.input_file, "a", encoding="utf-8") as damaged:
        damaged.write("{ this was cut off\n")

    with pytest.raises(ValueError, match="cannot be read"):
        analysis.simulate(4, append=True)
