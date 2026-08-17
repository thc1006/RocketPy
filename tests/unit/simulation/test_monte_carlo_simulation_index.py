import json

import pytest

from rocketpy.simulation.monte_carlo import MonteCarlo


def _indices(analysis):
    with open(analysis.output_file, "r", encoding="utf-8") as written:
        return sorted(json.loads(line)["index"] for line in written if line.strip())


def _a_study(tmp_path, stem, environment, rocket, flight):
    return MonteCarlo(
        filename=str(tmp_path / stem),
        environment=environment,
        rocket=rocket,
        flight=flight,
    )


@pytest.mark.parametrize("parallel", [False, True])
def test_a_run_numbers_its_simulations_from_zero(
    stochastic_environment, stochastic_calisto, stochastic_flight, tmp_path, parallel
):
    # Serial used to write 1, 2, 3 while parallel wrote 0, 1, 2, so the same
    # simulation had two names depending on how the run was started.
    analysis = _a_study(
        tmp_path,
        f"study-{parallel}",
        stochastic_environment,
        stochastic_calisto,
        stochastic_flight,
    )

    analysis.simulate(
        number_of_simulations=3,
        append=False,
        parallel=parallel,
        n_workers=2 if parallel else None,
    )

    assert _indices(analysis) == [0, 1, 2]


def test_both_modes_agree_on_what_a_simulation_is_called(
    stochastic_environment, stochastic_calisto, stochastic_flight, tmp_path
):
    one = _a_study(
        tmp_path,
        "serial",
        stochastic_environment,
        stochastic_calisto,
        stochastic_flight,
    )
    other = _a_study(
        tmp_path, "para", stochastic_environment, stochastic_calisto, stochastic_flight
    )

    one.simulate(number_of_simulations=3, append=False, parallel=False)
    other.simulate(number_of_simulations=3, append=False, parallel=True, n_workers=2)

    assert _indices(one) == _indices(other)


def test_an_appended_run_carries_on_from_the_last_index(
    stochastic_environment, stochastic_calisto, stochastic_flight, tmp_path
):
    analysis = _a_study(
        tmp_path, "study", stochastic_environment, stochastic_calisto, stochastic_flight
    )

    analysis.simulate(number_of_simulations=2, append=False)
    analysis.simulate(number_of_simulations=4, append=True)

    assert _indices(analysis) == [0, 1, 2, 3]
