"""End-to-end determinism tests for ``MonteCarlo.simulate(random_seed=...)``.

With a fixed ``random_seed`` the generated random *inputs* are reproducible and
identical across serial and parallel execution and across any number of workers.
Each simulation index draws from its own child stream spawned from the run's root
seed, and ``SeedSequence.spawn`` is prefix-stable, so index ``i`` maps to the same
seed regardless of the worker that runs it. (The seed-handling helpers themselves
are unit tested in ``tests/unit/simulation/test_monte_carlo_determinism``.)

The trajectory integration (``Flight``) is stubbed: worker invariance is a
property of the *input sampling*, which happens before ``Flight`` is built, so a
stub keeps the runs fast while still driving the real serial and parallel loops.
Stubbing the module-level ``Flight`` symbol reaches the parallel workers only
under the ``fork`` start method, so the worker-invariance test skips otherwise and
is marked ``slow`` to match the other Monte Carlo multiprocessing tests.

A dedicated numpy-only rocket keeps the fork-based end-to-end test simple: it
gives the motor a single ``thrust_source`` so the run has no list-valued attribute
at all. List sampling is itself seeded now (it draws through the model generator,
not the stdlib ``random.choice``) and is covered directly in
``tests/unit/stochastic/test_stochastic_model``.

Two tests cover the start methods, both unmarked.
``test_seed_derivation_is_start_method_invariant`` checks the derivation with a
top-level picklable target, and
``test_the_real_parallel_path_is_worker_invariant_under_every_start_method``
drives the shipped parallel path. Both set the method rather than taking the
platform default, which is worth the seconds it costs: ``multiprocess``
hard-codes ``fork`` on every POSIX platform, macOS and 3.14 included, so a
default-taking test would gate ``spawn`` on Windows and nothing anywhere else.
"""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

import rocketpy.simulation.monte_carlo as mc_module
from rocketpy import Environment
from rocketpy.simulation import MonteCarlo
from rocketpy.simulation.monte_carlo import _seed_sequence_to_int
from rocketpy.stochastic import (
    StochasticAirBrakes,
    StochasticEnvironment,
    StochasticRocket,
    StochasticSolidMotor,
)

_child_seed = MonteCarlo._MonteCarlo__child_seed

# Parametrizing over start methods runs while tests are collected, and
# `multiprocess` is an optional extra, so skipping there would take the whole
# module down instead of skipping it. The tests themselves still importorskip.
try:
    import multiprocess as _start_methods_from
except ImportError:
    import multiprocessing as _start_methods_from


def _available_start_methods():
    """The start methods this platform supports, as ``multiprocess`` sees them.

    Asked of ``multiprocess`` rather than the standard library, because that is
    what creates the workers and the two do not have to agree.
    """
    supported = _start_methods_from.get_all_start_methods()
    return [method for method in ("fork", "spawn", "forkserver") if method in supported]


def _derive_index_seeds(root_state, indices):
    """Derive the per-index seed fingerprints from ``root_state``.

    Top-level and picklable (only a small tuple and a list of ints cross the
    process boundary), so it runs unchanged under every start method -- including
    ``spawn``/``forkserver``, which re-import this module rather than inheriting
    the parent's memory. It calls the real production helpers (``__child_seed``
    and ``_seed_sequence_to_int``) so the test tracks the shipped derivation.
    """
    plan = SimpleNamespace(_MonteCarlo__root_state=root_state)
    return {index: _seed_sequence_to_int(_child_seed(plan, index)) for index in indices}


class _StubFlight:
    """Minimal stand-in for ``Flight`` that skips trajectory integration."""

    def __init__(self, **kwargs):  # accepts and ignores MonteCarlo's Flight kwargs
        pass

    def __getattr__(self, name):
        return 0.0


@pytest.fixture
def stochastic_calisto_numpy_only(
    cesaroni_m1670,
    calisto_robust,
    stochastic_nose_cone,
    stochastic_trapezoidal_fins,
    stochastic_tail,
    stochastic_rail_buttons,
    stochastic_main_parachute,
    stochastic_drogue_parachute,
):
    """A ``StochasticRocket`` whose randomness flows entirely through numpy.

    Mirrors the shared ``stochastic_calisto`` fixture but gives the solid motor a
    single ``thrust_source`` instead of a multi-element list, so no attribute is
    sampled through the unseeded standard-library ``random.choice``.
    """
    motor = StochasticSolidMotor(
        solid_motor=cesaroni_m1670,
        burn_out_time=(4, 0.1),
        grains_center_of_mass_position=0.001,
        grain_density=50,
        grain_separation=1 / 1000,
        grain_initial_height=1 / 1000,
        grain_initial_inner_radius=0.375 / 1000,
        grain_outer_radius=0.375 / 1000,
        total_impulse=(6500, 1000),
        throat_radius=0.5 / 1000,
        nozzle_radius=0.5 / 1000,
        nozzle_position=0.001,
    )
    rocket = StochasticRocket(
        rocket=calisto_robust,
        radius=0.0127 / 2000,
        mass=(15.426, 0.5, "normal"),
        inertia_11=(6.321, 0),
        inertia_22=0.01,
        inertia_33=0.01,
        center_of_mass_without_motor=0,
    )
    rocket.add_motor(motor, position=0.001)
    rocket.add_nose(stochastic_nose_cone, position=(1.134, 0.001))
    rocket.add_trapezoidal_fins(stochastic_trapezoidal_fins, position=(0.001, "normal"))
    rocket.add_tail(stochastic_tail)
    rocket.set_rail_buttons(
        stochastic_rail_buttons, lower_button_position=(-0.618, 0.001, "normal")
    )
    rocket.add_parachute(parachute=stochastic_main_parachute)
    rocket.add_parachute(parachute=stochastic_drogue_parachute)
    return rocket


def _read_inputs_by_index(input_file):
    """Read a ``.inputs.txt`` file into ``{index: raw_json_line}``."""
    by_index = {}
    with open(input_file, mode="r", encoding="utf-8") as rows:
        for line in rows:
            line = line.strip()
            if not line:
                continue
            by_index[json.loads(line)["index"]] = line
    return by_index


def _count_rows(log_file):
    """How many records were written, before anything is keyed by index.

    Keying by index hides a duplicate: two workers claiming the same index
    write two rows and the second overwrites the first in the dict, so the
    result looks complete. The claim is meant to be atomic, and the count is
    what says so.
    """
    with open(log_file, mode="r", encoding="utf-8") as rows:
        return sum(1 for line in rows if line.strip())


def _simulate_inputs(
    monkeypatch, tmp_path, environment, rocket, flight, tag, **simulate_kwargs
):
    """Run a Monte Carlo with a stubbed ``Flight`` and return inputs by index."""
    monkeypatch.setattr(mc_module, "Flight", _StubFlight)
    montecarlo = MonteCarlo(
        filename=str(tmp_path / tag),
        environment=environment,
        rocket=rocket,
        flight=flight,
    )
    montecarlo.simulate(**simulate_kwargs)
    return _read_inputs_by_index(montecarlo.input_file)


def test_invalid_seed_does_not_truncate_existing_output(
    monkeypatch,
    tmp_path,
    stochastic_environment,
    stochastic_calisto_numpy_only,
    stochastic_flight,
):
    """A rejected seed must fail before any output file is truncated, so passing
    an invalid seed cannot destroy the results of a previous run."""
    monkeypatch.setattr(mc_module, "Flight", _StubFlight)
    montecarlo = MonteCarlo(
        filename=str(tmp_path / "keep"),
        environment=stochastic_environment,
        rocket=stochastic_calisto_numpy_only,
        flight=stochastic_flight,
    )
    with open(montecarlo.input_file, "w", encoding="utf-8") as existing:
        existing.write("previous results\n")

    # A Generator is not a seed and is rejected; the run must raise before the
    # ``w+`` file setup truncates anything.
    with pytest.raises(TypeError):
        montecarlo.simulate(
            number_of_simulations=3, random_seed=np.random.default_rng(0)
        )

    with open(montecarlo.input_file, encoding="utf-8") as kept:
        assert kept.read() == "previous results\n"


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"number_of_simulations": 2.5}, TypeError),
        ({"number_of_simulations": True}, TypeError),
        ({"number_of_simulations": -1}, ValueError),
        ({"number_of_simulations": 3, "parallel": True, "n_workers": 1}, ValueError),
    ],
    ids=["float count", "boolean count", "negative count", "one worker"],
)
def test_a_rejected_argument_does_not_truncate_existing_output(
    monkeypatch,
    tmp_path,
    stochastic_environment,
    stochastic_calisto_numpy_only,
    stochastic_flight,
    kwargs,
    error,
):
    """Every check that needs only the arguments belongs before the logs open.

    ``__setup_files`` opens both of them "w+", which empties them, and
    ``n_workers`` was validated after that. So asking for a worker count the run
    cannot use destroyed the previous run's results on the way to raising.

    ``True`` is the one that does not raise on its own: it is an ``int`` to
    ``isinstance``, so it would quietly have run one simulation.
    """
    monkeypatch.setattr(mc_module, "Flight", _StubFlight)
    montecarlo = MonteCarlo(
        filename=str(tmp_path / f"keep-{sorted(kwargs.items())}"),
        environment=stochastic_environment,
        rocket=stochastic_calisto_numpy_only,
        flight=stochastic_flight,
    )
    with open(montecarlo.input_file, "w", encoding="utf-8") as existing:
        existing.write("previous results\n")

    with pytest.raises(error):
        montecarlo.simulate(random_seed=11, **kwargs)

    with open(montecarlo.input_file, encoding="utf-8") as kept:
        assert kept.read() == "previous results\n"


def test_serial_inputs_are_reproducible(
    monkeypatch,
    tmp_path,
    stochastic_environment,
    stochastic_calisto_numpy_only,
    stochastic_flight,
):
    """Two serial runs with the same seed yield byte-identical inputs per index.

    This drives the serial ``simulate`` path end to end; the flexible seed types
    are covered by the unit test of ``__root_seed_sequence``.
    """
    models = (stochastic_environment, stochastic_calisto_numpy_only, stochastic_flight)
    run_a = _simulate_inputs(
        monkeypatch, tmp_path, *models, "a", number_of_simulations=3, random_seed=7
    )
    run_b = _simulate_inputs(
        monkeypatch, tmp_path, *models, "b", number_of_simulations=3, random_seed=7
    )
    assert sorted(run_a) == list(range(3))
    assert run_a == run_b


@pytest.mark.slow
def test_inputs_are_worker_invariant(
    monkeypatch,
    tmp_path,
    stochastic_environment,
    stochastic_calisto_numpy_only,
    stochastic_flight,
):
    """serial == parallel(2) == parallel(4): inputs are bit-identical per index."""
    multiprocess = pytest.importorskip("multiprocess")
    if multiprocess.get_start_method() != "fork":
        pytest.skip(
            "stub-based parallel determinism test requires the 'fork' start method"
        )

    models = (stochastic_environment, stochastic_calisto_numpy_only, stochastic_flight)
    common = {"number_of_simulations": 8, "random_seed": 314159}

    serial = _simulate_inputs(monkeypatch, tmp_path, *models, "serial", **common)
    par2 = _simulate_inputs(
        monkeypatch, tmp_path, *models, "par2", parallel=True, n_workers=2, **common
    )
    par4 = _simulate_inputs(
        monkeypatch, tmp_path, *models, "par4", parallel=True, n_workers=4, **common
    )

    expected = list(range(8))
    assert sorted(serial) == expected
    assert sorted(par2) == expected
    assert sorted(par4) == expected
    for index in expected:
        assert serial[index] == par2[index], f"serial vs parallel(2) differ at {index}"
        assert serial[index] == par4[index], f"serial vs parallel(4) differ at {index}"


@pytest.mark.parametrize("start_method", _available_start_methods())
def test_seed_derivation_is_start_method_invariant(start_method):
    """Per-index seeds derived in a worker match the main process under every
    available start method (fork, spawn, forkserver).

    The full worker-invariance test above stubs the module-level ``Flight`` and so
    only reaches workers under ``fork``. This one instead checks the property that
    actually has to hold cross-platform -- that a simulation index maps to the same
    seed no matter which process derives it -- using a top-level picklable target
    and small picklable arguments, so it is valid under ``spawn``/``forkserver``
    without relying on any inherited parent state. Two workers split the indices;
    their combined result must equal the single-process derivation.
    """
    root = np.random.SeedSequence(2718281828)
    root_state = (
        root.entropy,
        root.spawn_key,
        root.pool_size,
        root.n_children_spawned,
    )
    indices = list(range(6))
    expected = _derive_index_seeds(root_state, indices)

    multiprocess = pytest.importorskip("multiprocess")
    context = multiprocess.get_context(start_method)
    chunks = [(root_state, indices[0::2]), (root_state, indices[1::2])]
    with context.Pool(2) as pool:
        results = pool.starmap(_derive_index_seeds, chunks)

    combined = {}
    for result in results:
        combined.update(result)
    assert combined == expected
    assert sorted(combined) == indices


def _assert_the_same_environment_was_flown(runs, expected_indices, start_method):
    """Every worker count flew index i with the same effective environment.

    This is the half the inputs file cannot show. It records
    ``wind_velocity_x_factor``, which is the same for index i however the run
    was executed even when the baseline it multiplies has drifted from one
    simulation to the next.
    """
    effective = {
        label: _read_inputs_by_index(montecarlo.output_file)
        for label, (montecarlo, _inputs) in runs.items()
    }
    for label, by_index in effective.items():
        assert sorted(by_index) == expected_indices, f"{label}: outputs are incomplete"

    for index in expected_indices:
        reference = json.loads(effective["serial"][index])
        for key in _EFFECTIVE_ENVIRONMENT:
            assert key in reference, f"{key} was not recorded"
        assert reference["effective_wind_x"] != 0.0, (
            "the wind baseline is zero, so a compounding baseline cannot show"
        )
        for label in ("parallel-2", "parallel-4"):
            drawn = json.loads(effective[label][index])
            for key in _EFFECTIVE_ENVIRONMENT:
                assert drawn[key] == reference[key], (
                    f"{start_method}: {label} flew a different {key} at index "
                    f"{index}: {drawn[key]} against {reference[key]}"
                )


def _assert_the_run_is_complete(label, montecarlo, inputs, count):
    """Every index written once, to both files, with nothing in the error log.

    The row counts are taken before anything is keyed by index: two workers
    claiming the same index write two rows, and the second overwrites the first
    in the dict, so a duplicate looks like a complete run.
    """
    expected_indices = list(range(count))
    rows = _count_rows(montecarlo.input_file)

    assert sorted(inputs) == expected_indices, (
        f"{label}: indices {sorted(inputs)}, expected {expected_indices}"
    )
    assert rows == count, (
        f"{label}: {rows} rows for {count} simulations, so an index was claimed "
        f"more than once"
    )
    assert _count_rows(montecarlo.output_file) == count, (
        f"{label}: the output rows do not match the simulations run"
    )
    assert sorted(_read_inputs_by_index(montecarlo.output_file)) == expected_indices, (
        f"{label}: the outputs do not match the inputs"
    )
    assert not os.path.getsize(montecarlo.error_file), (
        f"{label}: the run wrote to its error file"
    )


@pytest.fixture
def stochastic_environment_with_wind(example_spaceport_env):
    """A stochastic environment whose wind is not zero.

    The shared ``stochastic_environment`` fixture sits on an Environment whose
    ``wind_velocity_x`` is 0 at every altitude, and zero times any factor is
    zero, so a baseline that compounds from one simulation to the next cannot
    show up in it at all. Measured: with the baseline fix reverted, every
    assertion in this file still passed. A wind that is actually blowing is
    what makes the property testable.
    """
    environment = Environment(
        latitude=example_spaceport_env.latitude,
        longitude=example_spaceport_env.longitude,
        elevation=example_spaceport_env.elevation,
    )
    environment.set_atmospheric_model(
        type="custom_atmosphere", wind_u=12.0, wind_v=-7.0
    )
    return StochasticEnvironment(
        environment=environment,
        elevation=(1400, 10, "normal"),
        wind_velocity_x_factor=(1.0, 0.05, "normal"),
        wind_velocity_y_factor=(1.0, 0.05, "normal"),
    )


def _wind_x(flight):
    """The wind the simulation actually flew with, not the factor drawn for it."""
    return float(flight.env.wind_velocity_x(0))


def _wind_y(flight):
    return float(flight.env.wind_velocity_y(0))


def _elevation(flight):
    return float(flight.env.elevation)


_EFFECTIVE_ENVIRONMENT = {
    "effective_wind_x": _wind_x,
    "effective_wind_y": _wind_y,
    "effective_elevation": _elevation,
}


def _sampled_only(record):
    """The recorded inputs with object identity stripped out.

    A ``Function``'s ``signature.hash`` and its serialised ``source`` encode the
    object, not the value drawn for it, and an object built in another process
    has a different one. Under ``fork`` they happen to agree because the child
    inherits the parent's objects; under ``spawn`` and ``forkserver`` they
    cannot. Measured on a real run: six fields differ across the boundary and
    all six are these, while every sampled quantity matches exactly.
    """
    flat = {}

    def walk(value, path=""):
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{path}.{key}" if path else str(key))
        elif isinstance(value, list):
            for position, item in enumerate(value):
                walk(item, f"{path}[{position}]")
        else:
            flat[path] = value

    walk(record)
    return {
        key: value
        for key, value in flat.items()
        if "signature" not in key and not key.endswith(".source")
    }


def _real_run_inputs(tmp_path, environment, rocket, flight, tag, **simulate_kwargs):
    """Run a real Monte Carlo, no stub, and return the inputs keyed by index.

    Deliberately without the ``Flight`` stub. Stubbing is what confines the test
    above to ``fork``: it replaces a module-level symbol in the parent, and a
    ``spawn`` or ``forkserver`` child re-imports the module instead of inheriting
    it. A real run has nothing that needs to cross the boundary except the
    pickled MonteCarlo, which is the thing worth testing.
    """
    montecarlo = MonteCarlo(
        filename=str(tmp_path / tag),
        environment=environment,
        rocket=rocket,
        flight=flight,
        data_collector=_EFFECTIVE_ENVIRONMENT,
    )
    montecarlo.simulate(**simulate_kwargs)
    return montecarlo, _read_inputs_by_index(montecarlo.input_file)


@pytest.fixture
def restore_start_method():
    """Set the start method for one test and put it back afterwards."""
    multiprocess = pytest.importorskip("multiprocess")
    original = multiprocess.get_start_method()
    yield multiprocess
    multiprocess.set_start_method(original, force=True)


@pytest.mark.parametrize("start_method", _available_start_methods())
def test_the_real_parallel_path_is_worker_invariant_under_every_start_method(
    restore_start_method,
    tmp_path,
    stochastic_environment_with_wind,
    stochastic_calisto_numpy_only,
    stochastic_flight,
    calisto_air_brakes_clamp_on,
    start_method,
):
    """The whole parallel path, not just the seed arithmetic.

    ``test_seed_derivation_is_start_method_invariant`` covers the derivation on
    every start method, and the stubbed test above covers the real loop on
    ``fork``. Neither covers ``multiprocess.Process``, ``__sim_producer``, the
    manager proxies or pickling the stochastic object graph anywhere but
    ``fork``.

    Sets each method rather than taking the platform default, because
    ``multiprocess`` hard-codes ``fork`` on every POSIX platform including macOS
    and 3.14, so a default-taking test gates ``spawn`` on Windows and nothing
    else. Unmarked despite the cost: 26 s for the three, against 89 s for the
    rest of this directory, and it is the only thing covering the path this
    change exists to support.
    """
    multiprocess = restore_start_method
    if start_method not in multiprocess.get_all_start_methods():
        pytest.skip(f"{start_method} is not available here")
    multiprocess.set_start_method(start_method, force=True)

    # Air brakes and eccentricity are sampled by their own code paths, and each
    # one was reseeded from somewhere other than the simulation index.
    stochastic_calisto_numpy_only.add_air_brakes(
        StochasticAirBrakes(
            air_brakes=calisto_air_brakes_clamp_on.air_brakes[0],
            drag_coefficient_curve_factor=(1.0, 0.1),
        ),
        calisto_air_brakes_clamp_on._controllers[0],
    )
    stochastic_calisto_numpy_only.add_cp_eccentricity(x=(0.0, 0.001, "normal"), y=0.001)
    stochastic_calisto_numpy_only.add_thrust_eccentricity(
        x=(0.0, 0.001, "normal"), y=0.001
    )

    count = 4
    common = {"number_of_simulations": count, "random_seed": 987654321}
    models = (
        stochastic_environment_with_wind,
        stochastic_calisto_numpy_only,
        stochastic_flight,
    )
    runs = {
        "serial": _real_run_inputs(
            tmp_path, *models, f"{start_method}-serial", **common
        ),
        "parallel-2": _real_run_inputs(
            tmp_path,
            *models,
            f"{start_method}-p2",
            parallel=True,
            n_workers=2,
            **common,
        ),
        "parallel-4": _real_run_inputs(
            tmp_path,
            *models,
            f"{start_method}-p4",
            parallel=True,
            n_workers=4,
            **common,
        ),
    }

    expected_indices = list(range(count))
    for label, (montecarlo, inputs) in runs.items():
        _assert_the_run_is_complete(label, montecarlo, inputs, count)

    _assert_the_same_environment_was_flown(runs, expected_indices, start_method)

    serial = runs["serial"][1]
    for label in ("parallel-2", "parallel-4"):
        for index in expected_indices:
            expected = _sampled_only(json.loads(serial[index]))
            actual = _sampled_only(json.loads(runs[label][1][index]))

            # Or stripping identity could quietly empty the comparison.
            assert len(expected) > 20, f"only {len(expected)} fields left to compare"
            assert sum("eccentricity" in key for key in expected) == 4, (
                "the four eccentricities are not among the compared fields"
            )
            assert sum("brake" in key for key in expected) >= 1, (
                "the air brake is not among the compared fields"
            )
            assert actual == expected, (
                f"{start_method}: serial and {label} differ at index {index} in "
                f"{sorted(k for k in set(expected) | set(actual) if expected.get(k) != actual.get(k))}"
            )


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
def test_a_missing_simulation_is_not_reported_as_a_successful_run(
    monkeypatch,
    tmp_path,
    stochastic_environment,
    stochastic_calisto,
    stochastic_flight,
    parallel,
):
    """A run that wrote fewer records than it claimed has to fail.

    Neither file shows this on its own: every row is well formed, and reading
    them back keyed by index cannot tell four rows from three plus a duplicate.
    """
    montecarlo = MonteCarlo(
        filename=str(tmp_path / f"short-{parallel}"),
        environment=stochastic_environment,
        rocket=stochastic_calisto,
        flight=stochastic_flight,
    )
    kwargs = {"parallel": True, "n_workers": 2} if parallel else {}

    # Lose the simulation from both files, the way a worker that dies between
    # the claim and the writes does. Driven through ``simulate`` rather than by
    # calling the check afterwards, so this also proves the check is reached.
    real_inputs = montecarlo._MonteCarlo__evaluate_flight_inputs
    real_outputs = montecarlo._MonteCarlo__evaluate_flight_outputs

    def drop_the_second_inputs(sim_idx):
        return "" if sim_idx == 1 else real_inputs(sim_idx)

    def drop_the_second_outputs(flight, sim_idx):
        return "" if sim_idx == 1 else real_outputs(flight, sim_idx)

    monkeypatch.setattr(
        montecarlo, "_MonteCarlo__evaluate_flight_inputs", drop_the_second_inputs
    )
    monkeypatch.setattr(
        montecarlo, "_MonteCarlo__evaluate_flight_outputs", drop_the_second_outputs
    )

    with pytest.raises(RuntimeError, match="never written"):
        montecarlo.simulate(number_of_simulations=2, random_seed=5150, **kwargs)


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
def test_a_simulation_whose_outputs_went_missing_also_fails(
    monkeypatch,
    tmp_path,
    stochastic_environment,
    stochastic_calisto,
    stochastic_flight,
    parallel,
):
    """A worker that wrote its inputs and stopped before its outputs leaves the
    two logs disagreeing. Checking each file against the expected range on its
    own cannot see that: the inputs file is complete, and it is only complete
    because the row it is missing is in the other file.
    """
    montecarlo = MonteCarlo(
        filename=str(tmp_path / f"no-output-{parallel}"),
        environment=stochastic_environment,
        rocket=stochastic_calisto,
        flight=stochastic_flight,
    )
    kwargs = {"parallel": True, "n_workers": 2} if parallel else {}
    real = montecarlo._MonteCarlo__evaluate_flight_outputs

    def drop_the_second(flight, sim_idx):
        return "" if sim_idx == 1 else real(flight, sim_idx)

    monkeypatch.setattr(
        montecarlo, "_MonteCarlo__evaluate_flight_outputs", drop_the_second
    )

    with pytest.raises(RuntimeError, match="disagree about which simulations"):
        montecarlo.simulate(number_of_simulations=2, random_seed=5150, **kwargs)


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
def test_a_row_cut_off_mid_write_is_named_as_unreadable(
    monkeypatch,
    tmp_path,
    stochastic_environment,
    stochastic_calisto,
    stochastic_flight,
    parallel,
):
    """A worker killed part way through a write leaves a truncated row.

    That row is the corruption this check exists to find, so it is named and
    raised on. Skipping it and reporting the index as missing was a worse
    answer: with every expected index present, a corrupt file passed.

    This run wrote the whole file, so the damage is its own. A run appending
    onto a file an earlier run damaged is judged only on what it added, which
    ``test_an_append_run_is_not_judged_on_the_damage_it_inherited`` covers.
    """
    montecarlo = MonteCarlo(
        filename=str(tmp_path / f"truncated-{parallel}"),
        environment=stochastic_environment,
        rocket=stochastic_calisto,
        flight=stochastic_flight,
    )
    kwargs = {"parallel": True, "n_workers": 2} if parallel else {}
    real = montecarlo._MonteCarlo__evaluate_flight_inputs

    def cut_the_second_short(sim_idx):
        row = real(sim_idx)
        if sim_idx != 1:
            return row
        half = row[: len(row) // 2] + "\n"
        with pytest.raises(ValueError):
            json.loads(half)  # the row has to be unparseable for this to test it
        return half

    monkeypatch.setattr(
        montecarlo, "_MonteCarlo__evaluate_flight_inputs", cut_the_second_short
    )

    with pytest.raises(RuntimeError, match="cannot be read"):
        montecarlo.simulate(number_of_simulations=2, random_seed=5150, **kwargs)


def test_a_run_stopped_with_ctrl_c_keeps_what_it_saved(
    monkeypatch, tmp_path, stochastic_environment, stochastic_calisto, stochastic_flight
):
    """Ctrl-C is a stop, not a fault.

    The run path catches it, prints that the files are saved and returns. The
    completeness check then counted the simulations that never ran and called
    the run a failure, contradicting the message printed a moment earlier.
    """
    montecarlo = MonteCarlo(
        filename=str(tmp_path / "interrupted"),
        environment=stochastic_environment,
        rocket=stochastic_calisto,
        flight=stochastic_flight,
    )
    real = montecarlo._MonteCarlo__run_single_simulation
    finished = []

    def stop_after_the_first():
        if finished:
            raise KeyboardInterrupt("user pressed ctrl-c")
        finished.append(1)
        return real()

    monkeypatch.setattr(
        montecarlo, "_MonteCarlo__run_single_simulation", stop_after_the_first
    )

    montecarlo.simulate(number_of_simulations=3, random_seed=42)

    # Short of the three asked for, so the check really was in a position to
    # reject this run, and the one simulation that did finish is still there.
    assert _count_rows(montecarlo.input_file) == 1


def test_appending_continues_a_checkpoint_and_leaves_the_whole_range(
    tmp_path, stochastic_environment, stochastic_calisto, stochastic_flight
):
    """A second run carries on from the first, and the pair ends up whole.

    This test used to empty both logs before appending and then assert that a
    four-simulation result holding only indices 2 and 3 was a success. That is
    the shape of the bug it was meant to guard: the resume point came from a
    row count rather than the indices actually on disk, so nothing noticed the
    first two were gone. The run is judged on the whole range now.
    """
    montecarlo = MonteCarlo(
        filename=str(tmp_path / "appended"),
        environment=stochastic_environment,
        rocket=stochastic_calisto,
        flight=stochastic_flight,
    )
    montecarlo.simulate(number_of_simulations=2, random_seed=606)
    assert _count_rows(montecarlo.input_file) == 2

    montecarlo.simulate(number_of_simulations=4, append=True, random_seed=606)

    assert montecarlo._initial_sim_idx == 2, (
        "the second run should have started where the first stopped"
    )
    for label, path in (
        ("inputs", montecarlo.input_file),
        ("outputs", montecarlo.output_file),
    ):
        assert sorted(_read_inputs_by_index(path)) == [0, 1, 2, 3], (
            f"the {label} do not hold every simulation that was asked for"
        )


def test_appending_onto_a_checkpoint_with_a_hole_is_refused(
    tmp_path, stochastic_environment, stochastic_calisto, stochastic_flight
):
    """The other half, and the reason the resume point cannot be a row count.

    Two rows plus a blank line load as three simulations, so the next run would
    start at index 2 and leave index 1 missing for good while reporting
    success. Refused before it runs, with both files left as they were found.
    """
    montecarlo = MonteCarlo(
        filename=str(tmp_path / "holed"),
        environment=stochastic_environment,
        rocket=stochastic_calisto,
        flight=stochastic_flight,
    )
    montecarlo.simulate(number_of_simulations=2, random_seed=606)

    with open(montecarlo.output_file, "a", encoding="utf-8") as log:
        log.write("\n")
    montecarlo.set_num_of_loaded_sims()
    assert montecarlo.num_of_loaded_sims == 3, "the blank line was not counted"
    before = (
        montecarlo.input_file.read_bytes(),
        montecarlo.output_file.read_bytes(),
    )

    with pytest.raises(ValueError):
        montecarlo.simulate(number_of_simulations=5, append=True, random_seed=606)

    assert (
        montecarlo.input_file.read_bytes(),
        montecarlo.output_file.read_bytes(),
    ) == before, "a refused checkpoint was modified on the way out"


def test_a_missing_parallel_dependency_does_not_cost_the_previous_run(
    monkeypatch,
    tmp_path,
    stochastic_environment,
    stochastic_calisto_numpy_only,
    stochastic_flight,
):
    """``multiprocess`` is an optional extra, so an install without
    ``rocketpy[monte-carlo]`` cannot run in parallel at all.

    It used to be imported inside the parallel path, which runs after
    ``__setup_files`` has opened both logs "w+" and emptied them, so asking for
    a parallel run on such an install destroyed the previous results on the way
    to the ImportError.
    """
    monkeypatch.setattr(mc_module, "Flight", _StubFlight)
    montecarlo = MonteCarlo(
        filename=str(tmp_path / "kept"),
        environment=stochastic_environment,
        rocket=stochastic_calisto_numpy_only,
        flight=stochastic_flight,
    )
    with open(montecarlo.input_file, "w", encoding="utf-8") as existing:
        existing.write("previous results\n")

    def no_multiprocess():
        raise ImportError("No module named 'multiprocess'")

    monkeypatch.setattr(mc_module, "_import_multiprocess", no_multiprocess)

    with pytest.raises(ImportError):
        montecarlo.simulate(
            number_of_simulations=2, parallel=True, n_workers=2, random_seed=7
        )

    with open(montecarlo.input_file, encoding="utf-8") as kept:
        assert kept.read() == "previous results\n"


def test_appending_below_the_checkpoint_is_refused_and_changes_nothing(
    tmp_path, stochastic_environment, stochastic_calisto, stochastic_flight
):
    """``number_of_simulations`` is the total to reach, not a batch to add.

    Asking for fewer than the checkpoint already holds ran nothing and returned
    success: every index it wanted was present, so the completeness check was
    satisfied by simulations an earlier run had made. The caller was told three
    while the file held five.
    """
    montecarlo = MonteCarlo(
        filename=str(tmp_path / "shrunk"),
        environment=stochastic_environment,
        rocket=stochastic_calisto,
        flight=stochastic_flight,
    )
    montecarlo.simulate(number_of_simulations=3, random_seed=606)
    before = (
        montecarlo.input_file.read_bytes(),
        montecarlo.output_file.read_bytes(),
    )

    with pytest.raises(ValueError, match="already holds 3"):
        montecarlo.simulate(number_of_simulations=2, append=True, random_seed=606)

    assert (
        montecarlo.input_file.read_bytes(),
        montecarlo.output_file.read_bytes(),
    ) == before, "the refusal touched the checkpoint it was protecting"


def test_appending_to_the_size_it_already_has_is_allowed(
    tmp_path, stochastic_environment, stochastic_calisto, stochastic_flight
):
    """The control. The guard must refuse only what is below the checkpoint,
    not every append that adds no work."""
    montecarlo = MonteCarlo(
        filename=str(tmp_path / "unchanged"),
        environment=stochastic_environment,
        rocket=stochastic_calisto,
        flight=stochastic_flight,
    )
    montecarlo.simulate(number_of_simulations=2, random_seed=606)

    montecarlo.simulate(number_of_simulations=2, append=True, random_seed=606)

    assert sorted(_read_inputs_by_index(montecarlo.input_file)) == [0, 1]
