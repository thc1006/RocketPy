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

Seed derivation being independent of the multiprocessing start method (fork,
spawn or forkserver) is verified separately by
``test_seed_derivation_is_start_method_invariant``, which uses a top-level
picklable target so it is safe under ``spawn``/``forkserver`` -- unlike the
``Flight``-stub test above, which reaches workers only under ``fork``.
"""

import json
import multiprocessing
from types import SimpleNamespace

import numpy as np
import pytest

import rocketpy.simulation.monte_carlo as mc_module
from rocketpy.simulation import MonteCarlo
from rocketpy.simulation.monte_carlo import _seed_sequence_to_int
from rocketpy.stochastic import StochasticRocket, StochasticSolidMotor

_child_seed = MonteCarlo._MonteCarlo__child_seed


def _available_start_methods():
    """The multiprocessing start methods this platform actually supports."""
    supported = multiprocessing.get_all_start_methods()
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
    (Python 3.14's POSIX default) without relying on any inherited parent state.
    Two workers split the indices; their combined result must equal the
    single-process derivation.
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

    context = multiprocessing.get_context(start_method)
    chunks = [(root_state, indices[0::2]), (root_state, indices[1::2])]
    with context.Pool(2) as pool:
        results = pool.starmap(_derive_index_seeds, chunks)

    combined = {}
    for result in results:
        combined.update(result)
    assert combined == expected
    assert sorted(combined) == indices
