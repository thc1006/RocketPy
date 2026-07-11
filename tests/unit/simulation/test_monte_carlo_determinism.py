"""Unit tests for the Monte Carlo seeding helpers.

``MonteCarlo.simulate(random_seed=...)`` makes the sampled inputs reproducible by
turning the run's root seed into one independent child stream per simulation
index. Two private helpers do the work:

* ``__root_seed_sequence`` normalizes the ``random_seed`` argument (an int, a
  sequence of ints, a ``SeedSequence`` or None) into a fresh ``SeedSequence``
  that can be spawned;
* ``__seed_simulation`` splits one per-index child seed three ways so the
  environment, rocket and flight draw from independent streams.

These tests exercise the helpers directly, with no fixtures and no simulation, so
they stay fast. The end-to-end reproducibility of ``simulate`` (serial and across
workers) is covered by ``tests/integration/simulation/test_monte_carlo_determinism``.

Reaching a name-mangled member is an established pattern in this suite (see
``tests/unit/test_sensitivity.py`` and ``tests/unit/environment/test_environment.py``);
it lets the seeding invariants be asserted without running a Monte Carlo.
"""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from rocketpy.simulation import MonteCarlo
from rocketpy.simulation.monte_carlo import _SimMonitor, _claim_next_index

_root_seed_sequence = MonteCarlo._MonteCarlo__root_seed_sequence
_seed_simulation = MonteCarlo._MonteCarlo__seed_simulation


def _entropy(seed_sequence, n=4):
    """A stable, comparable fingerprint of a ``SeedSequence``'s stream."""
    return tuple(int(x) for x in seed_sequence.generate_state(n))


# --------------------------------------------------------------------------- #
# __root_seed_sequence: normalizing the flexible seed argument                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "make_seed",
    [
        pytest.param(lambda: 12345, id="int"),
        pytest.param(lambda: np.int64(12345), id="numpy-int"),
        pytest.param(lambda: [1, 2, 3], id="sequence"),
        pytest.param(lambda: np.random.SeedSequence(12345), id="seedsequence"),
    ],
)
def test_root_seed_sequence_accepts_seed_like_values(make_seed):
    """An int, a numpy integer, a sequence of ints and a SeedSequence are all
    accepted, normalize to a SeedSequence, and are reproducible."""
    root = _root_seed_sequence(make_seed())
    assert isinstance(root, np.random.SeedSequence)
    assert _entropy(root) == _entropy(_root_seed_sequence(make_seed()))


def test_root_seed_sequence_none_draws_fresh_entropy():
    """None yields a SeedSequence seeded from fresh OS entropy (not reproducible)."""
    root = _root_seed_sequence(None)
    assert isinstance(root, np.random.SeedSequence)
    assert root.entropy is not None


@pytest.mark.parametrize(
    "make_generator",
    [
        pytest.param(lambda: np.random.default_rng(999), id="generator"),
        pytest.param(lambda: np.random.PCG64(999), id="bitgenerator"),
    ],
)
def test_root_seed_sequence_rejects_stateful_generators(make_generator):
    """A Generator/BitGenerator is a stateful RNG, not a seed value, so it is
    rejected instead of being reduced to its underlying SeedSequence."""
    with pytest.raises(TypeError, match="SeedSequence"):
        _root_seed_sequence(make_generator())


def test_root_seed_sequence_copies_seedsequence_without_mutating_it():
    """A supplied SeedSequence is copied from its full state: repeated calls with
    the same object reproduce the same children, the caller's spawn counter is
    left untouched, and a spawned child (non-empty spawn_key) round-trips too."""

    def children(seed_sequence):
        return [
            _entropy(child) for child in _root_seed_sequence(seed_sequence).spawn(3)
        ]

    # A SeedSequence that has already spawned children, so its counter is not 0.
    seed = np.random.SeedSequence(2024)
    seed.spawn(5)
    counter_before = seed.n_children_spawned

    assert children(seed) == children(seed), "same object twice must reproduce"
    assert seed.n_children_spawned == counter_before, "caller must not be mutated"
    assert _root_seed_sequence(seed) is not seed, "must return a copy, not the caller"

    # A spawned child carries a non-empty spawn_key that the copy must preserve.
    child = np.random.SeedSequence(2024).spawn(1)[0]
    assert child.spawn_key != ()
    assert children(child) == children(child)


# --------------------------------------------------------------------------- #
# __seed_simulation: splitting one child seed across the three models         #
# --------------------------------------------------------------------------- #


class _RecordingModel:
    """Stand-in stochastic model that records the seeds it is handed."""

    def __init__(self):
        self.seeds = []

    def _set_stochastic(self, seed=None):
        self.seeds.append(seed)


def _split_seeds(child_seed):
    """Run ``__seed_simulation`` against recording models; return the three seeds."""
    models = SimpleNamespace(
        environment=_RecordingModel(),
        rocket=_RecordingModel(),
        flight=_RecordingModel(),
    )
    _seed_simulation(models, child_seed)
    return models.environment.seeds, models.rocket.seeds, models.flight.seeds


def test_seed_simulation_decorrelates_env_rocket_flight():
    """The per-index child seed is split three ways so environment, rocket and
    flight draw from independent streams instead of sharing one."""
    env_seeds, rocket_seeds, flight_seeds = _split_seeds(np.random.SeedSequence(2024))
    assert [len(env_seeds), len(rocket_seeds), len(flight_seeds)] == [1, 1, 1]
    fingerprints = {
        _entropy(env_seeds[0]),
        _entropy(rocket_seeds[0]),
        _entropy(flight_seeds[0]),
    }
    assert len(fingerprints) == 3


def test_seed_simulation_is_deterministic_per_child():
    """A given child seed reseeds the three models identically every time."""

    def split(child):
        env, rocket, flight = _split_seeds(child)
        return [_entropy(env[0]), _entropy(rocket[0]), _entropy(flight[0])]

    assert split(np.random.SeedSequence(2024)) == split(np.random.SeedSequence(2024))


# --------------------------------------------------------------------------- #
# _claim_next_index: atomic hand-out of the next simulation index             #
# --------------------------------------------------------------------------- #


def test_claim_next_index_hands_out_each_index_once_under_contention():
    """Holding the mutex across keep_simulating() and increment() must hand out
    each index exactly once, even when every worker reaches the claim together.

    A barrier releases all workers at once and a widened check-to-increment
    window would let an unlocked claim run several workers past the count < n
    check before any increments; the lock is what keeps the result to exactly
    n_simulations indices (0..n-1, none repeated) and the counter from
    overshooting.
    """
    n_simulations = 5
    n_workers = 8
    monitor = _SimMonitor(initial_count=0, n_simulations=n_simulations, start_time=0.0)

    # Widen the window between the check and the increment so that, without the
    # lock, several workers could pass count < n before any of them increments.
    real_keep_simulating = monitor.keep_simulating

    def slow_keep_simulating():
        result = real_keep_simulating()
        time.sleep(0.02)
        return result

    monitor.keep_simulating = slow_keep_simulating

    mutex = threading.Lock()
    barrier = threading.Barrier(n_workers)
    claimed = []
    claimed_lock = threading.Lock()

    def worker():
        barrier.wait()
        while True:
            index = _claim_next_index(monitor, mutex)
            if index is None:
                break
            with claimed_lock:
                claimed.append(index)

    workers = [threading.Thread(target=worker) for _ in range(n_workers)]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()

    assert sorted(claimed) == list(range(n_simulations))
    assert monitor.count == n_simulations
