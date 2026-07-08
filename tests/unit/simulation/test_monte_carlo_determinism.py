"""Unit tests for the Monte Carlo seeding helpers.

``MonteCarlo.simulate(random_seed=...)`` makes the sampled inputs reproducible by
turning the run's root seed into one independent child stream per simulation
index. Two private helpers do the work:

* ``__root_seed_sequence`` normalizes the flexible ``random_seed`` argument (int,
  ``SeedSequence``, ``Generator``, ``BitGenerator`` or None) into a
  ``SeedSequence`` that can be spawned;
* ``__seed_simulation`` splits one per-index child seed three ways so the
  environment, rocket and flight draw from independent streams.

These tests exercise the helpers directly, with no fixtures and no simulation, so
they stay fast. The end-to-end reproducibility of ``simulate`` (serial and across
workers) is covered by ``tests/integration/simulation/test_monte_carlo_determinism``.

Reaching a name-mangled member is an established pattern in this suite (see
``tests/unit/test_sensitivity.py`` and ``tests/unit/environment/test_environment.py``);
it lets the seeding invariants be asserted without running a Monte Carlo.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from rocketpy.simulation import MonteCarlo

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
        pytest.param(lambda: np.random.SeedSequence(12345), id="seedsequence"),
        pytest.param(lambda: np.random.default_rng(12345), id="generator"),
        pytest.param(lambda: np.random.PCG64(12345), id="bitgenerator"),
    ],
)
def test_root_seed_sequence_accepts_supported_types(make_seed):
    """int, SeedSequence, Generator and BitGenerator all normalize to the same
    root SeedSequence stream for an equivalent seed value."""
    root = _root_seed_sequence(make_seed())
    assert isinstance(root, np.random.SeedSequence)
    assert _entropy(root) == _entropy(_root_seed_sequence(12345))


def test_root_seed_sequence_none_draws_fresh_entropy():
    """None yields a SeedSequence seeded from fresh OS entropy (not reproducible)."""
    root = _root_seed_sequence(None)
    assert isinstance(root, np.random.SeedSequence)
    assert root.entropy is not None


@pytest.mark.parametrize(
    "make_seed, resolve",
    [
        pytest.param(
            lambda: np.random.SeedSequence(999),
            lambda seed: seed,
            id="seedsequence",
        ),
        pytest.param(
            lambda: np.random.default_rng(999),
            lambda seed: seed.bit_generator.seed_seq,
            id="generator",
        ),
        pytest.param(
            lambda: np.random.PCG64(999),
            lambda seed: seed.seed_seq,
            id="bitgenerator",
        ),
    ],
)
def test_root_seed_sequence_reuses_existing_seed_sequence(make_seed, resolve):
    """When given something that already carries a SeedSequence, the helper
    reuses that object rather than copying it."""
    seed = make_seed()
    assert _root_seed_sequence(seed) is resolve(seed)


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
