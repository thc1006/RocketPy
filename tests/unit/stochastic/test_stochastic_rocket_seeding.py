"""Nested StochasticRocket components are reseeded from distinct SeedSequence
children, so components that sample the same distribution (a main and a drogue
parachute, for example) do not draw identical values. Reproducible under a fixed
seed. See the seeding design in ``StochasticRocket._set_stochastic``.
"""

from rocketpy.stochastic.stochastic_model import StochasticModel

# Captured once, before any patching, so wrapping it repeatedly in one test does
# not stack (each recorder wraps the real method, not a previous recorder).
_REAL_SET_STOCHASTIC = StochasticModel._set_stochastic


def _record_component_seeds(monkeypatch, rocket, seed):
    """Return the seeds handed to every nested component for one reseed."""
    recorded = []

    def recording(self, seed=None):
        recorded.append(seed)
        return _REAL_SET_STOCHASTIC(self, seed)

    monkeypatch.setattr(StochasticModel, "_set_stochastic", recording)
    rocket._set_stochastic(seed)
    return recorded


def test_rocket_components_receive_distinct_seeds(monkeypatch, stochastic_calisto):
    """Every nested component (body, aerodynamic surfaces, motor, rail buttons and
    the two parachutes) is reseeded from its own child, so none collide."""
    seeds = _record_component_seeds(monkeypatch, stochastic_calisto, 42)

    assert len(seeds) > 3, "expected the rocket body plus several components"
    assert len(seeds) == len(set(seeds)), (
        "components share a seed -- they would draw perfectly correlated samples"
    )


def test_rocket_component_seeds_are_reproducible(monkeypatch, stochastic_calisto):
    """The same root seed reseeds every component identically; a different root
    seed changes them."""
    first = _record_component_seeds(monkeypatch, stochastic_calisto, 42)
    again = _record_component_seeds(monkeypatch, stochastic_calisto, 42)
    different = _record_component_seeds(monkeypatch, stochastic_calisto, 43)

    assert again == first, "same seed must reproduce every component seed"
    assert different != first, "a different seed must change the component seeds"
