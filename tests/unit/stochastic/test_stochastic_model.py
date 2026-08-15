import numpy as np
import pytest

from rocketpy import Environment
from rocketpy.mathutils.function import Function
from rocketpy.stochastic import StochasticEnvironment, StochasticFreeFormFins
from rocketpy.stochastic.stochastic_model import _snapshot_of


@pytest.mark.parametrize(
    "fixture_name",
    [
        "stochastic_rail_buttons",
        "stochastic_main_parachute",
        "stochastic_environment",
        "stochastic_environment_custom_sampler",
        "stochastic_tail",
        "stochastic_calisto",
        "stochastic_free_form_fins",
    ],
)
def test_visualize_attributes(request, fixture_name):
    """Tests the visualize_attributes method of the StochasticModel class. It
    must run without breaking and return the formatted report string (which is
    also printed), so the report is never silently lost.
    """
    fixture = request.getfixturevalue(fixture_name)
    report = fixture.visualize_attributes()
    assert isinstance(report, str)
    assert report


def test_list_choices_are_reproducible(calisto_free_form_fins):
    """Choosing between the candidate values of a list input must come from the
    model's own generator, so that the same seed replays the same choices.

    The interpreter-wide ``random.choice`` was used, which ``_set_stochastic``
    does not reseed: a fixed-seed run picked different values every time, and
    Monte Carlo workers forked from one process walked a single shared stream
    instead of sampling independently.
    """
    taller = [(0, 0), (0.06, 0.12), (0.12, 0.12), (0.12, 0)]
    stochastic = StochasticFreeFormFins(
        free_form_fins=calisto_free_form_fins,
        shape_points=[calisto_free_form_fins.shape_points, taller],
    )

    def spans(seed):
        stochastic._set_stochastic(seed)
        return [round(stochastic.create_object().span, 4) for _ in range(20)]

    assert spans(7) == spans(7)
    assert spans(7) != spans(8)
    # Both candidates must stay reachable, or the assertions above would also
    # hold for a generator that always returned the same one.
    assert set(spans(7)) == {0.1, 0.12}


def _windy_environment():
    environment = Environment()
    environment.set_atmospheric_model(type="standard_atmosphere")
    environment.wind_velocity_x = 10.0
    return environment


def _effective_wind_x(environment):
    """The wind the Environment would actually fly with."""
    wind = environment.wind_velocity_x
    return float(wind(0)) if callable(wind) else float(wind)


def test_a_factor_does_not_compound_across_reseeds():
    """Reseeding with the same seed has to give the same inputs.

    ``StochasticEnvironment.create_object`` writes the sampled value back onto
    the Environment rather than building a copy, so re-reading the nominal from
    it compounded: 10 -> 8.576 -> 7.355 -> 6.308, each the last one multiplied
    by the same factor again.
    """
    stochastic = StochasticEnvironment(
        environment=_windy_environment(), wind_velocity_x_factor=(1.0, 0.1)
    )

    winds = []
    for _ in range(4):
        stochastic._set_stochastic(12345)
        winds.append(_effective_wind_x(stochastic.create_object()))

    assert len(set(winds)) == 1, f"the same seed drifted across reseeds: {winds}"


def test_a_seed_gives_the_same_input_whatever_was_sampled_before_it():
    """Caching the nominal per model, not per seed, is what this pins.

    Reseeding to 103 has to give what it gives on a fresh model, whether or not
    102 and 101 ran first. A cache keyed by the seed would satisfy the test
    above and still fail here, because each new seed would re-read a nominal
    the previous ``create_object`` had already moved.
    """

    def wind_after(seeds):
        stochastic = StochasticEnvironment(
            environment=_windy_environment(), wind_velocity_x_factor=(1.0, 0.1)
        )
        wind = None
        for seed in seeds:
            stochastic._set_stochastic(seed)
            wind = _effective_wind_x(stochastic.create_object())
        return wind

    assert wind_after([101, 102, 103]) == wind_after([103])


def test_a_scalar_nominal_does_not_drift_across_reseeds():
    """Not only the factors.

    ``_validate_scalar`` and the ``(std, "distribution")`` tuple both take their
    nominal from the wrapped object, so a plain scalar spec drifts the same way
    a factor compounds.
    """
    environment = Environment()
    environment.set_atmospheric_model(type="standard_atmosphere")
    stochastic = StochasticEnvironment(environment=environment, elevation=100.0)

    elevations = []
    for _ in range(4):
        stochastic._set_stochastic(2024)
        elevations.append(float(stochastic.create_object().elevation))

    assert len(set(elevations)) == 1, f"the nominal elevation drifted: {elevations}"


def test_the_nominal_is_the_one_the_model_was_built_with(example_plain_env):
    """Snapshot semantics, stated once and pinned here.

    A model samples around what the wrapped object held when it was built, so a
    later change to that object deliberately does not move what is sampled
    around. That is the same rule the drift above depends on.
    """
    example_plain_env.elevation = 1000
    # A scalar is a spread around the object's own value, so this is the form
    # that reads the nominal. A tuple carries its own centre and would not.
    model = StochasticEnvironment(environment=example_plain_env, elevation=5)

    model._set_stochastic(4242)
    around_first = model.elevation[0]

    example_plain_env.elevation = 9000
    model._set_stochastic(4242)

    assert model.elevation[0] == around_first == 1000, (
        "the model followed the object instead of the value it was built with"
    )


def test_a_mutable_nominal_survives_a_write_through_the_object(
    calisto_free_form_fins,
):
    """Holding the object itself would let ``obj.shape_points[:] = ...`` reach
    the nominal and move a value the model is supposed to sample around."""
    stochastic = StochasticFreeFormFins(
        free_form_fins=calisto_free_form_fins, shape_points=0.001
    )
    stochastic._set_stochastic(7)
    expected = np.array(stochastic.shape_points[0], copy=True)

    stochastic.obj.shape_points[:] = [(9.9, 9.9)] * len(stochastic.obj.shape_points)
    stochastic._set_stochastic(7)

    assert np.array_equal(stochastic.shape_points[0], expected)


def test_the_snapshot_keeps_by_reference_what_it_does_not_copy():
    """Only containers are copied, so the rule can be stated as it behaves."""
    array = np.array([1.0, 2.0])
    listed = [[1.0], [2.0]]
    function = Function(lambda x: x)

    assert _snapshot_of(array) is not array
    assert _snapshot_of(listed) is not listed
    assert _snapshot_of(listed)[0] is not listed[0]  # deep, not shallow
    assert _snapshot_of(function) is function
    assert _snapshot_of(3.0) == 3.0
