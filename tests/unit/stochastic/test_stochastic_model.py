from types import SimpleNamespace

import numpy as np
import pytest

from rocketpy import Environment
from rocketpy.mathutils.function import Function
from rocketpy.stochastic import StochasticEnvironment, StochasticFreeFormFins
from rocketpy.stochastic.stochastic_model import StochasticModel, _snapshot_of


def _sampled_option(model):
    """Return the value ``dict_generator`` picks for the ``options`` attribute."""
    return next(model.dict_generator())["options"]


def test_list_attribute_sampling_is_reproducible_under_seed():
    """A list-valued stochastic attribute is drawn through the model's own seeded
    numpy generator, so a fixed seed reproduces the choice. It used to be drawn
    with the stdlib ``random.choice`` (an unseeded global instance), which
    ``random_seed`` could not govern. Heterogeneous entries (paths, callables,
    lists) are returned unchanged rather than coerced to a numpy dtype the way
    ``numpy.random.choice`` would.
    """
    options = ["/motor/a.eng", "/motor/b.eng", (lambda t: t), [1, 2, 3]]
    model = StochasticModel(obj=SimpleNamespace(), options=options)

    model._set_stochastic(42)
    first = _sampled_option(model)
    model._set_stochastic(42)
    assert _sampled_option(model) == first, "same seed must reproduce the choice"
    assert any(first is option for option in options), "object returned unchanged"

    chosen_ids = set()
    for seed in range(16):
        model._set_stochastic(seed)
        chosen_ids.add(id(_sampled_option(model)))
    assert len(chosen_ids) > 1, "different seeds must be able to pick differently"


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


def _effective_wind_x(environment):
    """The wind the Environment would actually fly with."""
    wind = environment.wind_velocity_x
    return float(wind(0)) if callable(wind) else float(wind)


def test_reseeding_does_not_take_the_last_run_as_the_next_nominal():
    """Reseeding with the same seed has to give the same inputs.

    ``StochasticEnvironment.create_object`` writes the randomised value back
    onto the Environment rather than building a copy, so re-reading the nominal
    from it on the next reseed compounded: 10 -> 8.576 -> 7.355 -> 6.308, each
    one the last multiplied by the same factor again.
    """
    environment = Environment()
    environment.set_atmospheric_model(type="standard_atmosphere")
    environment.wind_velocity_x = 10.0
    stochastic = StochasticEnvironment(
        environment=environment, wind_velocity_x_factor=(1.0, 0.1)
    )

    winds = []
    for _ in range(4):
        stochastic._set_stochastic(12345)
        winds.append(_effective_wind_x(stochastic.create_object()))

    assert len(set(winds)) == 1, f"the same seed drifted across reseeds: {winds}"


def test_a_simulation_index_does_not_depend_on_the_indices_before_it():
    """What the per-index seeding claims: index i gets the same inputs however
    it is reached. Running 0, 1, 2 in order has to match running 2 on its own,
    which is what a worker that happens to pick up index 2 first would do.
    """

    def wind_for(seeds):
        environment = Environment()
        environment.set_atmospheric_model(type="standard_atmosphere")
        environment.wind_velocity_x = 10.0
        stochastic = StochasticEnvironment(
            environment=environment, wind_velocity_x_factor=(1.0, 0.1)
        )
        wind = None
        for seed in seeds:
            stochastic._set_stochastic(seed)
            wind = _effective_wind_x(stochastic.create_object())
        return wind

    assert wind_for([101, 102, 103]) == wind_for([103])


def test_a_scalar_nominal_does_not_drift_across_reseeds():
    """Not only the factors. ``_validate_scalar`` and the ``(std, "distribution")``
    tuple both take their nominal from the object, and ``create_object`` writes
    the drawn value back onto that same object, so a plain scalar spec drifts
    the same way a factor compounds.
    """
    environment = Environment()
    environment.set_atmospheric_model(type="standard_atmosphere")
    stochastic = StochasticEnvironment(environment=environment, elevation=100.0)

    elevations = []
    for _ in range(4):
        stochastic._set_stochastic(2024)
        elevations.append(float(stochastic.create_object().elevation))

    assert len(set(elevations)) == 1, f"the nominal elevation drifted: {elevations}"


def test_a_custom_sampler_answers_to_the_seed_it_is_given(elevation_sampler):
    """The 128-bit int this package hands a sampler has to reach its draws.

    The fixture used to build a generator in ``reset_seed`` and drop it, while
    ``sample`` drew from the process-global ``np.random``, so nothing in it
    answered to a seed and the guarantee went untested.
    """
    wide = 271828182845904523536028747135266249775

    elevation_sampler.reset_seed(wide)
    first = elevation_sampler.sample(5)
    elevation_sampler.reset_seed(wide)
    again = elevation_sampler.sample(5)

    assert first == again, "the same seed gave a different sample"

    elevation_sampler.reset_seed(wide + 1)
    other = elevation_sampler.sample(5)

    assert other != first, "a different seed gave the same sample"


def test_a_custom_sampler_is_not_moved_by_the_global_generator(elevation_sampler):
    """The control for the test above. Drawing from the global stream in
    between must not change what the seeded sampler produces, or the sampler is
    still reading from somewhere this package does not seed."""
    seed = 12345678901234567890123456789012345678

    elevation_sampler.reset_seed(seed)
    expected = elevation_sampler.sample(5)

    elevation_sampler.reset_seed(seed)
    np.random.random(100)
    assert elevation_sampler.sample(5) == expected


def test_the_nominal_is_the_one_the_model_was_built_with(example_plain_env):
    """Snapshot semantics, stated once and pinned here.

    A model samples around what the wrapped object held when it was built.
    This exists because ``StochasticEnvironment.create_object`` writes the
    sampled value back onto that object on purpose, and reading the nominal
    back off it made a factor compound from one simulation to the next. The
    rule is the same for every model, so a change to the wrapped object after
    construction deliberately does not move what is sampled around.
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


def test_a_mutable_nominal_survives_a_write_through_the_object(calisto_free_form_fins):
    """Writing through the wrapped object must not move the nominal.

    The cache held the object itself, so ``obj.outline[:] = ...`` reached it and
    a value the model is supposed to sample around moved underneath it.
    """
    stochastic = StochasticFreeFormFins(
        free_form_fins=calisto_free_form_fins, shape_points=0.001
    )
    stochastic._set_stochastic(7)
    expected = list(stochastic._nominal("shape_points", getattr))

    stochastic.obj.shape_points[:] = [(9.9, 9.9)] * len(stochastic.obj.shape_points)

    assert list(stochastic._nominal("shape_points", getattr)) == expected


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
