from types import SimpleNamespace

import pytest

from rocketpy.stochastic import StochasticFreeFormFins
from rocketpy.stochastic.stochastic_model import StochasticModel


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
