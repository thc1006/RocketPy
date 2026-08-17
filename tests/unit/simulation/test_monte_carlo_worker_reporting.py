import os
from types import SimpleNamespace

import pytest

from rocketpy.simulation.monte_carlo import MonteCarlo


class _Mutex:
    def acquire(self):
        pass

    def release(self):
        pass


class _ErrorEvent:
    def __init__(self):
        self.was_set = False

    def is_set(self):
        return self.was_set

    def set(self):
        self.was_set = True


def _a_worker(tmp_path, model):
    study = MonteCarlo(
        filename=os.path.join(str(tmp_path), "study"),
        environment=model,
        rocket=model,
        flight=model,
    )
    return study, _ErrorEvent()


def _run(study, monitor, error_event, seed=42):
    # Name-mangled: the producer is what each worker process runs, and nothing
    # else in the suite calls it.
    study._MonteCarlo__sim_producer(seed, monitor, _Mutex(), error_event)


def test_a_worker_that_fails_before_seeding_finishes_says_so(tmp_path, capsys):
    def refuse(_seed):
        raise RuntimeError("the models would not reseed")

    model = SimpleNamespace(last_rnd_dict={}, _set_stochastic=refuse)
    monitor = SimpleNamespace(keep_simulating=lambda: True)
    study, error_event = _a_worker(tmp_path, model)

    _run(study, monitor, error_event)

    assert error_event.was_set
    reported = capsys.readouterr().out
    assert "worker startup" in reported
    assert "the models would not reseed" in reported


def test_a_worker_that_fails_before_claiming_an_index_says_so(tmp_path, capsys):
    def refuse():
        raise RuntimeError("the monitor would not hand out an index")

    model = SimpleNamespace(last_rnd_dict={}, _set_stochastic=lambda _seed: None)
    monitor = SimpleNamespace(keep_simulating=lambda: True, increment=refuse)
    study, error_event = _a_worker(tmp_path, model)

    _run(study, monitor, error_event)

    assert error_event.was_set
    assert "worker startup" in capsys.readouterr().out


def test_a_worker_that_fails_inside_a_simulation_names_the_index(
    tmp_path, capsys, monkeypatch
):
    # The control. An index is claimed and the simulation then fails, which is
    # the path that already worked, so the report still has to name it.
    def refuse(_self):
        raise RuntimeError("the simulation would not run")

    monkeypatch.setattr(
        MonteCarlo, "_MonteCarlo__run_single_simulation", refuse, raising=True
    )
    model = SimpleNamespace(last_rnd_dict={}, _set_stochastic=lambda _seed: None)
    monitor = SimpleNamespace(keep_simulating=lambda: True, increment=lambda: 8)
    study, error_event = _a_worker(tmp_path, model)

    _run(study, monitor, error_event)

    assert error_event.was_set
    assert "iteration 7" in capsys.readouterr().out


def test_the_error_file_is_left_alone_when_nothing_was_drawn(tmp_path):
    def refuse(_seed):
        raise RuntimeError("the models would not reseed")

    model = SimpleNamespace(last_rnd_dict={}, _set_stochastic=refuse)
    monitor = SimpleNamespace(keep_simulating=lambda: True)
    study, error_event = _a_worker(tmp_path, model)

    _run(study, monitor, error_event)

    with open(study.error_file, "r", encoding="utf-8") as recorded:
        assert recorded.read() == ""


@pytest.mark.parametrize("failing", ["_set_stochastic", "increment"])
def test_a_worker_failure_never_raises_out_of_the_producer(tmp_path, failing):
    # The handler used to reach for names the loop had not bound yet, so the
    # process died with UnboundLocalError and the parent waited forever.
    def refuse(*_args):
        raise RuntimeError("boom")

    model = SimpleNamespace(
        last_rnd_dict={},
        _set_stochastic=refuse if failing == "_set_stochastic" else lambda _s: None,
    )
    monitor = SimpleNamespace(
        keep_simulating=lambda: True,
        increment=refuse if failing == "increment" else (lambda: 1),
    )
    study, error_event = _a_worker(tmp_path, model)

    _run(study, monitor, error_event)

    assert error_event.was_set
