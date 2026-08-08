"""The parent has to notice a worker that died without saying so.

``join()`` returns None however the child ended, so the shared error event was
the only signal the run had. A worker can leave without setting it: ``SystemExit``,
``os._exit``, a segfault in a native extension, a target that will not unpickle
under spawn, or the error handler of the worker itself failing. The exit status
is what separates those from a clean finish.
"""

import traceback
import types
from contextlib import contextmanager
from time import monotonic, sleep

import pytest

import rocketpy.simulation.monte_carlo as mc


class _Process:
    """A worker that does not run, and reports the exit code it was given."""

    instances = []

    def __init__(self, target=None, args=(), **_kwargs):  # pylint: disable=unused-argument
        self.name = f"worker-{len(self.instances)}"
        self.exitcode = None
        self.started = False
        self.terminated = False
        self._planned_exitcode = 0
        self.instances.append(self)

    def start(self):
        self.started = True

    def join(self, *_a, **_k):
        self.exitcode = self._planned_exitcode

    def is_alive(self):
        return False

    def terminate(self):
        self.terminated = True


class _Event:
    def __init__(self):
        self.flag = False

    def set(self):
        self.flag = True

    def is_set(self):
        return self.flag


class _Monitor:
    def __init__(self, **_kwargs):
        pass

    def print_final_status(self):
        pass


@pytest.fixture
def parallel_runner(monkeypatch, tmp_path):
    """Run ``__run_in_parallel`` over stub workers and hand back the stubs."""
    _Process.instances = []
    fake_multiprocess = types.SimpleNamespace(Process=_Process)

    class _Manager:  # pylint: disable=invalid-name
        """Method names mirror the multiprocess manager API."""

        def Lock(self):  # noqa: N802
            return types.SimpleNamespace(acquire=lambda: None, release=lambda: None)

        def Event(self):  # noqa: N802
            return _Event()

        def _SimMonitor(self, **kwargs):  # noqa: N802
            return _Monitor(**kwargs)

    @contextmanager
    def fake_manager(*_a, **_k):
        yield _Manager()

    monkeypatch.setattr(mc, "_import_multiprocess", lambda: (fake_multiprocess, None))
    monkeypatch.setattr(mc, "_create_multiprocess_manager", fake_manager)

    runner = types.SimpleNamespace(
        error_file=tmp_path / "errors.txt",
        input_file=tmp_path / "inputs.txt",
        output_file=tmp_path / "outputs.txt",
        _initial_sim_idx=0,
        number_of_simulations=4,
        _interrupted=False,
        _MonteCarlo__validate_number_of_workers=lambda n: 2,
        _MonteCarlo__sim_producer=lambda *a: None,
    )
    runner.input_file.write_text("")
    runner.output_file.write_text("")
    return runner


def test_a_worker_that_crashes_without_setting_the_event_fails_the_run(
    parallel_runner,
):
    """The case the event alone cannot see."""
    original_join = _Process.join

    def crash(self, *a, **k):
        original_join(self, *a, **k)
        self.exitcode = -11  # SIGSEGV

    _Process.join = crash
    try:
        with pytest.raises(RuntimeError, match="did not exit cleanly"):
            mc.MonteCarlo._MonteCarlo__run_in_parallel(parallel_runner, n_workers=2)
    finally:
        _Process.join = original_join


def test_a_clean_run_is_not_reported_as_a_crash(parallel_runner):
    """The other half: every worker exits 0, so nothing is raised."""
    mc.MonteCarlo._MonteCarlo__run_in_parallel(parallel_runner, n_workers=2)

    assert all(p.exitcode == 0 for p in _Process.instances)


def test_a_failed_start_still_cleans_up_the_workers_already_running(
    parallel_runner, monkeypatch
):
    """The start loop is inside the try for this. A ``start()`` that fails part
    way through used to leave the ones already running with nobody to reap
    them."""
    started = []
    original_start = _Process.start

    def start_then_fail(self):
        if len(started) >= 1:
            raise OSError("cannot allocate a process")
        original_start(self)
        started.append(self)

    monkeypatch.setattr(_Process, "start", start_then_fail)
    monkeypatch.setattr(_Process, "is_alive", lambda self: not self.terminated)

    with pytest.raises(OSError):
        mc.MonteCarlo._MonteCarlo__run_in_parallel(parallel_runner, n_workers=3)

    assert started, "the fixture never started anything"
    assert all(p.terminated for p in started), "a started worker was left running"


def test_a_worker_still_writing_gets_a_window_before_it_is_signalled(
    parallel_runner, monkeypatch
):
    """The event asks the fleet to stop, it does not stop it.

    Cutting a worker off the moment another one reports an error truncates
    whatever row it was part way through, which is the corruption the
    completeness check then reports. Measured before the fix: the main error
    path gave a running worker 0.0 ms, while the interrupt path gave it the
    full grace period.
    """
    grace = 0.3
    monkeypatch.setattr(mc, "_WORKER_SHUTDOWN_GRACE", grace)
    signalled = []
    original_terminate = _Process.terminate

    def note_when(self):
        signalled.append(monotonic())
        self._alive = False
        original_terminate(self)

    monkeypatch.setattr(_Process, "start", lambda self: setattr(self, "_alive", True))
    monkeypatch.setattr(
        _Process, "is_alive", lambda self: getattr(self, "_alive", False)
    )
    monkeypatch.setattr(_Process, "terminate", note_when)

    # Another worker has already reported an error while this one is going.
    class _AlreadyFailed(_Event):
        def __init__(self):
            super().__init__()
            self.flag = True

    class _FailedManager:  # pylint: disable=invalid-name
        def Lock(self):  # noqa: N802
            return types.SimpleNamespace(acquire=lambda: None, release=lambda: None)

        def Event(self):  # noqa: N802
            return _AlreadyFailed()

        def _SimMonitor(self, **kwargs):  # noqa: N802
            return _Monitor(**kwargs)

    @contextmanager
    def failed_manager(*_a, **_k):
        yield _FailedManager()

    monkeypatch.setattr(mc, "_create_multiprocess_manager", failed_manager)

    began = monotonic()
    with pytest.raises(RuntimeError):
        mc.MonteCarlo._MonteCarlo__run_in_parallel(parallel_runner, n_workers=2)

    assert signalled, "the worker was never signalled at all"
    assert signalled[0] - began >= grace, (
        f"the worker was cut off after {(signalled[0] - began) * 1000:.1f} ms, "
        f"before the {grace * 1000:.0f} ms window it is meant to get"
    )


def test_an_interrupted_run_is_not_then_reported_as_incomplete(
    parallel_runner, monkeypatch
):
    """Ctrl-C in the parent is caught and deliberately not re-raised, so
    ``simulate`` carries on to the completeness check with the run unfinished.
    The two are composed here in that order, since the check has to be able to
    tell a run the user stopped from a worker that went missing.
    """
    interrupted = []
    original_join = _Process.join

    def ctrl_c(self, *a, **k):
        # Once: the handler joins again on its way out, and that has to work.
        if not interrupted:
            interrupted.append(True)
            raise KeyboardInterrupt("user pressed ctrl-c")
        original_join(self, *a, **k)

    monkeypatch.setattr(_Process, "join", ctrl_c)

    mc.MonteCarlo._MonteCarlo__run_in_parallel(parallel_runner, n_workers=2)
    mc.MonteCarlo._MonteCarlo__check_each_index_was_recorded_once(parallel_runner)

    assert interrupted, "the run was never interrupted"
    assert parallel_runner.input_file.read_text() == "", (
        "nothing was written, so the check really was in a position to reject this"
    )


class _Crashed:
    """Killed outright: gone, non-zero exit code, no event set."""

    def __init__(self):
        self.exitcode = 7

    def join(self, *_a, **_k):
        pass

    def is_alive(self):
        return False


class _BlockedForever:
    """Waiting on the lock the crashed worker still holds."""

    def __init__(self, give_up_after=50):
        self.exitcode = None
        self.joins = 0
        self._give_up_after = give_up_after

    def join(self, *_a, **_k):
        self.joins += 1
        if self.joins > self._give_up_after:
            raise AssertionError(
                "the parent is still waiting on a worker that a dead sibling "
                "has blocked, and nothing else will end this wait"
            )

    def is_alive(self):
        return True


def test_the_parent_stops_waiting_when_a_worker_dies_holding_the_lock():
    """A worker killed outright sets no event, so the wait had only
    ``is_alive`` to end it, and a sibling blocked on the lock it held kept that
    true forever. The exit-code check downstream was never reached."""
    crashed, blocked = _Crashed(), _BlockedForever()

    mc._wait_for_workers([crashed, blocked], _Event())

    assert blocked.is_alive(), "the blocked worker is meant to still be running"


def test_the_shutdown_window_is_not_cut_short_by_a_worker_already_known_dead():
    """The grace period exists so the survivors can finish their writes. It is
    bounded by its own timeout, so a crash must not end it early.

    On elapsed time rather than on how many times a worker was joined: that
    counted the old per-worker blocking join, which is the mechanism and not the
    behaviour. The wait can only overshoot the deadline, never undershoot it, so
    a floor well under the timeout is safe on a coarse clock.
    """
    crashed, blocked = _Crashed(), _BlockedForever(give_up_after=10**6)

    started = monotonic()
    mc._wait_for_workers([crashed, blocked], timeout=0.3)
    elapsed = monotonic() - started

    assert elapsed >= 0.2, f"the grace period returned after {elapsed:.3f}s"


class _Stubborn:
    """A worker that sits through terminate and kill, recording its waits."""

    def __init__(self, waits):
        self.exitcode = None
        self._waits = waits

    def join(self, timeout=None, **_k):
        self._waits.append(timeout)
        if timeout:
            sleep(timeout)

    def is_alive(self):
        return True

    def terminate(self):
        pass

    def kill(self):
        pass


def _granted_shutting_down(fleet_size, grace):
    """How long the fleet was granted in total, across both phases."""
    waits = []
    mc._stop_any_worker_still_running(
        [_Stubborn(waits) for _ in range(fleet_size)], grace=grace
    )
    assert len(waits) == 2 * fleet_size, "every worker is still waited on"
    return sum(waits)


def test_shutdown_does_not_grow_with_the_fleet():
    """Each worker used to get the full grace to itself, so the wait scaled with
    the fleet: six stubborn workers held the parent for six grace periods per
    phase rather than one.

    Compares two fleet sizes rather than checking either against the grace.
    A single fleet has to be measured against a constant, and the slack that
    needs is the machine's timer granularity, which on Windows is 15 ms against
    a 200 ms grace. Both measurements carry the same granularity, so comparing
    them cancels it.
    """
    grace = 0.2

    alone = _granted_shutting_down(1, grace)
    crowd = _granted_shutting_down(6, grace)

    assert crowd < alone * 2, (
        f"six workers were granted {crowd:.2f}s against {alone:.2f}s for one, "
        f"so the wait is still scaling with the fleet"
    )


def test_the_parent_does_not_repeat_its_own_frame_in_the_traceback(
    parallel_runner, monkeypatch
):
    """``raise error`` names the exception again, so the handler's line joins the
    traceback and the reader walks past it. The serial path already re-raises
    bare; this is the parent doing the same.

    On the duplicate rather than the original frame: the failing call survives
    either spelling, so asserting it is there passes both.
    """

    def start_then_fail(self):
        raise OSError("cannot start")

    monkeypatch.setattr(_Process, "start", start_then_fail)

    with pytest.raises(OSError) as raised:
        mc.MonteCarlo._MonteCarlo__run_in_parallel(parallel_runner, n_workers=2)

    frames = [frame.name for frame in traceback.extract_tb(raised.value.__traceback__)]
    assert "start_then_fail" in frames, frames
    assert frames.count("__run_in_parallel") == 1, frames


class _NeverFinishes:
    """Alive throughout, and costly to join, as a real blocked worker is."""

    def __init__(self):
        self.exitcode = None

    def join(self, timeout=None, **_k):
        if timeout:
            sleep(timeout)

    def is_alive(self):
        return True


class _SetMidRound:
    """Not set when a round begins, set while the parent is inside it."""

    def __init__(self):
        self.checks = 0

    def is_set(self):
        self.checks += 1
        return self.checks > 1


@pytest.mark.parametrize("fleet", [1, 24], ids=["one", "twenty_four"])
def test_noticing_an_error_does_not_get_slower_with_a_bigger_fleet(fleet):
    """Joining every worker for 0.1s before rechecking the event made the delay
    the fleet size times that: 24 workers took 2.4s to notice. One sleep per
    round instead, so the cost is the round and not the fleet.

    Both sizes are measured against the same ceiling rather than against each
    other, because the point is that neither depends on the count.
    """
    started = monotonic()
    mc._wait_for_workers([_NeverFinishes() for _ in range(fleet)], _SetMidRound())
    elapsed = monotonic() - started

    assert elapsed < 0.5, f"{fleet} workers delayed the check by {elapsed:.2f}s"


class _OriginalFailure(RuntimeError):
    """What the caller should end up seeing."""


class _CleanupFailure(RuntimeError):
    """What must not take its place."""


def test_the_final_cleanup_does_not_replace_the_failure_on_its_way_out(
    parallel_runner, monkeypatch
):
    """The handler's own cleanup is best effort, but ``finally`` runs again on
    the way out and was calling the same thing raw. A failure there landed on
    the caller instead of the one being re-raised."""
    monkeypatch.setattr(
        mc, "_start_the_fleet", _raiser(_OriginalFailure, "cannot start")
    )
    monkeypatch.setattr(
        mc,
        "_stop_any_worker_still_running",
        _raiser(_CleanupFailure, "cannot clean up"),
    )

    with pytest.warns(RuntimeWarning, match="final worker shutdown"):
        with pytest.raises(_OriginalFailure, match="cannot start"):
            mc.MonteCarlo._MonteCarlo__run_in_parallel(parallel_runner, n_workers=2)


def _raiser(exception, message):
    def raise_it(*_args, **_kwargs):
        raise exception(message)

    return raise_it
