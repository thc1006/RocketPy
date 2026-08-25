import pytest

from rocketpy.simulation.monte_carlo import _join_the_workers


class _Worker:
    """A process that stops after a set number of polls, or never.

    ``never`` stands in for one blocked on a lock its dead sibling was holding,
    which is the case an unbounded join waits out forever.
    """

    def __init__(self, exitcode=0, alive_for=0, never=False, ignores_terminate=False):
        self.exitcode = None
        self._final_exitcode = exitcode
        self._alive_for = alive_for
        self._never = never
        self._ignores_terminate = ignores_terminate
        self.joins = 0
        self.timeouts = []
        self.terminated = False
        self.killed = False

    def is_alive(self):
        return self.exitcode is None

    def join(self, timeout=None):
        self.joins += 1
        self.timeouts.append(timeout)
        # A real worker that never returns makes the caller hang, which is the
        # bug. Reproducing that here would hang CI instead of reporting, so the
        # stand-in gives up and says so.
        assert self.joins < 200, "the join loop never stopped waiting"
        if self._never or self.joins <= self._alive_for:
            return
        self.exitcode = self._final_exitcode

    def terminate(self):
        self.terminated = True
        if not self._ignores_terminate:
            self.exitcode = -15

    def kill(self):
        self.killed = True
        self.exitcode = -9


class _Event:
    def __init__(self, already_set=False):
        self.was_set = already_set

    def is_set(self):
        return self.was_set

    def set(self):
        self.was_set = True


def test_a_run_where_every_worker_finishes_is_left_alone():
    """A healthy fleet is joined to completion and never terminated."""
    workers = [_Worker(alive_for=3), _Worker(alive_for=5)]

    _join_the_workers(workers, _Event(), grace_period=0)

    assert [worker.exitcode for worker in workers] == [0, 0]
    assert not any(worker.terminated for worker in workers)


def test_a_worker_blocked_behind_a_dead_one_does_not_wait_forever():
    """One bad exit ends the wait for a sibling that never returns."""
    # The one that mattered. Without a bound this call never returns, so the
    # parent never reaches the check that would have reported the failure.
    died = _Worker(exitcode=-9, alive_for=1)
    blocked = _Worker(never=True)

    _join_the_workers([died, blocked], _Event(), grace_period=0)

    assert blocked.terminated


def test_the_survivors_are_asked_before_they_are_ended():
    """The event is set before anything is terminated."""
    died = _Worker(exitcode=1, alive_for=1)
    blocked = _Worker(never=True)
    event = _Event()

    _join_the_workers([died, blocked], event, grace_period=0)

    assert event.was_set


def test_a_survivor_that_stops_on_its_own_is_not_terminated():
    """A worker that leaves during the grace period is left alone."""
    died = _Worker(exitcode=1, alive_for=1)
    cooperative = _Worker(alive_for=2)

    _join_the_workers([died, cooperative], _Event(), grace_period=0)

    assert not cooperative.terminated
    assert cooperative.exitcode == 0


@pytest.mark.parametrize("exitcode", [-9, 1, 2])
def test_any_bad_exit_starts_the_shutdown(exitcode):
    """Signals and non-zero codes both start the shutdown."""
    died = _Worker(exitcode=exitcode, alive_for=1)
    blocked = _Worker(never=True)

    _join_the_workers([died, blocked], _Event(), grace_period=0)

    assert blocked.terminated


def test_a_slow_run_is_never_bounded():
    """Elapsed time is not evidence: a slow fleet is polled, never stopped."""
    # Nothing here may act on how long a worker takes, only on it having died.
    slow = _Worker(alive_for=50)
    slower = _Worker(alive_for=80)

    _join_the_workers([slow, slower], _Event(), grace_period=0)

    assert not any(worker.terminated for worker in (slow, slower))
    assert slower.joins > 50


def test_a_reported_failure_leaves_a_working_sibling_alone():
    """A worker that only reported gives its siblings no reason to be ended."""
    # Measured: with the event treated as a reason to stop, this sibling was
    # terminated after three polls of the forty it needed. It was not stuck,
    # it was mid-simulation, and it would have seen the event and left with
    # its rows intact.
    reported = _Worker(exitcode=0, alive_for=1)
    working = _Worker(alive_for=40)

    _join_the_workers([reported, working], _Event(already_set=True), grace_period=0)

    assert not working.terminated
    assert working.exitcode == 0


def test_a_clean_run_is_not_stopped_by_an_event_nobody_set():
    """An unset event leaves a healthy run running."""
    first, second = _Worker(alive_for=2), _Worker(alive_for=3)

    _join_the_workers([first, second], _Event(), grace_period=0)

    assert not any(worker.terminated for worker in (first, second))


def test_a_worker_that_ignores_terminate_is_killed():
    """Terminate can be ignored; the fleet still has to come down."""
    died = _Worker(exitcode=-9, alive_for=1)
    stubborn = _Worker(never=True, ignores_terminate=True)

    _join_the_workers([died, stubborn], _Event(), grace_period=0)

    assert stubborn.terminated
    assert stubborn.killed


def test_the_fleet_comes_down_on_one_deadline_not_one_each():
    """A stage gives the fleet one grace period between them, not each."""
    # Observed through what each worker is offered: with a deadline of its own
    # every worker is given the whole grace, so a fleet of thirty takes thirty
    # times as long to give up on.
    died = _Worker(exitcode=-9, alive_for=1)
    stuck = [_Worker(never=True) for _ in range(4)]

    _join_the_workers([died, *stuck], _Event(), grace_period=0.05)

    offered = [t for t in stuck[-1].timeouts if t is not None]
    assert offered
    assert min(offered) < 0.05
