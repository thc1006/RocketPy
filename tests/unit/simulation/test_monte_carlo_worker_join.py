import pytest

from rocketpy.simulation.monte_carlo import _join_the_workers


class _Worker:
    """A process that stops after a set number of polls, or never.

    ``never`` stands in for one blocked on a lock its dead sibling was holding,
    which is the case an unbounded join waits out forever.
    """

    def __init__(self, exitcode=0, alive_for=0, never=False):
        self.exitcode = None
        self._final_exitcode = exitcode
        self._alive_for = alive_for
        self._never = never
        self.joins = 0
        self.terminated = False

    def is_alive(self):
        return self.exitcode is None

    def join(self, timeout=None):  # pylint: disable=unused-argument
        self.joins += 1
        # A real worker that never returns makes the caller hang, which is the
        # bug. Reproducing that here would hang CI instead of reporting, so the
        # stand-in gives up and says so.
        assert self.joins < 200, "the join loop never stopped waiting"
        if self._never or self.joins <= self._alive_for:
            return
        self.exitcode = self._final_exitcode

    def terminate(self):
        self.terminated = True
        self.exitcode = -15


class _Event:
    def __init__(self, already_set=False):
        self.was_set = already_set

    def is_set(self):
        return self.was_set

    def set(self):
        self.was_set = True


class _BrokenEvent(_Event):
    """A manager proxy that has gone away."""

    def is_set(self):
        raise OSError("the manager is gone")


def test_a_run_where_every_worker_finishes_is_left_alone():
    workers = [_Worker(alive_for=3), _Worker(alive_for=5)]

    _join_the_workers(workers, _Event(), grace_period=0)

    assert [worker.exitcode for worker in workers] == [0, 0]
    assert not any(worker.terminated for worker in workers)


def test_a_worker_blocked_behind_a_dead_one_does_not_wait_forever():
    # The one that mattered. Without a bound this call never returns, so the
    # parent never reaches the check that would have reported the failure.
    died = _Worker(exitcode=-9, alive_for=1)
    blocked = _Worker(never=True)

    _join_the_workers([died, blocked], _Event(), grace_period=0)

    assert blocked.terminated


def test_the_survivors_are_asked_before_they_are_ended():
    died = _Worker(exitcode=1, alive_for=1)
    blocked = _Worker(never=True)
    event = _Event()

    _join_the_workers([died, blocked], event, grace_period=0)

    assert event.was_set


def test_a_survivor_that_stops_on_its_own_is_not_terminated():
    died = _Worker(exitcode=1, alive_for=1)
    cooperative = _Worker(alive_for=2)

    _join_the_workers([died, cooperative], _Event(), grace_period=0)

    assert not cooperative.terminated
    assert cooperative.exitcode == 0


@pytest.mark.parametrize("exitcode", [-9, 1, 2])
def test_any_bad_exit_starts_the_shutdown(exitcode):
    died = _Worker(exitcode=exitcode, alive_for=1)
    blocked = _Worker(never=True)

    _join_the_workers([died, blocked], _Event(), grace_period=0)

    assert blocked.terminated


def test_a_slow_run_is_never_bounded():
    # Nothing here may act on how long a worker takes, only on it having died.
    slow = _Worker(alive_for=50)
    slower = _Worker(alive_for=80)

    _join_the_workers([slow, slower], _Event(), grace_period=0)

    assert not any(worker.terminated for worker in (slow, slower))
    assert slower.joins > 50


def test_a_reported_failure_also_stops_a_sibling_that_never_returns():
    # A worker that fails the ordinary way is caught by the producer, reports
    # through the event and returns, so it exits cleanly. Waiting only on exit
    # codes leaves the parent sitting behind whichever sibling is stuck.
    reported = _Worker(exitcode=0, alive_for=1)
    stuck = _Worker(never=True)

    _join_the_workers([reported, stuck], _Event(already_set=True), grace_period=0)

    assert stuck.terminated


def test_a_clean_run_is_not_stopped_by_an_event_nobody_set():
    first, second = _Worker(alive_for=2), _Worker(alive_for=3)

    _join_the_workers([first, second], _Event(), grace_period=0)

    assert not any(worker.terminated for worker in (first, second))


def test_an_event_that_cannot_be_read_does_not_stop_the_run():
    # The control on the control. If asking the manager raises, that is not
    # evidence of failure and must not end a healthy run.
    first, second = _Worker(alive_for=2), _Worker(alive_for=3)

    _join_the_workers([first, second], _BrokenEvent(), grace_period=0)

    assert not any(worker.terminated for worker in (first, second))
