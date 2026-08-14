"""What a worker does when something fails part way through.

A worker that dies has to leave three things true: the failure that started it
is the one that escapes, the shared mutex is not still held, and the error event
is set. Before this, a failure in the claim itself broke all three at once --
``sim_idx`` and ``inputs_json`` were only bound inside the loop, so the handler
raised ``UnboundLocalError`` over the real error while holding the mutex, and
``error_event.set()`` sat after the write that never ran.
"""

import json
import threading
import traceback
import types
import warnings

import pytest

import rocketpy.simulation.monte_carlo as mc


class _RecordingMutex:
    """A real lock that counts acquires and releases."""

    def __init__(self):
        self.acquired = 0
        self.released = 0
        self._lock = threading.Lock()

    def acquire(self, *args, **kwargs):
        self.acquired += 1
        return self._lock.acquire(*args, **kwargs)

    def release(self):
        self.released += 1
        return self._lock.release()


class _Event:
    def __init__(self):
        self.flag = False

    def set(self):
        self.flag = True

    def is_set(self):
        return self.flag


class _Boom(RuntimeError):
    """The failure under test, so it cannot be confused with an incidental one."""


def _worker(tmp_path, **overrides):
    """A stand-in carrying only the attributes ``__sim_producer`` touches."""
    attributes = {
        "error_file": tmp_path / "errors.txt",
        "input_file": tmp_path / "inputs.txt",
        "output_file": tmp_path / "outputs.txt",
        # The failure path recovers whatever was drawn from these, so the
        # stand-in has to carry them the way a MonteCarlo does.
        "_export_config": {},
        "_MonteCarlo__draws_before_this_simulation": ({}, {}, {}),
        "environment": types.SimpleNamespace(last_rnd_dict={}),
        "rocket": types.SimpleNamespace(last_rnd_dict={}),
        "flight": types.SimpleNamespace(last_rnd_dict={}),
        "_MonteCarlo__child_seed": lambda index: index,
        "_MonteCarlo__seed_simulation": lambda seed: None,
        "_MonteCarlo__run_single_simulation": object,
        "_MonteCarlo__evaluate_flight_inputs": lambda index: "{}\n",
        "_MonteCarlo__evaluate_flight_outputs": lambda flight, index: "{}\n",
    }
    attributes.update(overrides)
    return types.SimpleNamespace(**attributes)


def _raise(*_args, **_kwargs):
    raise _Boom("injected")


@pytest.mark.parametrize(
    "stage",
    ["claim", "reseed", "flight", "inputs", "outputs"],
    ids=["claim", "reseed", "flight", "input_eval", "output_eval"],
)
def test_a_failure_anywhere_keeps_the_cause_and_frees_the_mutex(
    tmp_path, monkeypatch, stage
):
    """Whichever stage fails, the same three things have to hold."""
    indices = iter([0, None])
    monkeypatch.setattr(mc, "_claim_next_index", lambda *a, **k: next(indices))
    overrides = {}
    if stage == "claim":
        monkeypatch.setattr(mc, "_claim_next_index", _raise)
    elif stage == "reseed":
        overrides["_MonteCarlo__seed_simulation"] = _raise
    elif stage == "flight":
        overrides["_MonteCarlo__run_single_simulation"] = _raise
    elif stage == "inputs":
        overrides["_MonteCarlo__evaluate_flight_inputs"] = _raise
    elif stage == "outputs":
        overrides["_MonteCarlo__evaluate_flight_outputs"] = _raise

    mutex, event = _RecordingMutex(), _Event()

    with pytest.raises(_Boom):
        mc.MonteCarlo._MonteCarlo__sim_producer(
            _worker(tmp_path, **overrides), object(), mutex, event
        )

    assert mutex.acquired == mutex.released, "the mutex was left held"
    assert event.flag, "the parent was never told an error happened"


def test_the_cause_survives_even_when_the_error_report_also_fails(
    tmp_path, monkeypatch
):
    """Reporting is best effort. If the error file is unwritable too, the
    failure that started it is still what comes out, and the mutex is still
    released."""
    monkeypatch.setattr(mc, "_claim_next_index", lambda *a, **k: 0)
    monkeypatch.setattr(
        "builtins.open", lambda *a, **k: (_ for _ in ()).throw(OSError("no disk"))
    )
    worker = _worker(tmp_path, _MonteCarlo__run_single_simulation=_raise)
    mutex, event = _RecordingMutex(), _Event()

    with pytest.raises(_Boom):
        mc.MonteCarlo._MonteCarlo__sim_producer(worker, object(), mutex, event)

    assert mutex.acquired == mutex.released, "the mutex was left held"
    assert event.flag


def test_the_failure_is_still_reported_when_the_claim_itself_failed(
    tmp_path, monkeypatch
):
    """The report has to survive a failure before the loop body ran.

    ``sim_idx`` and ``inputs_json`` are bound before the try for this reason.
    Left to the loop, the handler raised ``UnboundLocalError`` at its first
    write, so nothing was written and nothing was printed: the run ended with
    no record of what went wrong.
    """
    monkeypatch.setattr(mc, "_claim_next_index", _raise)
    reported = []
    monkeypatch.setattr(mc._SimMonitor, "reprint", staticmethod(reported.append))
    mutex, event = _RecordingMutex(), _Event()

    with pytest.raises(_Boom):
        mc.MonteCarlo._MonteCarlo__sim_producer(
            _worker(tmp_path), object(), mutex, event
        )

    assert reported, "the worker died without reporting anything"
    assert "injected" in reported[0], f"the report does not name the cause: {reported}"


def test_an_interrupt_while_reporting_does_not_leave_the_mutex_held(
    tmp_path, monkeypatch
):
    """``except Exception`` does not catch ``KeyboardInterrupt``, so the release
    has to be in a ``finally``. Ctrl-C between the acquire and the release would
    otherwise leave every other worker blocked on it for good."""
    monkeypatch.setattr(mc, "_claim_next_index", lambda *a, **k: 0)

    def interrupt(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(mc._SimMonitor, "reprint", staticmethod(interrupt))
    worker = _worker(tmp_path, _MonteCarlo__run_single_simulation=_raise)
    mutex, event = _RecordingMutex(), _Event()

    with pytest.raises(KeyboardInterrupt):
        mc.MonteCarlo._MonteCarlo__sim_producer(worker, object(), mutex, event)

    assert mutex.acquired == mutex.released, "the mutex was left held on interrupt"


def test_a_failure_before_the_inputs_exist_still_leaves_a_readable_record(
    tmp_path, monkeypatch
):
    """The run tells the user to check the error file, so it has to say
    something. A failure in the claim has no inputs to write, and the file was
    left empty while the traceback went only to a worker's stdout, which under
    ``spawn`` on Windows the user may never see.

    It has to stay a JSON line: ``_read_log_file`` parses this file with
    ``json.loads`` per line, so free text would make the whole log unreadable.
    """
    monkeypatch.setattr(mc, "_claim_next_index", _raise)
    worker = _worker(tmp_path)
    mutex, event = _RecordingMutex(), _Event()

    with pytest.raises(_Boom):
        mc.MonteCarlo._MonteCarlo__sim_producer(worker, object(), mutex, event)

    lines = [
        line
        for line in worker.error_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert lines, "the error file was left empty"
    record = json.loads(lines[0])
    assert record["index"] is None, "an early failure has no simulation index"
    assert "injected" in record["error"], "the record does not carry the cause"


@pytest.mark.parametrize("failing_file", ["input_file", "output_file"])
def test_a_failed_write_is_reported_like_any_other_failure(
    tmp_path, monkeypatch, failing_file
):
    """A disk that fills up part way through is a failure like any other: the
    cause has to escape, the mutex has to come back, and the event has to be
    set. These two writes sit inside the loop's own mutex block rather than the
    handler, so they are worth exercising separately from the stages above.
    """
    monkeypatch.setattr(mc, "_claim_next_index", lambda *a, **k: 0)
    worker = _worker(tmp_path)
    blocked = str(getattr(worker, failing_file))
    real_open = open

    def selective_open(path, *args, **kwargs):
        if str(path) == blocked:
            raise OSError("no space left on device")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", selective_open)
    mutex, event = _RecordingMutex(), _Event()

    with pytest.raises(OSError, match="no space left"):
        mc.MonteCarlo._MonteCarlo__sim_producer(worker, object(), mutex, event)

    assert mutex.acquired == mutex.released, "the mutex was left held"
    assert event.flag


def test_an_unreachable_error_event_does_not_replace_the_failure_it_reports(
    tmp_path, monkeypatch
):
    """The event is a manager proxy, so notifying can fail on its own.

    It is set first and every other report is guarded, which left this one
    statement able to do the thing the guards exist to prevent: raise over the
    failure being reported, so the parent sees a connection error instead.
    """

    class _UnreachableEvent:
        def set(self):
            raise ConnectionResetError("the manager is gone")

        def is_set(self):
            raise ConnectionResetError("the manager is gone")

    monkeypatch.setattr(mc, "_claim_next_index", lambda *a, **k: 0)
    worker = _worker(tmp_path, _MonteCarlo__run_single_simulation=_raise)
    mutex = _RecordingMutex()

    with pytest.raises(_Boom):
        mc.MonteCarlo._MonteCarlo__sim_producer(
            worker, object(), mutex, _UnreachableEvent()
        )

    assert mutex.acquired == mutex.released, "the mutex was left held"


class _OneThenStop:
    """A monitor that allows exactly one simulation, then whatever is asked."""

    def __init__(self, on_update=None):
        self.count = 0
        self._on_update = on_update

    def keep_simulating(self):
        return self.count < 1

    def increment(self):
        self.count += 1
        return self.count

    def print_update_status(self):
        if self._on_update is not None:
            self._on_update()

    def print_final_status(self):
        pass


def _serial_runner(tmp_path, monkeypatch, monitor, **overrides):
    """A stand-in carrying only what ``__run_in_serial`` touches."""
    runner = _worker(tmp_path, **overrides)
    runner._error_file = runner.error_file
    runner._initial_sim_idx = 0
    runner.number_of_simulations = 1
    runner._interrupted = False
    runner._MonteCarlo__keep_the_inputs_that_did_not_finish = lambda payload: (
        runner.error_file.open("a", encoding="utf-8").write(payload)
    )
    monkeypatch.setattr(mc, "_SimMonitor", lambda **_kwargs: monitor)
    return runner


def test_a_serial_failure_records_the_traceback_not_only_the_inputs(
    tmp_path, monkeypatch
):
    """The worker path writes `{index, ...inputs, error: traceback}`. Serial
    wrote the inputs alone, so a failure after sampling named which inputs
    failed and never why, in the file the run points the user at."""
    runner = _serial_runner(
        tmp_path,
        monkeypatch,
        _OneThenStop(),
        _MonteCarlo__evaluate_flight_outputs=_raise,
    )

    with pytest.raises(_Boom):
        mc.MonteCarlo._MonteCarlo__run_in_serial(runner)

    record = json.loads(runner.error_file.read_text(encoding="utf-8").splitlines()[0])
    assert "_Boom" in record["error"]
    assert "injected" in record["error"]


def test_a_serial_failure_does_not_repeat_the_handler_frame(tmp_path, monkeypatch):
    """`raise error` names the exception again, so the handler's own line joins
    the traceback and the reader walks past it to reach the real one. A bare
    `raise` leaves the frame it came from.

    On the duplicate rather than on the original frame: the failing call
    survives either way, so asserting it is there passes both spellings.
    """
    runner = _serial_runner(
        tmp_path,
        monkeypatch,
        _OneThenStop(),
        _MonteCarlo__evaluate_flight_outputs=_raise,
    )

    with pytest.raises(_Boom) as raised:
        mc.MonteCarlo._MonteCarlo__run_in_serial(runner)

    frames = [frame.name for frame in traceback.extract_tb(raised.value.__traceback__)]
    assert "_raise" in frames, frames
    assert frames.count("__run_in_serial") == 1, frames


def test_a_committed_row_is_not_reported_as_unfinished(tmp_path, monkeypatch):
    """The pair is on disk before the progress call. A failure there used to
    append those same inputs to the error file, so one simulation appeared in
    both the inputs log and the failures."""
    runner = _serial_runner(tmp_path, monkeypatch, _OneThenStop(on_update=_raise))

    with pytest.raises(_Boom):
        mc.MonteCarlo._MonteCarlo__run_in_serial(runner)

    assert runner.input_file.read_text(encoding="utf-8").strip() == "{}"
    record = json.loads(runner.error_file.read_text(encoding="utf-8").splitlines()[0])
    assert record == {"index": 0, "error": record["error"]}
    assert "_Boom" in record["error"]


class _ClaimOnceThenFail:
    """Lets one simulation through, then fails the progress call."""

    def __init__(self):
        self.count = 0

    def keep_simulating(self):
        return self.count < 1

    def increment(self):
        self.count += 1
        return self.count

    def print_update_status(self):
        raise _Boom("progress failed")


def test_a_worker_progress_failure_does_not_repeat_committed_inputs(tmp_path):
    """The serial path clears its payload once the pair is on disk. The worker
    kept it, so a failure in the progress call wrote the same sampled inputs to
    the error file and one simulation appeared in the logs and the failures."""
    worker = _worker(
        tmp_path,
        _export_config={},
        environment=types.SimpleNamespace(last_rnd_dict={}),
        rocket=types.SimpleNamespace(last_rnd_dict={}),
        flight=types.SimpleNamespace(last_rnd_dict={}),
        _MonteCarlo__evaluate_flight_inputs=lambda index: '{"index": 0, "drew": 42}\n',
        _MonteCarlo__evaluate_flight_outputs=lambda flight, index: '{"index": 0}\n',
    )
    mutex, event = _RecordingMutex(), _Event()

    with pytest.raises(_Boom, match="progress failed"):
        mc.MonteCarlo._MonteCarlo__sim_producer(
            worker, _ClaimOnceThenFail(), mutex, event
        )

    committed = json.loads(
        worker.input_file.read_text(encoding="utf-8").splitlines()[0]
    )
    failure = json.loads(worker.error_file.read_text(encoding="utf-8").splitlines()[0])

    assert committed == {"index": 0, "drew": 42}
    assert "drew" not in failure, failure
    assert "_Boom" in failure["error"]


def test_a_serial_reporting_failure_does_not_replace_the_original(
    tmp_path, monkeypatch
):
    """The worker guards its own error write. Serial did not, so an unwritable
    error file raised OSError in place of the failure it was recording."""
    runner = _serial_runner(
        tmp_path,
        monkeypatch,
        _OneThenStop(),
        _MonteCarlo__evaluate_flight_outputs=_raise,
    )
    real_open = open

    def refuse_the_error_file(path, *args, **kwargs):
        if str(path) == str(runner.error_file):
            raise OSError("error disk unavailable")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", refuse_the_error_file)

    with pytest.warns(RuntimeWarning, match="could not be written"):
        with pytest.raises(_Boom, match="injected"):
            mc.MonteCarlo._MonteCarlo__run_in_serial(runner)


def test_a_shutdown_failure_does_not_replace_the_error_it_is_handling(monkeypatch):
    """``_bring_the_fleet_down`` promised not to raise over the failure being
    handled, but only the event was guarded. The two waits below it were not."""
    monkeypatch.setattr(mc, "_wait_for_workers", _raise)

    with pytest.warns(RuntimeWarning, match="graceful worker wait"):
        mc._bring_the_fleet_down([], _Event())


def test_reporting_a_failure_cannot_escape_when_warnings_are_errors(
    tmp_path, monkeypatch
):
    """The guard around the error write is only best effort under the default
    filter. Turn RuntimeWarning into an error, as a strict application or test
    run does, and the diagnostic becomes the exception it was describing."""
    runner = _serial_runner(
        tmp_path,
        monkeypatch,
        _OneThenStop(),
        _MonteCarlo__evaluate_flight_outputs=_raise,
    )
    real_open = open

    def refuse_the_error_file(path, *args, **kwargs):
        if str(path) == str(runner.error_file):
            raise OSError("error disk unavailable")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", refuse_the_error_file)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)

        with pytest.raises(_Boom, match="injected"):
            mc.MonteCarlo._MonteCarlo__run_in_serial(runner)


class _MutexDiesAfter(_RecordingMutex):
    """Releases cleanly a few times, then the manager connection goes.

    A mutex that fails on every release is the wrong model: the first clean
    claim would fail on its own release, before there is anything to mask.
    """

    def __init__(self, healthy_releases=1):
        super().__init__()
        self._left = healthy_releases

    def release(self):
        super().release()
        self._left -= 1
        if self._left < 0:
            raise BrokenPipeError("[Errno 32] Broken pipe")


def test_a_dying_mutex_does_not_replace_the_simulation_failure(tmp_path):
    """The release in the critical section used to run raw in ``finally``, so a
    manager that died while the lock was held handed the caller its own
    BrokenPipeError and left the real failure as context."""
    worker = _worker(tmp_path)
    monitor = _ClaimOnceThenFail()
    event = _Event()

    with pytest.warns(RuntimeWarning, match="manager mutex release"):
        with pytest.raises(_Boom, match="progress failed"):
            mc.MonteCarlo._MonteCarlo__sim_producer(
                worker, monitor, _MutexDiesAfter(healthy_releases=1), event
            )


def test_a_dying_mutex_does_not_replace_the_claim_failure():
    """Same shape one level down, where the claim itself is what fails."""

    class _ClaimRaises:
        def keep_simulating(self):
            raise _Boom("claim failed")

    with pytest.warns(RuntimeWarning, match="manager mutex release"):
        with pytest.raises(_Boom, match="claim failed"):
            mc._claim_next_index(_ClaimRaises(), _MutexDiesAfter(healthy_releases=0))


def test_a_release_that_fails_on_its_own_is_still_raised():
    """The other half. With nothing in flight a broken release is the failure,
    not a warning, so the run does not carry on against a dead manager."""

    class _Fine:
        def keep_simulating(self):
            return True

        def increment(self):
            return 1

    with pytest.raises(BrokenPipeError):
        mc._claim_next_index(_Fine(), _MutexDiesAfter(healthy_releases=0))


@pytest.mark.parametrize(
    "stopped_at, drawn",
    [
        ("environment", ({"wind": 1.0}, {}, {})),
        ("rocket", ({"wind": 1.0}, {"mass": 2.0}, {})),
        ("flight sampling", ({"wind": 1.0}, {"mass": 2.0}, {})),
        ("Flight.__init__", ({"wind": 1.0}, {"mass": 2.0}, {"inclination": 84.0})),
    ],
)
def test_an_error_row_keeps_whatever_was_drawn_before_the_failure(
    tmp_path, monkeypatch, stopped_at, drawn
):
    """A failure before the row is built must not cost the inputs as well.

    The whole row is only assembled once the flight is, so anything that raised
    inside ``create_object`` or ``Flight`` itself used to leave a traceback with
    nothing to say which inputs produced it (#1109 is one way in). Each model
    fills its own ``last_rnd_dict`` as it goes, so the draws that did happen are
    recoverable.
    """
    monkeypatch.setattr(mc, "_claim_next_index", lambda *a, **k: 0)
    environment, rocket, flight = drawn
    worker = _worker(
        tmp_path,
        environment=types.SimpleNamespace(last_rnd_dict=environment),
        rocket=types.SimpleNamespace(last_rnd_dict=rocket),
        flight=types.SimpleNamespace(last_rnd_dict=flight),
        _MonteCarlo__run_single_simulation=_raise,
    )

    with pytest.raises(_Boom):
        mc.MonteCarlo._MonteCarlo__sim_producer(
            worker, object(), _RecordingMutex(), _Event()
        )

    rows = [
        json.loads(line)
        for line in (tmp_path / "errors.txt").read_text().splitlines()
        if line.strip()
    ]
    assert rows, f"nothing was recorded for a failure at {stopped_at}"
    row = rows[0]
    assert row["partial_inputs"] is True
    assert "error" in row
    for key, value in {**environment, **rocket, **flight}.items():
        assert row[key] == value


def test_an_error_before_anything_was_drawn_still_records_the_failure(
    tmp_path, monkeypatch
):
    """With nothing drawn there is nothing to recover, and the row says so."""
    monkeypatch.setattr(mc, "_claim_next_index", lambda *a, **k: 0)
    worker = _worker(tmp_path, _MonteCarlo__run_single_simulation=_raise)

    with pytest.raises(_Boom):
        mc.MonteCarlo._MonteCarlo__sim_producer(
            worker, object(), _RecordingMutex(), _Event()
        )

    row = json.loads((tmp_path / "errors.txt").read_text().splitlines()[0])
    assert "error" in row
    assert "partial_inputs" not in row
