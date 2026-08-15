"""What the run is allowed to call a success, and how it comes down when it is not.

The completeness check reads the two logs back and decides whether the run can
be reported as complete. Everything it accepts is a claim about the results, so
a row it cannot read, an index it cannot trust, or a file that disagrees with
its pair has to stop the run rather than be skipped past.

The shutdown tests cover the other half: a fleet where one worker is not coming
back has to be brought down in bounded time, and the failure that started it has
to survive that.
"""

import json
import os
import pathlib
import tempfile
import threading
import types

import numpy as np
import pytest

import rocketpy.simulation.monte_carlo as mc
from rocketpy.simulation import MonteCarlo


def _runner(tmp_path, rows, outputs=None, count=2, initial=0, interrupted=False):
    """A stand-in carrying only what the completeness check reads."""
    inputs_file = tmp_path / "inputs.txt"
    outputs_file = tmp_path / "outputs.txt"
    inputs_file.write_text(rows, encoding="utf-8")
    outputs_file.write_text(rows if outputs is None else outputs, encoding="utf-8")
    return types.SimpleNamespace(
        input_file=inputs_file,
        output_file=outputs_file,
        number_of_simulations=count,
        _initial_sim_idx=initial,
        _interrupted=interrupted,
    )


def _check(runner):
    mc.MonteCarlo._MonteCarlo__check_each_index_was_recorded_once(runner)


COMPLETE = '{"index": 0}\n{"index": 1}\n'


CORRUPT = {
    "a row cut off mid-write": (COMPLETE + "{not json\n", "cannot be read"),
    "a row that is not an object": (COMPLETE + "[]\n", "cannot be read"),
    "a row with no index": (COMPLETE + '{"foo": 1}\n', "cannot be read"),
    "a boolean index": ('{"index": 0}\n{"index": true}\n', "cannot be read"),
    "a float index": ('{"index": 0}\n{"index": 1.0}\n', "cannot be read"),
    "a negative index": (COMPLETE + '{"index": -1}\n', "cannot be read"),
    "an index past the run": (COMPLETE + '{"index": 99}\n', "outside the range"),
    "the same index twice": (COMPLETE + '{"index": 1}\n', "more than once"),
    "an index never written": ('{"index": 0}\n', "never written"),
}


@pytest.mark.parametrize(
    ("rows", "expected"), list(CORRUPT.values()), ids=list(CORRUPT)
)
def test_a_corrupt_log_is_not_a_successful_run(tmp_path, rows, expected):
    """Every one of these was accepted as a complete run.

    ``True`` and ``1.0`` are the two that look harmless: both compare equal to
    ``1``, so an ``isinstance`` check or a bare dict lookup counts them as the
    index they are not. Hence ``type(index) is int``.
    """
    with pytest.raises(RuntimeError, match=expected):
        _check(_runner(tmp_path, rows))


def test_a_checkpoint_that_cannot_be_read_is_refused_before_the_run(tmp_path):
    """Where the history is judged: before anything runs, not after.

    An earlier round tolerated inherited damage at the end of the run, on the
    grounds that the documented way to reach an append is to interrupt a run.
    That was the wrong place for it. A torn row holds an index nobody can
    recover, so the resume point cannot be trusted either, and resuming at the
    wrong one silently skips a simulation. The preflight refuses instead, with
    both files left exactly as they were found.
    """
    rows = '{"index": 0}\n{not json\n'
    inputs, outputs = tmp_path / "i.txt", tmp_path / "o.txt"
    inputs.write_text(rows)
    outputs.write_text(rows)
    before = inputs.read_bytes(), outputs.read_bytes()

    with pytest.raises(ValueError, match="cannot be read"):
        mc._check_the_checkpoint_supports_appending(inputs, outputs, 2)

    assert (inputs.read_bytes(), outputs.read_bytes()) == before, (
        "a refused checkpoint was modified on the way out"
    )


def test_this_run_is_still_judged_strictly_while_appending(tmp_path):
    """The other half. Tolerating the history must not tolerate the rows this
    run added, or appending would become a way to launder a bad run."""
    runner = _runner(
        tmp_path,
        '{"index": 0}\n{"index": 2}\n{"index": 2}\n',
        count=4,
        initial=2,
    )
    with pytest.raises(RuntimeError, match="more than once"):
        _check(runner)


def test_a_complete_run_is_still_accepted(tmp_path):
    """The control. Without this the table above passes on a check that
    rejects everything."""
    _check(_runner(tmp_path, COMPLETE))


def test_an_earlier_run_left_in_the_files_is_not_an_error(tmp_path):
    """``append=True`` keeps the earlier run's rows, and they are below
    ``_initial_sim_idx``. Rejecting anything outside the new range would make
    every appended run fail."""
    rows = '{"index": 0}\n{"index": 1}\n{"index": 2}\n{"index": 3}\n'
    _check(_runner(tmp_path, rows, count=4, initial=2))


def test_an_interrupted_run_still_has_to_be_readable(tmp_path):
    """Being short is allowed after Ctrl-C. Being corrupt is not: skipping the
    check entirely meant a duplicate or an unreadable row went unreported."""
    _check(_runner(tmp_path, '{"index": 0}\n', interrupted=True))

    with pytest.raises(RuntimeError, match="more than once"):
        _check(_runner(tmp_path, '{"index": 0}\n{"index": 0}\n', interrupted=True))


def test_the_two_files_have_to_agree(tmp_path):
    """A worker that wrote its inputs and stopped before its outputs. Each file
    on its own can look complete, because the row one is missing is in the
    other."""
    with pytest.raises(RuntimeError, match="disagree about which simulations"):
        _check(_runner(tmp_path, COMPLETE, outputs='{"index": 0}\n'))


class _Worker:
    """A process stub that can be told to ignore termination.

    Every call is appended to a shared ``trace`` so the order across the whole
    fleet can be asserted, not just the per-worker counts. A stub join costs no
    time, so a test that only counts calls cannot tell "signal everyone, then
    wait" from "signal one and wait for it before reaching the next".
    """

    def __init__(self, name="worker", alive=False, deaf=False, exitcode=0, trace=None):
        self.name = name
        self._alive = alive
        self._deaf = deaf
        self.exitcode = exitcode
        self.terminated = 0
        self.killed = 0
        self.joins = []
        self.trace = [] if trace is None else trace

    def is_alive(self):
        return self._alive

    def terminate(self):
        self.terminated += 1
        self.trace.append(("terminate", self.name))
        if not self._deaf:
            self._alive = False

    def kill(self):
        self.killed += 1
        self.trace.append(("kill", self.name))
        self._alive = False

    def join(self, timeout=None):
        self.joins.append(timeout)
        self.trace.append(("join", self.name))


class _Event:
    def __init__(self, flag=False):
        self.flag = flag

    def set(self):
        self.flag = True

    def is_set(self):
        return self.flag


def test_the_wait_gives_up_as_soon_as_a_worker_reports_an_error():
    """One worker is not coming back and another has already failed.

    Joining the fleet in order held the parent on the first worker forever, so
    the error the second had already reported was never seen and the cleanup
    after it never ran.
    """
    stuck, failed = _Worker(alive=True), _Worker(exitcode=1)

    mc._wait_for_workers([stuck, failed], _Event(flag=True))

    assert stuck.is_alive(), "the wait should return, not stop the workers itself"


def test_the_wait_is_bounded_when_it_is_given_a_deadline():
    """The interrupt path waits a short while for workers to notice the event.
    Without a deadline that wait was the second place Ctrl-C could hang."""
    stuck = _Worker(alive=True)

    mc._wait_for_workers([stuck], timeout=0.2)

    assert stuck.is_alive()


def test_the_wait_reaps_workers_that_had_already_finished():
    """A worker gone before the loop starts is never joined by it, and an
    unjoined child has no exit code, which the crash check downstream reads as
    a crash."""
    done = _Worker(alive=False)

    mc._wait_for_workers([done], _Event())

    assert done.joins, "a finished worker was never joined, so it was not reaped"


def test_every_worker_is_signalled_before_any_of_them_is_waited_on():
    """Signal the whole fleet, then wait on it.

    Terminating one and joining it before reaching the next made every worker
    wait out the grace period of the ones ahead of it in the list, so a single
    worker that ignores the signal delays the rest by that much each.
    """
    trace = []
    deaf = _Worker(name="deaf", alive=True, deaf=True, trace=trace)
    ordinary = _Worker(name="ordinary", alive=True, trace=trace)

    mc._stop_any_worker_still_running([deaf, ordinary], grace=0.01)

    first_join = next(i for i, (call, _) in enumerate(trace) if call == "join")
    assert not [c for c in trace[first_join:] if c[0] == "terminate"], (
        f"a worker was signalled only after another had been waited on: {trace}"
    )
    assert ordinary.terminated == 1, "the second worker was never signalled"
    assert deaf.killed == 1, "the worker that sat through terminate was not killed"
    assert not deaf.is_alive()


def test_a_worker_that_already_exited_is_left_alone():
    """The control: cleanup runs on every path, including the ones where
    nothing went wrong."""
    done = _Worker(alive=False)

    mc._stop_any_worker_still_running([done], grace=0.01)

    assert done.terminated == 0 and done.killed == 0


class _RecordingMutex:
    def __init__(self, fail_on_acquire=False):
        self.acquired = 0
        self.released = 0
        self._fail = fail_on_acquire
        self._lock = threading.Lock()

    def acquire(self, *args, **kwargs):
        self.acquired += 1
        if self._fail:
            raise ConnectionResetError("the manager is gone")
        return self._lock.acquire(*args, **kwargs)

    def release(self):
        self.released += 1
        return self._lock.release()


class _Boom(RuntimeError):
    """The failure under test, so it cannot be confused with an incidental one."""


class _Monitor:
    """Enough of a monitor for a worker that completes an iteration."""

    count = 0

    def print_update_status(self):
        pass


def _sim_worker(tmp_path, **overrides):
    attributes = {
        # The failure path recovers what was drawn from these.
        "_export_config": {},
        "_MonteCarlo__draws_before_this_simulation": ({}, {}, {}),
        "environment": types.SimpleNamespace(last_rnd_dict={}),
        "rocket": types.SimpleNamespace(last_rnd_dict={}),
        "flight": types.SimpleNamespace(last_rnd_dict={}),
        "error_file": tmp_path / "errors.txt",
        "input_file": tmp_path / "inputs.txt",
        "output_file": tmp_path / "outputs.txt",
        "_MonteCarlo__child_seed": lambda index: index,
        "_MonteCarlo__seed_simulation": lambda seed: None,
        "_MonteCarlo__run_single_simulation": object,
        "_MonteCarlo__evaluate_flight_inputs": lambda index: '{"index": 0}\n',
        "_MonteCarlo__evaluate_flight_outputs": lambda flight, index: '{"index": 0}\n',
    }
    attributes.update(overrides)
    return types.SimpleNamespace(**attributes)


def test_a_claim_that_fails_after_a_completed_run_does_not_report_that_run(
    tmp_path, monkeypatch
):
    """The state was cleared after the claim rather than before it.

    So a claim that failed on the second lap reached the handler still holding
    the row that had just been written successfully, and the error file got a
    second copy of a simulation that never failed.
    """
    claims = iter([0])

    def claim_once_then_fail(*_args, **_kwargs):
        try:
            return next(claims)
        except StopIteration:
            raise _Boom("the claim failed on the second lap") from None

    monkeypatch.setattr(mc, "_claim_next_index", claim_once_then_fail)
    reported = []
    monkeypatch.setattr(mc._SimMonitor, "reprint", staticmethod(reported.append))
    worker = _sim_worker(tmp_path)

    with pytest.raises(_Boom):
        mc.MonteCarlo._MonteCarlo__sim_producer(
            worker, _Monitor(), _RecordingMutex(), _Event()
        )

    written = (tmp_path / "errors.txt").read_text()
    assert '"index": 0' not in written, (
        f"the completed simulation was written to the error file again: {written!r}"
    )
    assert "the claim failed on the second lap" in written


def test_a_mutex_that_cannot_be_taken_is_not_then_released(tmp_path, monkeypatch):
    """The normal write path released in ``finally`` whether or not it had the
    lock, so a manager that died during acquire raised a second error on the way
    out and buried the first."""
    monkeypatch.setattr(mc, "_claim_next_index", lambda *a, **k: 0)
    mutex = _RecordingMutex(fail_on_acquire=True)

    with pytest.raises(ConnectionResetError):
        mc.MonteCarlo._MonteCarlo__sim_producer(
            _sim_worker(tmp_path), _Monitor(), mutex, _Event()
        )

    assert mutex.released == 0, "released a lock it never held"


def _serial_runner(tmp_path, row=""):
    """A stand-in carrying only what ``__run_in_serial`` touches."""
    runner = types.SimpleNamespace(
        _initial_sim_idx=0,
        number_of_simulations=2,
        _interrupted=False,
        _error_file=tmp_path / "errors.txt",
        input_file=tmp_path / "inputs.txt",
        output_file=tmp_path / "outputs.txt",
        _MonteCarlo__child_seed=lambda index: index,
        _MonteCarlo__seed_simulation=lambda seed: None,
        _MonteCarlo__run_single_simulation=object,
        _export_config={},
        environment=types.SimpleNamespace(last_rnd_dict={}),
        rocket=types.SimpleNamespace(last_rnd_dict={}),
        flight=types.SimpleNamespace(last_rnd_dict={}),
        _MonteCarlo__evaluate_flight_inputs=lambda index: row,
        _MonteCarlo__evaluate_flight_outputs=lambda flight, index: row,
    )
    runner._MonteCarlo__keep_the_inputs_that_did_not_finish = lambda payload: (
        mc.MonteCarlo._MonteCarlo__keep_the_inputs_that_did_not_finish(runner, payload)
    )
    return runner


def test_ctrl_c_before_the_first_row_keeps_the_interrupt(tmp_path, monkeypatch):
    """Half of the fix: the payload is bound before the try.

    It was assigned inside the loop body, after the two monitor calls, so Ctrl-C
    in either of those reached the handler with it still unbound and the
    interrupt came out as an UnboundLocalError instead.
    """

    class _Monitor:
        count = 0

        def __init__(self, **_kwargs):
            pass

        def keep_simulating(self):
            raise KeyboardInterrupt("ctrl-c before the first simulation")

    monkeypatch.setattr(mc, "_SimMonitor", _Monitor)
    runner = _serial_runner(tmp_path)

    mc.MonteCarlo._MonteCarlo__run_in_serial(runner)

    assert runner._interrupted, "the interrupt was not recorded"


def test_ctrl_c_between_rows_does_not_report_the_row_that_succeeded(
    tmp_path, monkeypatch
):
    """The other half: the payload is cleared before each lap, not after.

    Binding it once before the try stops the UnboundLocalError but leaves it
    holding the last completed row, so an interrupt between two iterations
    wrote a simulation that had succeeded into the error file. Both halves are
    needed, and each one passes the other's test on its own.
    """

    class _Monitor:
        count = 0

        def __init__(self, **_kwargs):
            self.laps = 0

        def keep_simulating(self):
            self.laps += 1
            if self.laps > 1:
                raise KeyboardInterrupt("ctrl-c after the first simulation")
            return True

        def increment(self):
            return 1

        def print_update_status(self):
            pass

        def print_final_status(self):
            pass

    monkeypatch.setattr(mc, "_SimMonitor", _Monitor)
    runner = _serial_runner(tmp_path, row='{"index": 0}\n')

    mc.MonteCarlo._MonteCarlo__run_in_serial(runner)

    assert runner._interrupted
    assert (tmp_path / "inputs.txt").read_text() == '{"index": 0}\n', (
        "the simulation that completed should have been written normally"
    )
    assert (tmp_path / "errors.txt").read_text() == "", (
        "a completed simulation was written to the error file as if it failed"
    )


def test_a_history_that_went_missing_is_still_caught_at_the_end(tmp_path):
    """The run is judged on every index asked for, not on its own share.

    Appending normally reaches this past a preflight that found the checkpoint
    whole, so the two questions have the same answer there. They do not when
    the check is asked directly, and the invariant worth stating is the one
    about the whole file: a four-simulation result holds four simulations.
    """
    runner = _runner(tmp_path, '{"index": 2}\n{"index": 3}\n', count=4, initial=2)

    with pytest.raises(RuntimeError, match="never written"):
        _check(runner)


def test_a_checkpoint_numbered_from_one_is_named_as_such(tmp_path):
    """Serial runs used to number from 1. Appending onto one would rewrite the
    last index rather than continue, so it is refused by name: the fix is to
    re-baseline, not to retry, and an off-by-one message would not say that.
    """
    rows = "".join('{"index": %d}\n' % index for index in (1, 2, 3))
    inputs, outputs = tmp_path / "i.txt", tmp_path / "o.txt"
    inputs.write_text(rows)
    outputs.write_text(rows)

    with pytest.raises(ValueError, match="numbered from 1") as raised:
        mc._check_the_checkpoint_supports_appending(inputs, outputs, 3)

    # The message used to offer renumbering as an alternative to re-running,
    # which lines the indices up and leaves the seeds behind: those rows came
    # from the old sequential scheme, not from this release's per-index one.
    assert "Renumbering" in str(raised.value)
    assert "without lining the seeds up" in str(raised.value)


class _Collector:
    """A model with only what the collector checks and the writer touch."""

    export_list = ("apogee",)

    def __init__(self, data_collector=None):
        self.data_collector = data_collector
        self._export_config = {}


def _outputs(data_collector, sim_idx):
    model = _Collector(data_collector)
    flight = types.SimpleNamespace(apogee=100.0)
    return json.loads(
        mc.MonteCarlo._MonteCarlo__evaluate_flight_outputs(model, flight, sim_idx)
    )


def test_a_collector_cannot_relabel_the_row_it_is_attached_to():
    """`index` pairs an inputs row with its outputs row. The custom fields used
    to be merged over it, so a collector could file a row under a different
    simulation than the one that produced it."""
    labels = iter([1, 0])
    collector = {"index": lambda _flight: next(labels)}

    assert _outputs(collector, 0)["index"] == 0
    assert _outputs(collector, 1)["index"] == 1


def test_a_permutation_of_valid_indices_would_pass_every_other_check():
    """Why the one above matters more than a malformed value would.

    A collector returning `-1` writes a row the completeness check rejects. One
    returning a permutation writes the same index set with the same counts, so
    nothing downstream can tell the outputs are on the wrong simulations.
    """
    labels = iter([1, 0])
    rows = [_outputs({"custom": lambda _f: next(labels)}, i) for i in (0, 1)]

    assert sorted(r["index"] for r in rows) == [0, 1]
    assert [r["custom"] for r in rows] == [1, 0], "the collector still runs"


@pytest.mark.parametrize(
    "key, expected",
    [("index", "reserved"), (7, "must be strings"), (None, "must be strings")],
    ids=["reserved-name", "int-key", "none-key"],
)
def test_a_collector_key_that_cannot_be_written_is_refused(key, expected):
    """A non-string key does not survive the round trip these files exist for.

    ``json.dumps`` stringifies it on the way out, so ``7`` comes back as
    ``"7"``, ``None`` as ``"null"``. Worse, it can collide: a collector holding
    both ``1`` and ``"1"`` writes ``{"1": ..., "1": ...}``, and reading that
    back leaves one column where there were two.
    """
    with pytest.raises(ValueError, match=expected):
        mc.MonteCarlo._check_data_collector(_Collector(), {key: lambda _f: 0})


def test_a_collector_changed_after_construction_is_checked_again(tmp_path):
    """`data_collector` is public and mutable, so validating it once in
    ``__init__`` is not enough. The check has to run before ``__setup_files``
    opens the logs "w+", or a rejected run destroys the previous one."""
    inputs = tmp_path / "inputs.txt"
    outputs = tmp_path / "outputs.txt"
    inputs.write_text("previous input\n", encoding="utf-8")
    outputs.write_text("previous output\n", encoding="utf-8")
    runner = types.SimpleNamespace(
        input_file=inputs,
        output_file=outputs,
        export_list=("apogee",),
        data_collector={"index": lambda _flight: 0},
        _check_data_collector=lambda collector: mc.MonteCarlo._check_data_collector(
            runner, collector
        ),
    )

    with pytest.raises(ValueError, match="reserved"):
        mc.MonteCarlo.simulate(runner, number_of_simulations=1, random_seed=42)

    assert inputs.read_text(encoding="utf-8") == "previous input\n"
    assert outputs.read_text(encoding="utf-8") == "previous output\n"


def test_a_failed_log_leaves_every_other_log_as_it_was(tmp_path, monkeypatch):
    """One unwritable log must not cost the run that is already on disk.

    The logs used to be truncated one after another, so a permission error on
    the second emptied the first on the way to raising.
    """
    logs = []
    for name in ("run.inputs.txt", "run.outputs.txt", "run.errors.txt"):
        path = tmp_path / name
        path.write_text('{"index": 0}\n', encoding="utf-8")
        logs.append(path)
    before = [path.read_bytes() for path in logs]

    real = tempfile.NamedTemporaryFile
    calls = {"n": 0}

    def fail_on_the_second(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise PermissionError(13, "Permission denied")
        return real(*args, **kwargs)

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", fail_on_the_second)

    analysis = object.__new__(MonteCarlo)
    analysis._input_file, analysis._output_file, analysis._error_file = (
        str(path) for path in logs
    )

    with pytest.raises(OSError):
        analysis._MonteCarlo__setup_files(append=False)

    assert [path.read_bytes() for path in logs] == before
    assert not list(tmp_path.glob("*.partial"))


def test_the_logs_are_emptied_when_every_one_of_them_can_be_written(tmp_path):
    """The ordinary path still leaves three empty logs behind."""
    logs = []
    for name in ("run.inputs.txt", "run.outputs.txt", "run.errors.txt"):
        path = tmp_path / name
        path.write_text('{"index": 0}\n', encoding="utf-8")
        logs.append(path)

    analysis = object.__new__(MonteCarlo)
    analysis._input_file, analysis._output_file, analysis._error_file = (
        str(path) for path in logs
    )
    analysis._MonteCarlo__setup_files(append=False)

    assert [path.read_bytes() for path in logs] == [b"", b"", b""]
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize("status", ["cancelled", "interrupted"])
def test_a_row_that_did_not_finish_says_which_way_it_stopped(status):
    """Three ways a row lands in the error file, and they must be tellable apart.

    A simulation that raised carries ``error``; one dropped because a peer
    crashed or the user interrupted carried nothing, so it read as a success
    with no error attached.
    """
    row = json.loads(mc._build_unfinished_record('{"index": 3, "a": 1}', status))

    assert row["status"] == status
    assert row["index"] == 3  # the inputs are kept, not replaced
    assert "error" not in row


def test_an_unreadable_input_row_still_reports_its_status():
    """The marker matters most when the inputs could not be written."""
    row = json.loads(mc._build_unfinished_record("not json at all", "cancelled"))

    assert row == {"status": "cancelled"}


def test_a_failed_simulation_is_not_labelled_as_stopped():
    """A real exception keeps ``error`` and gains no status, so the three differ."""
    row = json.loads(mc._build_error_record(5, '{"index": 5}', "boom"))

    assert row["error"] == "boom"
    assert "status" not in row


def test_nothing_in_flight_writes_no_row_at_all():
    """An interrupt between two simulations has nothing to mark."""
    assert mc._build_unfinished_record("", "interrupted") == ""
    assert mc._build_unfinished_record(None, "cancelled") == ""


def _generation(output, root=(42, (), 4, 0), count=0, inputs=None):
    """The record a run keeps about the logs it owns."""
    return {
        "run_id": "0123456789abcdef",
        "root_state": root,
        "seed_chosen": True,
        "committed_count": count,
        "input_file": inputs or str(pathlib.Path(output).with_name("run.inputs.txt")),
        "output_file": output,
    }


def _old_parallel_checkpoint(tmp_path, rows=2):
    """Logs shaped exactly like a clean run from before per-index seeding.

    The previous release numbered parallel runs from 0 as well, so these pass
    every check that reads only the indices.
    """
    body = "".join(json.dumps({"index": i}) + "\n" for i in range(rows))
    inputs, outputs = tmp_path / "run.inputs.txt", tmp_path / "run.outputs.txt"
    inputs.write_text(body, encoding="utf-8")
    outputs.write_text(body, encoding="utf-8")
    return inputs, outputs


def test_a_checkpoint_from_the_old_parallel_scheme_is_refused(tmp_path):
    """Index shape cannot tell the two schemes apart, so something else must.

    A clean 0..N-1 log from the previous release passes every structural check
    while its rows came from per-worker entropy, shared component seeds and a
    different sampling call sequence.
    """
    inputs, outputs = _old_parallel_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="manifest"):
        mc._check_the_checkpoint_supports_appending(str(inputs), str(outputs), 2)


def test_a_checkpoint_this_release_wrote_is_accepted(tmp_path):
    """The same rows, with the manifest beside them, continue normally."""
    inputs, outputs = _old_parallel_checkpoint(tmp_path)
    mc._write_run_manifest(str(outputs), _generation(str(outputs), root=(42, (), 4, 0)))

    mc._check_the_checkpoint_supports_appending(str(inputs), str(outputs), 2)


def test_a_manifest_from_another_scheme_is_refused(tmp_path):
    """A future or foreign scheme is named rather than guessed at."""
    inputs, outputs = _old_parallel_checkpoint(tmp_path)
    mc._write_run_manifest(str(outputs), _generation(str(outputs), root=(42, (), 4, 0)))
    path = mc._manifest_path(str(outputs))
    document = json.loads(path.read_text(encoding="utf-8"))
    document["sampling_scheme"] = "something-else-v9"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="something-else-v9"):
        mc._check_the_checkpoint_supports_appending(str(inputs), str(outputs), 2)


def test_an_empty_checkpoint_needs_no_manifest(tmp_path):
    """A first run has nothing to be continued from, so nothing to prove."""
    inputs, outputs = _old_parallel_checkpoint(tmp_path, rows=0)

    mc._check_the_checkpoint_supports_appending(str(inputs), str(outputs), 0)


def test_the_lineage_outlives_the_object_that_wrote_it(tmp_path):
    """A fresh MonteCarlo has no memory, so the manifest has to carry it.

    Rebuilding the object between two runs used to lose the root entirely, and
    the append after it could not tell it was leaving the lineage.
    """
    output = str(tmp_path / "run.outputs.txt")

    first = object.__new__(MonteCarlo)
    first._output_file = output
    first._input_file = str(pathlib.Path(output).with_name("run.inputs.txt"))
    first._MonteCarlo__capture_root_state(42)
    first._MonteCarlo__start_a_generation("0123456789abcdef")

    second = object.__new__(MonteCarlo)  # nothing carried over in memory
    second._output_file = output
    second._input_file = str(pathlib.Path(output).with_name("run.inputs.txt"))
    with pytest.raises(ValueError, match="different root"):
        second._MonteCarlo__capture_root_state(7, appending=True)

    third = object.__new__(MonteCarlo)
    third._output_file = output
    third._input_file = str(pathlib.Path(output).with_name("run.inputs.txt"))
    third._MonteCarlo__capture_root_state(None, appending=True)

    assert third._MonteCarlo__root_fingerprint == first._MonteCarlo__root_fingerprint


def _three_logs_on_disk(tmp_path, mode=0o644):
    """Three logs holding a row each, at a mode a caller might have chosen."""
    logs = []
    for name in ("run.inputs.txt", "run.outputs.txt", "run.errors.txt"):
        path = tmp_path / name
        path.write_text('{"index": 0}\n', encoding="utf-8")
        os.chmod(path, mode)
        logs.append(path)
    return logs


@pytest.mark.parametrize("failing_call", [1, 2, 3, 4, 5, 6])
def test_a_failed_install_puts_every_log_back(tmp_path, monkeypatch, failing_call):
    """``os.replace`` is atomic for one file, not for three of them.

    Each destination is moved aside and replaced in turn, so a failure part way
    through used to leave the earlier ones already emptied. Every step is driven
    to fail here, since which one breaks is not something to assume.
    """
    logs = _three_logs_on_disk(tmp_path)
    before = [path.read_bytes() for path in logs]
    real_replace = os.replace
    calls = {"n": 0}

    def fail_on_the_chosen_call(source, destination):
        calls["n"] += 1
        if calls["n"] == failing_call:
            raise OSError(13, "injected")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_on_the_chosen_call)

    with pytest.raises(OSError):
        mc._create_empty_logs_atomically([str(path) for path in logs])

    assert [path.read_bytes() for path in logs] == before
    assert not [x for x in tmp_path.iterdir() if x.suffix in (".partial", ".kept")]


def test_a_keyboard_interrupt_part_way_through_puts_every_log_back(
    tmp_path, monkeypatch
):
    """Rollback has to catch BaseException, not only OSError."""
    logs = _three_logs_on_disk(tmp_path)
    before = [path.read_bytes() for path in logs]
    real_replace = os.replace
    calls = {"n": 0}

    def interrupt_on_the_third(source, destination):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", interrupt_on_the_third)

    with pytest.raises(KeyboardInterrupt):
        mc._create_empty_logs_atomically([str(path) for path in logs])

    assert [path.read_bytes() for path in logs] == before
    assert not [x for x in tmp_path.iterdir() if x.suffix in (".partial", ".kept")]


def test_the_logs_keep_the_mode_they_had(tmp_path):
    """A staged file opens at 0600, which must not narrow the log it replaces.

    Compared against what the log had rather than against 0644: Windows honours
    only the read-only bit, so a file asked for 0644 reports 0666 there and a
    literal would be testing the platform instead of the carry-over.
    """
    logs = _three_logs_on_disk(tmp_path, mode=0o644)
    before = [path.stat().st_mode & 0o777 for path in logs]

    mc._create_empty_logs_atomically([str(path) for path in logs])

    assert [path.read_bytes() for path in logs] == [b"", b"", b""]
    assert [path.stat().st_mode & 0o777 for path in logs] == before
    assert not [x for x in tmp_path.iterdir() if x.suffix in (".partial", ".kept")]


def _models_that_have_drawn():
    """Three stochastic models carrying the values of a finished simulation."""
    return [
        types.SimpleNamespace(
            last_rnd_dict={f"{name}_value": 1.0}, _set_stochastic=lambda seed: None
        )
        for name in ("environment", "rocket", "flight")
    ]


def test_nothing_drawn_recovers_nothing():
    """The recovery is a best effort, not a source of empty rows."""
    empty = [types.SimpleNamespace(last_rnd_dict={}) for _ in range(3)]

    assert mc._inputs_drawn_so_far(empty, [{}, {}, {}], 3, {}) == ""


def test_two_logs_that_differ_only_in_extension_get_their_own_manifest(tmp_path):
    """``with_suffix`` gave ``run.txt`` and ``run.json`` one manifest between them."""
    first = mc._manifest_path(str(tmp_path / "run.txt"))
    second = mc._manifest_path(str(tmp_path / "run.json"))

    assert first != second
    assert first.name.startswith("run.txt")


def test_recovery_takes_the_models_that_published_and_leaves_the_rest():
    """Recovery is per model, because publication is.

    A model that finished its draw binds a new ``last_rnd_dict``; one that
    raised part way through is still holding the object it had. Comparing
    against what was held at seeding tells the two apart without touching that
    documented state.
    """
    finished = types.SimpleNamespace(last_rnd_dict={"wind": 1.0})
    stale = types.SimpleNamespace(last_rnd_dict={"from_the_run_before": 9.0})
    failed_mid_draw = types.SimpleNamespace(last_rnd_dict={})
    held_before = [{}, stale.last_rnd_dict, failed_mid_draw.last_rnd_dict]

    row = json.loads(
        mc._inputs_drawn_so_far((finished, stale, failed_mid_draw), held_before, 7, {})
    )

    assert row["wind"] == 1.0
    assert row["partial_inputs"] is True
    assert row["index"] == 7
    assert "from_the_run_before" not in row, "a previous simulation was reported"


def test_recovery_reports_nothing_when_no_model_published():
    """A failure before the first model finished has nothing to recover."""
    models = [types.SimpleNamespace(last_rnd_dict={"old": 1.0}) for _ in range(3)]
    held_before = [model.last_rnd_dict for model in models]

    assert mc._inputs_drawn_so_far(models, held_before, 7, {}) == ""


def test_seeding_marks_what_each_model_was_already_holding():
    """The marks are what make a stale draw tellable from a fresh one.

    Taken by reference rather than copied, and taken at seeding rather than
    after, so nothing has to clear ``last_rnd_dict`` to keep the recovery
    honest.
    """
    analysis = object.__new__(MonteCarlo)
    held = [{"a": 1}, {"b": 2}, {"c": 3}]
    analysis.environment, analysis.rocket, analysis.flight = (
        types.SimpleNamespace(last_rnd_dict=one, _set_stochastic=lambda seed: None)
        for one in held
    )

    analysis._MonteCarlo__seed_simulation(np.random.SeedSequence(42))

    marks = analysis._MonteCarlo__draws_before_this_simulation
    assert [mark is one for mark, one in zip(marks, held)] == [True, True, True]
    assert [
        model.last_rnd_dict
        for model in (analysis.environment, analysis.rocket, analysis.flight)
    ] == held


@pytest.mark.parametrize(
    "broken, complaint",
    [
        ({"log_format": "csv-v1"}, "log format"),
        ({"seed_chosen": "false"}, "seed_chosen"),
        ({"root_state": None}, "no root_state"),
        ({"root_state": {"entropy": 1}}, "cannot be rebuilt"),
    ],
)
def test_a_manifest_that_does_not_describe_a_root_is_refused(
    tmp_path, broken, complaint
):
    """Scheme and version alone were not enough to trust a checkpoint.

    A document naming the right scheme but carrying no usable root passed, and
    the append then went ahead with nothing to compare its own root against.
    ``bool("false")`` is True, so a string there read as a chosen seed.
    """
    inputs, outputs = _old_parallel_checkpoint(tmp_path)
    mc._write_run_manifest(str(outputs), _generation(str(outputs), root=(42, (), 4, 0)))
    path = mc._manifest_path(str(outputs))
    document = json.loads(path.read_text(encoding="utf-8"))
    document.update(broken)
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match=complaint):
        mc._check_the_checkpoint_supports_appending(str(inputs), str(outputs), 2)


def test_a_failed_manifest_write_leaves_the_previous_one(tmp_path, monkeypatch):
    """The manifest gates appends now, so a torn write cannot be left behind."""
    output = str(tmp_path / "run.outputs.txt")
    mc._write_run_manifest(output, _generation(output, root=(42, (), 4, 0)))
    before = mc._manifest_path(output).read_bytes()

    real_replace = os.replace

    def fail(source, destination):
        if str(destination).endswith(".manifest.json"):
            raise OSError(13, "injected")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail)
    with pytest.warns(RuntimeWarning, match="run manifest"):
        mc._write_run_manifest(output, _generation(output, root=(7, (), 4, 0)))

    assert mc._manifest_path(output).read_bytes() == before
    assert not list(tmp_path.glob("*.partial"))


def test_the_manifest_counts_the_rows_that_are_there(tmp_path):
    """The count comes from the logs, not from the number asked for.

    An interrupt before the first new row used to move the metadata without
    moving the logs, and a target that was never reached claimed rows that do
    not exist.
    """
    _, outputs = _old_parallel_checkpoint(tmp_path, rows=3)
    analysis = object.__new__(MonteCarlo)
    analysis._output_file = str(outputs)
    analysis._input_file = str(tmp_path / "run.inputs.txt")
    analysis._MonteCarlo__generation = _generation(str(outputs), count=99)

    analysis._MonteCarlo__record_what_was_committed()

    recorded = mc._read_run_manifest(str(outputs))
    assert recorded["committed_count"] == 3, "the target, not the rows, was recorded"


def test_a_count_that_cannot_be_taken_leaves_the_previous_one(tmp_path):
    """Bookkeeping after a finished run must not fail the run."""
    output = str(tmp_path / "gone.outputs.txt")
    analysis = object.__new__(MonteCarlo)
    analysis._output_file = output
    analysis._input_file = str(tmp_path / "run.inputs.txt")
    analysis._MonteCarlo__generation = _generation(output, count=7)

    with pytest.warns(RuntimeWarning, match="committed count"):
        analysis._MonteCarlo__record_what_was_committed()

    assert mc._read_run_manifest(output)["committed_count"] == 7


@pytest.mark.parametrize("missing", ["run_id", "committed_count"])
def test_a_manifest_without_an_identity_is_refused(tmp_path, missing):
    """A manifest has to say which run and how many rows it describes."""
    inputs, outputs = _old_parallel_checkpoint(tmp_path)
    mc._write_run_manifest(str(outputs), _generation(str(outputs), count=2))
    path = mc._manifest_path(str(outputs))
    document = json.loads(path.read_text(encoding="utf-8"))
    del document[missing]
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match=missing.replace("_", "[_ ]")):
        mc._check_the_checkpoint_supports_appending(str(inputs), str(outputs), 2)


def test_the_manifest_names_the_logs_it_was_written_for(tmp_path):
    """Pairing one run's inputs with another's outputs has to be visible."""
    _, outputs = _old_parallel_checkpoint(tmp_path)
    mc._write_run_manifest(str(outputs), _generation(str(outputs), count=2))

    recorded = mc._read_run_manifest(str(outputs))

    assert recorded["output_log"] == "run.outputs.txt"
    assert recorded["input_log"] == "run.inputs.txt"
    assert recorded["run_id"]
