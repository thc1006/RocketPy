import ast
import inspect
import json
import os

import pytest

from rocketpy.simulation import monte_carlo as mc_module
from rocketpy.simulation.monte_carlo import (
    MonteCarlo,
    _refuse_logs_missing_a_simulation,
)


def _a_log(tmp_path, name, rows):
    path = tmp_path / name
    path.write_text("".join(rows), encoding="utf-8")
    return str(path)


def _row(index):
    return json.dumps({"index": index, "mass": 1.0}) + "\n"


def _complete(tmp_path, count=3, name="ok"):
    rows = [_row(index) for index in range(count)]
    return (
        _a_log(tmp_path, f"{name}.inputs.txt", rows),
        _a_log(tmp_path, f"{name}.outputs.txt", rows),
    )


def test_a_run_that_recorded_everything_is_accepted(tmp_path):
    """Logs holding every index the run asked for raise nothing."""
    inputs, outputs = _complete(tmp_path)

    _refuse_logs_missing_a_simulation(inputs, outputs, 3)


def test_blank_lines_between_rows_are_not_simulations(tmp_path):
    """A blank line is skipped rather than counted as an unreadable row."""
    # An interrupted write leaves them, and reading one as a row would report
    # a damaged log for a run that lost nothing.
    rows = [_row(0), "\n", _row(1), "   \n", _row(2)]
    inputs = _a_log(tmp_path, "gappy.inputs.txt", rows)
    outputs = _a_log(tmp_path, "gappy.outputs.txt", rows)

    _refuse_logs_missing_a_simulation(inputs, outputs, 3)


def test_a_missing_simulation_is_refused(tmp_path):
    """A gap in the output log names the first index that is missing."""
    inputs, outputs = _complete(tmp_path)
    _a_log(tmp_path, "ok.outputs.txt", [_row(0), _row(2)])

    with pytest.raises(RuntimeError, match=r"output log.*missing.*being 1"):
        _refuse_logs_missing_a_simulation(inputs, outputs, 3)


def test_a_simulation_recorded_twice_is_refused(tmp_path):
    """A duplicated index is refused, since a set alone would hide it."""
    inputs, outputs = _complete(tmp_path)
    _a_log(tmp_path, "ok.inputs.txt", [_row(0), _row(1), _row(1), _row(2)])

    with pytest.raises(RuntimeError, match="more than once"):
        _refuse_logs_missing_a_simulation(inputs, outputs, 3)


def test_a_row_that_cannot_be_read_is_refused(tmp_path):
    """A torn row means the log's contents cannot be established."""
    inputs, outputs = _complete(tmp_path)
    _a_log(tmp_path, "ok.outputs.txt", [_row(0), "{half a row\n", _row(2)])

    with pytest.raises(RuntimeError, match="cannot be read"):
        _refuse_logs_missing_a_simulation(inputs, outputs, 3)


def test_rows_numbered_past_the_run_are_left_alone(tmp_path):
    """An append below what a checkpoint already holds loses no simulation."""
    # Refusing these said rows were missing when they were extra, and moved
    # what append means inside a change about worker failure.
    rows = [_row(index) for index in range(4)]
    inputs = _a_log(tmp_path, "big.inputs.txt", rows)
    outputs = _a_log(tmp_path, "big.outputs.txt", rows)

    _refuse_logs_missing_a_simulation(inputs, outputs, 2)


def test_logs_that_hold_different_simulations_are_refused(tmp_path):
    """The input and output logs have to hold the same indices."""
    inputs = _a_log(tmp_path, "a.inputs.txt", [_row(0), _row(1)])
    outputs = _a_log(tmp_path, "a.outputs.txt", [_row(0), _row(2)])

    with pytest.raises(RuntimeError):
        _refuse_logs_missing_a_simulation(inputs, outputs, 2)


def _leave_cleanly_without_recording(_flight):
    # A worker that ends the way an out-of-memory kill ends it, but with the
    # status of one that finished. Nothing about the process says otherwise.
    os._exit(0)


def test_a_worker_that_leaves_cleanly_without_recording_is_not_a_success(
    stochastic_environment, stochastic_calisto, stochastic_flight, tmp_path
):
    """A zero exit with no row written makes ``simulate`` raise."""
    analysis = MonteCarlo(
        filename=str(tmp_path / "study"),
        environment=stochastic_environment,
        rocket=stochastic_calisto,
        flight=stochastic_flight,
        data_collector={"leave": _leave_cleanly_without_recording},
    )

    with pytest.raises(RuntimeError, match="incomplete"):
        analysis.simulate(
            number_of_simulations=6, append=False, parallel=True, n_workers=2
        )


def test_no_failure_path_waits_on_a_worker_without_a_bound():
    """No ``join`` in the parallel path is called without a timeout."""
    # An unbounded join anywhere in the parallel path puts back the hang that
    # the bounded teardown exists to end, and it does so where it is hardest
    # to notice: only when a worker is already stuck.
    tree = ast.parse(inspect.getsource(mc_module))
    run_in_parallel = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "__run_in_parallel"
    )

    unbounded = [
        node.lineno
        for node in ast.walk(run_in_parallel)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
        and not node.args
        and not node.keywords
    ]

    assert not unbounded, f"join() with no timeout at lines {unbounded}"
