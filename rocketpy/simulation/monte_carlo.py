"""
Monte Carlo Simulation Module for RocketPy

This module defines the `MonteCarlo` class, which is used to perform Monte Carlo
simulations of rocket flights. The Monte Carlo simulation is a powerful tool for
understanding the variability and uncertainty in the performance of rocket flights
by running multiple simulations with varied input parameters.

Notes
-----
This module is still under active development, and some features or attributes may
change in future versions. Users are encouraged to check for updates and read the
latest documentation.
"""

import csv
import json
import os
import tempfile
import traceback
import warnings
from contextlib import contextmanager
from copy import deepcopy
from numbers import Real
from pathlib import Path
from time import monotonic, sleep, time

import numpy as np
import simplekml
from scipy.stats import bootstrap

from rocketpy._encoders import RocketPyEncoder
from rocketpy.plots.monte_carlo_plots import _MonteCarloPlots
from rocketpy.prints.monte_carlo_prints import _MonteCarloPrints
from rocketpy.simulation.flight import Flight
from rocketpy.tools import (
    _seed_sequence_to_int,
    generate_monte_carlo_ellipses,
    generate_monte_carlo_ellipses_coordinates,
    import_optional_dependency,
)


# TODO: Create evolution plots to analyze convergence
def _stage_an_empty_log(destination):
    """An empty file beside ``destination``, ready to be moved onto it.

    Same directory, so the move is a rename rather than a copy across devices,
    and at the mode ``destination`` already has: a staged file opens at 0600,
    which would otherwise narrow a log the caller left readable.
    """
    keep_mode = destination.stat().st_mode & 0o7777 if destination.exists() else None
    handle = tempfile.NamedTemporaryFile(  # pylint: disable=consider-using-with
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f"{destination.name}.",
        suffix=".partial",
        delete=False,
    )
    handle.close()
    if keep_mode is not None:
        os.chmod(handle.name, keep_mode)
    return handle.name


def _create_empty_logs_atomically(paths):
    """Put every log in place empty, or leave every one of them as it was.

    Truncating them one after another empties the first before a bad path or a
    permission can stop the second. Each is staged beside its destination and
    installed only once all of them exist, the one it replaces is kept until
    every install has gone through, and anything already installed is put back
    if a later one fails.

    Not a filesystem transaction: ``os.replace`` is atomic for one file, and
    three of them are three atomic steps with a rollback between. A generation
    directory swapped by a single pointer would be the stronger guarantee, and
    would change what the three public log paths mean.
    """
    staged, moved_aside, created, done = [], [], [], 0
    try:
        for path in paths:
            destination = Path(path)
            staged.append((_stage_an_empty_log(destination), destination))

        for temporary, destination in staged:
            # Recorded as each step happens, not after the pair: an install that
            # fails between them would otherwise leave the original moved aside
            # with nothing tracking where it went.
            if destination.exists():
                kept = f"{temporary}.kept"
                os.replace(destination, kept)
                moved_aside.append((destination, kept))
            else:
                created.append(destination)
            os.replace(temporary, destination)
            done += 1
    except BaseException as error:
        for destination, kept in reversed(moved_aside):
            _best_effort(lambda k=kept, d=destination: os.replace(k, d), "log rollback")
        for destination in reversed(created):
            _best_effort(lambda d=destination: os.remove(d), "log rollback")
        for temporary, _ in staged[done:]:
            _best_effort(lambda t=temporary: os.remove(t), "staged log cleanup")
        if isinstance(error, OSError):
            raise OSError(f"Error creating files: {error}") from error
        raise

    for _, kept in moved_aside:
        _best_effort(lambda k=kept: os.remove(k), "replaced log cleanup")


def _seed_root_fingerprint(root):
    """A comparable identity for a seed root, when the root itself is not one.

    ``entropy`` may be an ndarray, and comparing two of those answers with an
    array rather than a verdict, so a tuple holding one raises instead of
    deciding. ``[1, 2, 3]`` and ``(1, 2, 3)`` are the same seed and compare
    unequal. The generated words settle both. ``n_children_spawned`` stays,
    because ``__child_seed`` counts from it and two roots that differ only there
    hand out different children.
    """
    return (
        tuple(int(word) for word in root.generate_state(4, dtype=np.uint32)),
        tuple(int(key) for key in root.spawn_key),
        int(root.pool_size),
        int(root.n_children_spawned),
    )


def _warn_when_appending_leaves_the_lineage(previous, previous_chosen, current):
    """Say so when appended rows stop sharing the seed lineage below them.

    A file holding two lineages is valid on every structural check, so nothing
    else can notice. Only reachable when this object already ran from a chosen
    seed and is now continuing from a different root (#1075).
    """
    if not previous_chosen or previous is None or previous == current:
        return
    warnings.warn(
        "Appending to a run that was seeded, with a different root. Rows from "
        "here on derive from a new seed lineage, and the file records both "
        "without saying which is which. Pass the original random_seed to "
        "continue the same run.",
        RuntimeWarning,
        stacklevel=3,
    )


# Written by the run itself and used to pair an inputs row with its outputs row.
# A collector that supplies one can relabel a row without tripping any check.
_RESERVED_RECORD_KEYS = frozenset({"index"})


class MonteCarlo:  # pylint: disable=too-many-public-methods
    """Class to run a Monte Carlo simulation of a rocket flight.

    Attributes
    ----------
    filename : str
        Represents the initial part of the export filenames or the .txt file
        containing the outputs of a previous simulation.
    environment : StochasticEnvironment
        The stochastic environment object to be iterated over.
    rocket : StochasticRocket
        The stochastic rocket object to be iterated over.
    flight : StochasticFlight
        The stochastic flight object to be iterated over.
    export_list : list
        The list of variables to export at each simulation.
    data_collector : dict
        A dictionary whose keys are the names of the additional
        exported variables and the values are callback functions.
    inputs_log : list
        List of dictionaries with the inputs used in each simulation.
    outputs_log : list
        List of dictionaries with the outputs of each simulation.
    errors_log : list
        List of dictionaries with the errors of each simulation.
    num_of_loaded_sims : int
        Number of simulations loaded from output_file currently being used.
    results : dict
        Monte Carlo analysis results organized in a dictionary where the keys
        are the names of the saved attributes, and the values are lists with all
        the result numbers of the respective attributes.
    processed_results : dict
        Dictionary with the mean and standard deviation of each parameter
        available in the results.
    prints : _MonteCarloPrints
        Object with methods to print information about the Monte Carlo simulation.
        Use help(MonteCarlo.prints) for more information.
    plots : _MonteCarloPlots
        Object with methods to plot information about the Monte Carlo simulation.
        Use help(MonteCarlo.plots) for more information.
    number_of_simulations : int
        Number of simulations to be run.
    total_wall_time : float
        The total elapsed real-world time from the start to the end of the
        simulation, including all waiting times and delays.
    total_cpu_time : float
        The total CPU time spent running the simulation, excluding the time
        spent waiting for I/O operations or other processes to complete.
    """

    # No run yet, so nothing to continue. Class-level so an instance built
    # without __init__ still answers.
    __root_state = None
    __root_fingerprint = None
    __root_seed_given = False
    # What the logs on disk actually hold, which only moves once a run has
    # put rows from its own root into them.
    __committed_fingerprint = None
    __committed_seed_given = False

    def __init__(
        self,
        filename,
        environment,
        rocket,
        flight,
        export_list=None,
        data_collector=None,
    ):
        """
        Initialize a MonteCarlo object.

        Parameters
        ----------
        filename : str
            Represents the initial part of the export filenames or the .txt file
            containing the outputs of a previous simulation.
        environment : StochasticEnvironment
            The stochastic environment object to be iterated over.
        rocket : StochasticRocket
            The stochastic rocket object to be iterated over.
        flight : StochasticFlight
            The stochastic flight object to be iterated over.
        export_list : list, optional
            The list of variables to export. If None, the default list will be
            used, which includes the following variables: `apogee`, `apogee_time`,
            `apogee_x`, `apogee_y`, `t_final`, `x_impact`, `y_impact`,
            `impact_velocity`, `initial_stability_margin`,
            `out_of_rail_stability_margin`, `out_of_rail_time`,
            `out_of_rail_velocity`, `max_mach_number`, `frontal_surface_wind`,
            `lateral_surface_wind`. Default is None.
        data_collector : dict, optional
            A dictionary whose keys are the names of the exported variables
            and the values are callback functions. The keys (variable names) must not
            overwrite the default names on 'export_list'. The callback functions receive
            a Flight object and returns a value of that variable. For instance

            .. code-block:: python

                custom_data_collector = {
                    "max_acceleration": lambda flight: max(flight.acceleration(flight.time)),
                    "date": lambda flight: flight.env.date,
                }

        Returns
        -------
        None
        """
        warnings.warn(
            "This class is still under testing and some attributes may be "
            "changed in next versions",
            UserWarning,
        )

        self.filename = Path(filename)
        self.environment = environment
        self.rocket = rocket
        self.flight = flight
        self.export_list = []
        self.inputs_log = []
        self.outputs_log = []
        self.errors_log = []
        self.num_of_loaded_sims = 0
        self.results = {}
        self.processed_results = {}
        self.prints = _MonteCarloPrints(self)
        self.plots = _MonteCarloPlots(self)

        self.export_list = self.__check_export_list(export_list)
        self._check_data_collector(data_collector)
        self.data_collector = data_collector

        self.import_inputs(self.filename.with_suffix(".inputs.txt"))
        self.import_outputs(self.filename.with_suffix(".outputs.txt"))
        self.import_errors(self.filename.with_suffix(".errors.txt"))

    def simulate(
        self,
        number_of_simulations,
        append=False,
        parallel=False,
        n_workers=None,
        *,
        random_seed=None,
        **kwargs,
    ):
        """
        Runs the Monte Carlo simulation and saves all data.

        Parameters
        ----------
        number_of_simulations : int
            Number of simulations to be run, must be non-negative.
        append : bool, optional
            If True, resume the existing files. ``number_of_simulations`` is
            then the target total rather than a number to add, and a value
            below what the files already hold is refused. This is not a
            reproducible resume: the root is not stored in the files, so pass
            the same ``random_seed`` to stay on one lineage. Continuing from a
            different root warns, but only within the object that ran both
            (#1075). If False, the files will be overwritten. Default is False.
        parallel : bool, optional
            If True, the simulations will be run in parallel. Default is False.
        n_workers : int, optional
            Number of workers to be used if ``parallel=True``. If None, the
            number of workers will be equal to the number of CPUs available.
            A minimum of 2 workers is required for parallel mode.
            Default is None.
        random_seed : int, numpy integer, sequence of ints, or SeedSequence, optional
            Root seed for the run. When provided, the mapping from simulation
            index to sampled inputs is reproducible and identical across serial
            and parallel execution and across any number of workers: each
            simulation index derives its own decorrelated child stream from this
            root, so index ``i`` receives the same inputs no matter which worker
            runs it. The rows themselves are written in completion order, since
            a worker takes the log lock once its simulation is done, so the file
            order can differ between runs. Compare by the recorded index rather
            than byte for byte.

            What is reproduced is the draw each index makes, not a transcript of
            what reached ``Flight``: #1090 still draws the ``StochasticFlight``
            dictionary more than once. ``append=True`` does not carry its seed
            lineage either (#1075), and ``simulate_convergence`` seeds neither
            its batches nor its bootstrap resampling (#1077), so a convergence
            study is outside this guarantee. Randomness that reaches a flight
            from anywhere but a stochastic model, a user callback most likely,
            is outside it as well.

            With ``include_function_data=True`` a record also carries a
            ``Function``'s signature hash and serialised source, which describe
            the object rather than the value drawn for it, so a run under
            ``spawn`` or ``forkserver`` writes different ones for the same
            inputs. Six fields differ across that boundary, all of them these.
            Pass ``include_function_data=False`` when the records need to
            compare field for field.

            Reproducibility is also scoped to one environment. NumPy promises a
            stream only for the same BitGenerator, seed, call sequence, build
            and machine, and reserves the right to change what ``default_rng``
            returns. A seed fixes the lineage of a run; it is not an archive
            format that survives a version bump. A supplied ``SeedSequence``
            is copied from its full state rather than consumed, so repeated calls
            with the same seed reproduce the same inputs. Each model is reseeded
            with a 128-bit integer -- the seed type a custom sampler's
            ``reset_seed`` accepts. A stateful ``numpy.random.Generator`` or
            ``BitGenerator`` is rejected (it is an RNG to draw from, not a fixed
            seed); pass the seed it was built from. Default is
            None, which draws fresh entropy on each run -- the previous,
            non-reproducible default. This seeding is informed by Scientific Python
            SPEC 7 but keeps immutable seed-snapshot semantics rather than sharing a
            ``Generator``.
        kwargs : dict
            Custom arguments for simulation export of the ``inputs`` file. Options
            are:

                * ``include_outputs``: whether to also include outputs data of the
                  simulation. Default is ``False``.

                * ``include_function_data``: whether to include ``rocketpy.Function``
                  results into the export. Default is ``True``.

            See ``rocketpy._encoders.RocketPyEncoder`` for more information.

        Returns
        -------
        None

        Notes
        -----
        If you need to stop the simulations after starting them, you can
        interrupt the process and the files will be saved with the results
        until the last iteration. You can then load the results and continue
        the simulation by running the ``simulate`` method again with the
        same number of simulations and setting `append=True`.

        Not every interruption leaves a checkpoint that can be continued. The
        two logs have to hold the same simulations as a complete run of indices
        from zero, and a parallel run can stop with one worker's index missing
        while a later one is already written. Such a checkpoint is refused
        rather than repaired, which is #1075.

        Important
        ---------
        If you use `append=False` and the files already exist, they will be
        overwritten. Make sure to save the files with the results before
        running the simulation again with `append=False`.
        """
        # Everything that can be judged from the arguments alone happens before
        # __setup_files, which opens both logs "w+" and empties them. Raising
        # after that point destroys the previous run on the way out.
        _validate_simulation_count(number_of_simulations)
        # Again rather than only in __init__: the attribute is public and a key
        # added after construction would otherwise reach the logs unchecked.
        self._check_data_collector(self.data_collector)
        if parallel:
            n_workers = self.__validate_number_of_workers(n_workers)
            # multiprocess is an optional extra. Imported here, an install
            # without rocketpy[monte-carlo] raised only after __setup_files had
            # already emptied the previous run's results.
            _import_multiprocess()

        self._export_config = kwargs
        self.number_of_simulations = number_of_simulations
        self._initial_sim_idx = self.num_of_loaded_sims if append else 0
        if append:
            self.__check_this_append_can_continue(number_of_simulations)
        # Both run paths catch Ctrl-C, save what they have and return, so a
        # stopped run is incomplete on purpose and the completeness check below
        # has to know the difference between that and a worker going missing.
        self._interrupted = False

        # Capture the small, picklable root seed state once per run (every
        # simulation index derives its child seed from it, see __child_seed).
        # This validates random_seed *before* __setup_files truncates any
        # existing output, so an invalid seed cannot destroy prior results on
        # the way to raising.
        self.__capture_root_state(random_seed, appending=append)

        print("Starting Monte Carlo analysis")

        self.__setup_files(append)
        if not append:
            # Emptied just now, so whatever they held is gone and this root
            # owns them even if the run below never reaches its first row.
            self.__commit_root_lineage()

        if parallel:
            self.__run_in_parallel(n_workers)
        else:
            self.__run_in_serial()

        if number_of_simulations > self._initial_sim_idx:
            self.__commit_root_lineage()
        self.__check_each_index_was_recorded_once()
        self.__terminate_simulation()

    def __setup_files(self, append):
        """
        Sets up the files for the simulation, creating them if necessary.

        Parameters
        ----------
        append : bool
            If ``True``, the results will be appended to the existing files. If
            ``False``, the files will be overwritten.

        Returns
        -------
        None
        """
        if not append:
            _create_empty_logs_atomically(
                (self._input_file, self._output_file, self._error_file)
            )
            return

        # Resuming reads only, so nothing here can damage what is already there.
        try:
            with open(self._input_file, "r+", encoding="utf-8") as input_file:
                idx_i = len(input_file.readlines())
            with open(self._output_file, "r+", encoding="utf-8") as output_file:
                idx_o = len(output_file.readlines())
            with open(self._error_file, "r+", encoding="utf-8"):
                pass

            if idx_i != idx_o:
                warnings.warn(
                    "Input and output files are not synchronized", UserWarning
                )

        except OSError as error:
            raise OSError(f"Error creating files: {error}") from error

    @staticmethod
    def __root_seed_sequence(random_seed):
        """Build a fresh ``SeedSequence`` root from ``random_seed``.

        ``random_seed`` may be an int (or any entropy ``numpy.random.SeedSequence``
        accepts), an existing ``SeedSequence``, or ``None`` for fresh entropy. A
        supplied ``SeedSequence`` is copied from its full ``state``, so the
        spawning below neither mutates the caller's object nor advances a shared
        child counter between calls; repeated ``simulate`` calls with the same
        seed then stay reproducible. A stateful ``Generator``/``BitGenerator`` is
        not accepted, since using it as an immutable seed would contradict its
        consume-on-use semantics. Pass the seed the generator was built from.
        ``rng.bit_generator.seed_seq`` also works, but only on NumPy 1.25 and
        above, which is later than this package's floor.
        """
        if isinstance(random_seed, np.random.SeedSequence):
            return np.random.SeedSequence(**random_seed.state)
        if isinstance(random_seed, (np.random.Generator, np.random.BitGenerator)):
            raise TypeError(
                "random_seed must be an int, a sequence of non-negative "
                "integers, or a numpy.random.SeedSequence, not a "
                f"{type(random_seed).__name__}. Pass the seed the generator "
                "was built from; rng.bit_generator.seed_seq also works on "
                "NumPy 1.25 and above."
            )
        return np.random.SeedSequence(random_seed)

    def __capture_root_state(self, random_seed, appending=False):
        """Capture the small, picklable root seed state for this run.

        Stored once so serial mode and every parallel worker derive the same
        per-index child seeds from it (see ``__child_seed``), instead of
        materializing and pickling the full ``spawn(number_of_simulations)``
        list to each process.

        Deep-copied, because ``SeedSequence`` keeps a sequence entropy by
        reference. Without it a caller who mutates the list they passed changes
        the children this run derives, which is the opposite of a snapshot.
        """
        root = self.__root_seed_sequence(random_seed)
        self.__root_state = (
            deepcopy(root.entropy),
            tuple(root.spawn_key),
            root.pool_size,
            root.n_children_spawned,
        )
        # The state above rebuilds children and cannot be compared; the
        # fingerprint compares and cannot rebuild. Both, rather than one.
        self.__root_fingerprint = _seed_root_fingerprint(root)
        # Whether a caller chose this root or it came from fresh entropy. Only a
        # chosen one is a lineage there is any point in continuing.
        self.__root_seed_given = random_seed is not None
        if appending:
            # The manifest first: it outlives this object, which the attributes
            # below do not, and it describes the rows actually in the log.
            recorded = _fingerprint_from_manifest(self.output_file)
            previous, previous_chosen = recorded or (
                self.__committed_fingerprint,
                self.__committed_seed_given,
            )
            _warn_when_appending_leaves_the_lineage(
                previous, previous_chosen, self.__root_fingerprint
            )

    def __check_this_append_can_continue(self, number_of_simulations):
        """Everything an append has to satisfy before a file is opened.

        Held here rather than after the run so a checkpoint that cannot be
        continued costs no simulations and is left exactly as it was found.
        """
        _check_the_checkpoint_supports_appending(
            self.input_file, self.output_file, self._initial_sim_idx
        )
        # ``number_of_simulations`` is the target to reach, not a batch to add.
        # Below the checkpoint it ran nothing, reported success, and left a file
        # with more simulations than the caller had asked for.
        if number_of_simulations < self._initial_sim_idx:
            raise ValueError(
                f"number_of_simulations is the total to reach when "
                f"append=True. The checkpoint already holds "
                f"{self._initial_sim_idx} simulations, more than the "
                f"requested {number_of_simulations}."
            )

    def __commit_root_lineage(self):
        """Record that the logs now hold rows derived from this run's root.

        Kept apart from capturing it, because the capture runs before anything
        opens a file. A run that warns and then adds nothing, or that dies before
        its first row, must leave the logs owned by the root already in them, or
        the append after it compares against a lineage that was never written.
        Object-local, so a rebuilt ``MonteCarlo`` starts blank; carrying it in
        the files is #1075.
        """
        self.__committed_fingerprint = self.__root_fingerprint
        self.__committed_seed_given = self.__root_seed_given
        _write_run_manifest(self.output_file, self.__root_state, self.__root_seed_given)

    def __child_seed(self, sim_idx):
        """Return the seed sequence for a single simulation index.

        This equals ``root.spawn(number_of_simulations)[sim_idx]`` but is O(1)
        in time and memory: ``SeedSequence.spawn`` derives child ``i`` by
        appending ``n_children_spawned + i`` to the parent ``spawn_key``, so
        rebuilding that one child directly reproduces it bit-for-bit while
        letting a worker reconstruct any index from the small root state alone.
        """
        entropy, spawn_key, pool_size, base = self.__root_state
        return np.random.SeedSequence(
            entropy=entropy,
            spawn_key=(*spawn_key, base + sim_idx),
            pool_size=pool_size,
        )

    def __seed_simulation(self, child_seed):
        """Reseed the stochastic models for a single simulation index.

        The per-index child seed is split three ways so the environment,
        rocket and flight draw from independent streams instead of sharing
        one. Seeding per simulation index (not per worker) is what makes the
        sampled inputs invariant to the execution mode and to the number of
        workers. Each sub-stream is handed over as a 128-bit ``int`` (see
        ``_seed_sequence_to_int``) so custom samplers keep working.
        """
        # Cleared as well as reseeded: a simulation that fails before it draws
        # would otherwise report the previous one's values as its own, and a
        # worker keeps the same models for every index it claims.
        for model in (self.environment, self.rocket, self.flight):
            model.last_rnd_dict = {}
        env_seed, rocket_seed, flight_seed = child_seed.spawn(3)
        self.environment._set_stochastic(_seed_sequence_to_int(env_seed))
        self.rocket._set_stochastic(_seed_sequence_to_int(rocket_seed))
        self.flight._set_stochastic(_seed_sequence_to_int(flight_seed))

    def __check_each_index_was_recorded_once(self):
        """Every index this run claimed left exactly one input and one output row.

        The counter hands each index out once, so a missing one means a worker
        stopped between claiming and writing, and a repeated one means two
        claimed the same index. Neither is visible in the files themselves: the
        rows look well formed, and reading them back keyed by index hides the
        duplicate behind the row that overwrote it. Both make the results wrong
        while the run reports success, which is the thing per-index seeding is
        supposed to rule out.

        Only over the range this run produced. ``append=True`` leaves earlier
        runs in the same files, and ``number_of_simulations`` is the total to
        reach rather than a count to add, so the new indices are
        ``_initial_sim_idx`` up to it.

        A run stopped with Ctrl-C is short on purpose, so the indices it never
        reached are not an error. What it did write is still held to the rest:
        readable rows, one row per index, and nothing outside the range.

        Only over the indices this run claimed. An ``append`` run exists to
        carry on from a file some earlier run left behind, and the documented
        way to reach one is to interrupt a run, so that file can hold a torn
        row or a pair that disagrees. Judging this run on that damage would
        make the very files ``append`` is for the ones it refuses, so anything
        below ``_initial_sim_idx`` is reported and not raised on.
        """
        inputs, damaged = _recorded_indices("inputs", self.input_file)
        outputs, damaged_outputs = _recorded_indices("outputs", self.output_file)
        damaged += damaged_outputs

        # First, because a torn row is the root cause and the checks below are
        # its symptoms: a row that will not parse also makes the two files
        # disagree, and "files disagree" points at the wrong thing.
        #
        # Always this run's doing. An append only gets here past a preflight
        # that read the checkpoint and found it whole, so anything unreadable
        # now was written during this run.
        if damaged:
            raise RuntimeError(
                f"{len(damaged)} row(s) this run wrote cannot be read: "
                f"{damaged[:5]}. The results are wrong, so they are not "
                f"reported as a successful run."
            )

        ours = lambda counts: {  # noqa: E731
            index: count
            for index, count in counts.items()
            if index >= self._initial_sim_idx
        }
        mine, theirs = ours(inputs), ours(outputs)
        if mine != theirs:
            only_in = lambda a, b: sorted(set(a) - set(b))  # noqa: E731
            raise RuntimeError(
                f"the input and output files disagree about which simulations "
                f"ran: {only_in(mine, theirs)[:5]} have inputs and no "
                f"outputs, {only_in(theirs, mine)[:5]} the other way round. "
                f"A worker stopped between the two writes, so the results are "
                f"not reported as a successful run."
            )

        repeated = sorted(index for index, count in mine.items() if count > 1)
        beyond = sorted(index for index in mine if index >= self.number_of_simulations)
        missing = (
            []
            if self._interrupted
            # The whole range, not this run's share of it. Appending is only
            # allowed onto a checkpoint the preflight found complete, so what
            # ends up on disk has to be every simulation that was asked for.
            else sorted(set(range(self.number_of_simulations)) - set(inputs))
        )
        if missing or repeated or beyond:
            raise RuntimeError(
                f"the files do not match the simulations that ran: "
                f"{len(missing)} never written {missing[:5]}, "
                f"{len(repeated)} written more than once {repeated[:5]}, "
                f"{len(beyond)} outside the range this run claimed {beyond[:5]}. "
                f"The results are wrong, so they are not reported as a "
                f"successful run."
            )

    def __run_in_serial(self):
        """
        Runs the monte carlo simulation in serial mode.

        The root seed state is captured by ``simulate`` before this runs, so each
        simulation index derives its child seed from ``self.__root_state``.

        Returns
        -------
        None
        """
        sim_monitor = _SimMonitor(
            initial_count=self._initial_sim_idx,
            n_simulations=self.number_of_simulations,
            start_time=time(),
        )
        # Bound before the loop: a failure on the very first iteration
        # would otherwise reach the error record with no index at all.
        sim_idx = self._initial_sim_idx
        try:
            while True:
                # First statement in the loop, so it is bound before the two
                # monitor calls rather than after them. Ctrl-C in either one
                # used to leave it unbound, or holding the last completed row.
                inputs_json = ""

                if not sim_monitor.keep_simulating():
                    break
                sim_idx = sim_monitor.increment() - 1

                self.__seed_simulation(self.__child_seed(sim_idx))
                flight = self.__run_single_simulation()
                inputs_json = self.__evaluate_flight_inputs(sim_idx)
                outputs_json = self.__evaluate_flight_outputs(flight, sim_idx)

                _record_simulation(
                    self.input_file,
                    self.output_file,
                    inputs_json,
                    outputs_json,
                    (self.environment, self.rocket, self.flight),
                )
                # The pair is on disk. Cleared before the monitor call so a
                # failure there reports itself rather than reporting a row that
                # has already been committed as one that never finished.
                inputs_json = ""
                sim_monitor.print_update_status()

            sim_monitor.print_final_status()

        except KeyboardInterrupt:
            self._interrupted = True
            print("Keyboard interrupt received. Files saved.")
            self.__keep_the_inputs_that_did_not_finish(inputs_json)

        except Exception as error:
            print(f"Error on iteration {sim_monitor.count}: {error}")
            # Captured before reporting, which may fail and must not be what
            # gets recorded or raised.
            _record_failure(
                self._error_file,
                sim_idx,
                inputs_json
                or _inputs_drawn_so_far(
                    (self.environment, self.rocket, self.flight),
                    sim_idx,
                    self._export_config,
                ),
                traceback.format_exc(),
            )
            # Bare, so the handler's own line does not join the traceback.
            raise

    def __keep_the_inputs_that_did_not_finish(self, inputs_json):
        """Append the inputs of a simulation that stopped part way through.

        Best effort: an unwritable error file must not turn a clean interrupt
        into a crash.
        """
        _best_effort(
            lambda: _write_unfinished_inputs(self._error_file, inputs_json),
            "interrupted simulation inputs",
        )

    def __run_in_parallel(self, n_workers=None):
        """
        Runs the monte carlo simulation in parallel.

        The root seed state is captured by ``simulate`` before this runs and
        travels with the pickled instance, so every worker derives the same
        per-index child seed from ``self.__root_state``.

        Parameters
        ----------
        n_workers: int, optional
            Number of workers to be used. If None, the number of workers
            will be equal to the number of CPUs available. Default is None.

        Returns
        -------
        None
        """
        n_workers = self.__validate_number_of_workers(n_workers)

        print(f"Running Monte Carlo simulation with {n_workers} workers.")

        multiprocess, managers = _import_multiprocess()

        with _create_multiprocess_manager(multiprocess, managers) as manager:
            mutex = manager.Lock()
            simulation_error_event = manager.Event()
            sim_monitor = manager._SimMonitor(
                initial_count=self._initial_sim_idx,
                n_simulations=self.number_of_simulations,
                start_time=time(),
            )

            # Started workers only, and inside the try, so a ``start()`` that
            # fails part way through the fleet does not leave the ones already
            # running with nobody to clean them up.
            started_processes = []
            try:
                # Each worker derives one independent child seed per simulation
                # index (not per worker) from the shared root state: the counter
                # assigns indices and index i always seeds from __child_seed(i),
                # so the sampled inputs do not depend on the number of workers.
                # The root state is small and travels with the pickled instance,
                # so no per-index seed list is materialized or sent.
                _start_the_fleet(
                    multiprocess,
                    self.__sim_producer,
                    n_workers,
                    (sim_monitor, mutex, simulation_error_event),
                    started_processes,
                )

                _wait_for_workers(started_processes, simulation_error_event)
                _close_the_fleet_down(
                    started_processes, simulation_error_event, self.error_file
                )
                sim_monitor.print_final_status()

            except KeyboardInterrupt:
                _bring_the_fleet_down(started_processes, simulation_error_event)
                self._interrupted = True

            # Handle error from the main process
            except Exception:
                _bring_the_fleet_down(started_processes, simulation_error_event)
                self._interrupted = False
                # Bare, so the handler's own line does not join the traceback.
                raise
            finally:
                # Also best effort: this runs while an exception may be on its
                # way out, and a cleanup that raises here would replace it.
                _best_effort(
                    lambda: _stop_any_worker_still_running(started_processes),
                    "final worker shutdown",
                )

    def __validate_number_of_workers(self, n_workers):
        # os.cpu_count() is documented as possibly None, and comparing against
        # it then raises rather than falling back to a usable default.
        available = os.cpu_count() or 2
        if n_workers is not None and not _is_whole_number(n_workers):
            raise TypeError(
                f"Number of workers must be an integer, not {type(n_workers).__name__}."
            )
        if n_workers is None or n_workers > available:
            n_workers = available

        if n_workers < 2:
            raise ValueError("Number of workers must be at least 2 for parallel mode.")
        return n_workers

    def __sim_producer(self, sim_monitor, mutex, error_event):  # pylint: disable=too-many-statements
        """Simulation producer to be used in parallel by multiprocessing.

        Parameters
        ----------
        sim_monitor : _SimMonitor
            The simulation monitor object to keep track of the simulations.
        mutex : multiprocess.Lock
            The mutex to lock access to critical regions.
        error_event : multiprocess.Event
            Event signaling an error occurred during the simulation.
        """
        try:
            while True:
                # First statement in the loop, so it is bound before the claim
                # rather than after it. A claim that failed left these unassigned
                # and the handler raised UnboundLocalError over the real error;
                # a claim that failed on a later lap reported the previous row.
                sim_idx, inputs_json, outputs_json = None, "", ""

                sim_idx = _claim_next_index(sim_monitor, mutex)
                if sim_idx is None:
                    break

                self.__seed_simulation(self.__child_seed(sim_idx))
                flight = self.__run_single_simulation()
                inputs_json = self.__evaluate_flight_inputs(sim_idx)
                outputs_json = self.__evaluate_flight_outputs(flight, sim_idx)

                with _manager_mutex(mutex):
                    if error_event.is_set():
                        _report_a_cancelled_simulation(
                            self.error_file, sim_idx, inputs_json
                        )
                        break

                    _record_simulation(
                        self.input_file,
                        self.output_file,
                        inputs_json,
                        outputs_json,
                        (self.environment, self.rocket, self.flight),
                    )
                    # Same as the serial path: the pair is on disk, so a failure
                    # in the monitor call below must report itself and not the
                    # row that has just been committed.
                    inputs_json, outputs_json = "", ""
                    _forget_the_last_draw((self.environment, self.rocket, self.flight))
                    sim_monitor.print_update_status()

        except Exception:
            # Set first, so a parent waiting on the join learns why. Best effort
            # like everything below it: this is a manager proxy, the manager may
            # already be gone, and reporting must not replace what it reports.
            try:
                error_event.set()
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            details = traceback.format_exc()

            # The failure goes onto the inputs record rather than replacing it,
            # the same shape the serial path writes.
            inputs_json = inputs_json or _inputs_drawn_so_far(
                (self.environment, self.rocket, self.flight),
                sim_idx,
                self._export_config,
            )
            record = _build_error_record(sim_idx, inputs_json, details)

            try:
                with _manager_mutex(mutex):
                    with open(self.error_file, "a", encoding="utf-8") as f:
                        f.write(record)

                    # See note above: must use print() to remain visible from a
                    # multiprocessing worker process.
                    _SimMonitor.reprint(f"Error on iteration {sim_idx}:\n{details}")
            except Exception:  # pylint: disable=broad-exception-caught
                # The mutex or the error file is unreachable too. Reporting is
                # not worth losing the failure that started this.
                pass

            # The worker exits non-zero, so the parent can tell a crash from a
            # clean finish rather than only from the error event.
            raise

    def __run_single_simulation(self):
        """Runs a single simulation and returns the inputs and outputs.

        Returns
        -------
        Flight
            The flight object of the simulation.
        """
        return Flight(
            rocket=self.rocket.create_object(),
            environment=self.environment.create_object(),
            rail_length=self.flight._randomize_rail_length(),
            inclination=self.flight._randomize_inclination(),
            heading=self.flight._randomize_heading(),
            initial_solution=self.flight.initial_solution,
            terminate_on_apogee=self.flight.terminate_on_apogee,
            time_overshoot=self.flight.time_overshoot,
            # The rest of what StochasticFlight.create_object passes. Left out
            # here, a run ignored the max_time, tolerances, solver, equations of
            # motion and simulation mode the caller had set, which is what #1070
            # added StochasticFlight's own handling of them for.
            max_time=self.flight.max_time,
            max_time_step=self.flight.obj.max_time_step,
            min_time_step=self.flight.obj.min_time_step,
            rtol=self.flight.obj.rtol,
            atol=self.flight.obj.atol,
            name=self.flight.obj.name,
            equations_of_motion=self.flight.obj.equations_of_motion,
            ode_solver=self.flight.obj.ode_solver,
            simulation_mode=self.flight.obj.simulation_mode,
        )

    def estimate_confidence_interval(
        self,
        attribute,
        statistic=np.mean,
        confidence_level=0.95,
        n_resamples=1000,
        random_state=None,
    ):
        """
        Estimates the confidence interval for a specific attribute of the results
        using the bootstrap method.

        Parameters
        ----------
        attribute : str
            The name of the attribute stored in self.results (e.g., "apogee", "max_velocity").
        statistic : callable, optional
            A function that computes the statistic of interest (e.g., np.mean, np.std).
            Default is np.mean.
        confidence_level : float, optional
            The confidence level for the interval (between 0 and 1). Default is 0.95.
        n_resamples : int, optional
            The number of resamples to perform. Default is 1000.
        random_state : int or None, optional
            Seed for the random number generator to ensure reproducibility. If None (default), the random number generator is not seeded.

        Returns
        -------
        confidence_interval : ConfidenceInterval
            An object containing the low and high bounds of the confidence interval.
            Access via .low and .high.
        """
        if attribute not in self.results:
            available = list(self.results.keys())
            raise ValueError(
                f"Attribute '{attribute}' not found in results. Available attributes: {available}"
            )

        if not 0 < confidence_level < 1:
            raise ValueError(
                f"confidence_level must be between 0 and 1, got {confidence_level}"
            )

        if not isinstance(n_resamples, int) or n_resamples <= 0:
            raise ValueError(
                f"n_resamples must be a positive integer, got {n_resamples}"
            )

        data = (np.array(self.results[attribute]),)

        res = bootstrap(
            data,
            statistic,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            random_state=random_state,
            method="percentile",
        )

        return res.confidence_interval

    def simulate_convergence(
        self,
        target_attribute="apogee_time",
        target_confidence=0.95,
        tolerance=0.5,
        max_simulations=1000,
        batch_size=50,
        parallel=False,
        n_workers=None,
    ):
        """Run Monte Carlo simulations in batches until the confidence interval
        width converges within the specified tolerance or the maximum number of
        simulations is reached.

        Parameters
        ----------
        target_attribute : str
            The target attribute to track its convergence (e.g., "apogee", "apogee_time", etc.).
        target_confidence : float, optional
            The confidence level for the interval (between 0 and 1). Default is 0.95.
        tolerance : float, optional
            The desired width of the confidence interval in seconds, meters, or other units. Default is 0.5.
        max_simulations : int, optional
            The maximum number of simulations to run to avoid infinite loops. Default is 1000.
        batch_size : int, optional
            The number of simulations to run in each batch. Default is 50.
        parallel : bool, optional
            Whether to run simulations in parallel. Default is False.
        n_workers : int, optional
            The number of worker processes to use if running in parallel. Default is None.

        Returns
        -------
        confidence_interval_history : list of float
            History of confidence interval widths, one value per batch of simulations.
            The last element corresponds to the width when the simulation stopped for
            either meeting the tolerance or reaching the maximum number of simulations.
        """

        # Validate inputs up-front. Without this, a non-positive batch_size makes
        # the loop run zero new simulations every iteration and spin forever.
        if not _is_whole_number(batch_size) or batch_size <= 0:
            raise ValueError(
                f"'batch_size' must be a positive integer, got {batch_size!r}."
            )
        if not _is_whole_number(max_simulations) or max_simulations <= 0:
            raise ValueError(
                f"'max_simulations' must be a positive integer, got "
                f"{max_simulations!r}."
            )
        # bool is an int to isinstance, so True would pass as a tolerance of 1.
        if (
            isinstance(tolerance, (bool, np.bool_))
            or not isinstance(tolerance, (int, float))
            or tolerance <= 0
        ):
            raise ValueError(
                f"'tolerance' must be a positive number, got {tolerance!r}."
            )
        if not 0 < target_confidence < 1:
            raise ValueError(
                "'target_confidence' must be between 0 and 1 (exclusive), got "
                f"{target_confidence!r}."
            )

        self.import_outputs(self.filename.with_suffix(".outputs.txt"))
        confidence_interval_history = []

        while self.num_of_loaded_sims < max_simulations:
            total_sims = min(self.num_of_loaded_sims + batch_size, max_simulations)

            self.simulate(
                number_of_simulations=total_sims,
                append=True,
                include_function_data=False,
                parallel=parallel,
                n_workers=n_workers,
            )

            self.import_outputs(self.filename.with_suffix(".outputs.txt"))

            ci = self.estimate_confidence_interval(
                attribute=target_attribute,
                confidence_level=target_confidence,
            )

            width = float(ci.high - ci.low)
            confidence_interval_history.append(width)

            # A NaN width means the target attribute contains NaN values; the
            # tolerance check would never pass, so the loop would run to
            # max_simulations and silently return a NaN history. Stop and warn.
            if np.isnan(width):
                warnings.warn(
                    f"The confidence interval width for '{target_attribute}' is "
                    "NaN, likely because the attribute contains NaN values. "
                    "Stopping convergence early; check the simulation outputs.",
                    stacklevel=2,
                )
                break

            if width <= tolerance:
                break

        return confidence_interval_history

    def __evaluate_flight_inputs(self, sim_idx):
        """Evaluates the inputs of a single flight simulation.

        Parameters
        ----------
        sim_idx : int
            The index of the simulation.

        Returns
        -------
        str
            A JSON compatible dictionary with the inputs of the simulation.
        """
        inputs_dict = dict(
            item
            for d in [
                self.environment.last_rnd_dict,
                self.rocket.last_rnd_dict,
                self.flight.last_rnd_dict,
            ]
            for item in d.items()
        )
        inputs_dict["index"] = sim_idx
        return (
            json.dumps(inputs_dict, cls=RocketPyEncoder, **self._export_config) + "\n"
        )

    def __evaluate_flight_outputs(self, flight, sim_idx):
        """Evaluates the outputs of a single flight simulation.

        Parameters
        ----------
        flight : Flight
            The flight object to be evaluated.
        sim_idx : int
            The index of the simulation.

        Returns
        -------
        str
            A JSON compatible dictionary with the outputs of the simulation.
        """
        outputs_dict = {
            export_item: getattr(flight, export_item)
            for export_item in self.export_list
        }
        if self.data_collector is not None:
            for key, callback in self.data_collector.items():
                try:
                    outputs_dict[key] = callback(flight)
                except Exception as e:
                    raise ValueError(
                        f"An error was encountered running 'data_collector' callback {key}. "
                    ) from e

        # Last, so that the index a row is filed under is the one the run
        # assigned even if the collector changed under a validated one.
        outputs_dict["index"] = sim_idx

        return (
            json.dumps(outputs_dict, cls=RocketPyEncoder, **self._export_config) + "\n"
        )

    def __terminate_simulation(self):
        """
        Terminates the simulation, closes the files and prints the results.

        Returns
        -------
        None
        """
        # resave the files on self and calculate post simulation attributes
        self.input_file = self._input_file
        self.output_file = self._output_file
        self.error_file = self._error_file

        print(f"Results saved to {self._output_file}")

    def __check_export_list(self, export_list):
        """
        Checks if the export_list is valid and returns a valid list. If no
        export_list is provided, the standard list is used.

        Parameters
        ----------
        export_list : list
            The list of variables to export. If None, the default list will be
            used. Default is None.

        Returns
        -------
        list
            Validated export list.
        """
        standard_output = set(
            {
                "apogee",
                "apogee_time",
                "apogee_x",
                "apogee_y",
                "t_final",
                "x_impact",
                "y_impact",
                "impact_velocity",
                "initial_stability_margin",
                "out_of_rail_stability_margin",
                "out_of_rail_time",
                "out_of_rail_velocity",
                "max_mach_number",
                "frontal_surface_wind",
                "lateral_surface_wind",
            }
        )
        # NOTE: this list needs to be updated with Flight numerical properties
        #       example: You added the property 'inclination' to Flight.
        #       But don't add other types.
        can_be_exported = set(
            {
                "inclination",
                "heading",
                "effective1rl",
                "effective2rl",
                "out_of_rail_time",
                "out_of_rail_time_index",
                "out_of_rail_state",
                "out_of_rail_velocity",
                "rail_button1_normal_force",
                "max_rail_button1_normal_force",
                "rail_button1_shear_force",
                "max_rail_button1_shear_force",
                "rail_button2_normal_force",
                "max_rail_button2_normal_force",
                "rail_button2_shear_force",
                "max_rail_button2_shear_force",
                "out_of_rail_static_margin",
                "apogee_state",
                "apogee_time",
                "apogee_x",
                "apogee_y",
                "apogee",
                "x_impact",
                "y_impact",
                "z_impact",
                "impact_velocity",
                "impact_state",
                "parachute_events",
                "apogee_freestream_speed",
                "final_static_margin",
                "frontal_surface_wind",
                "initial_static_margin",
                "lateral_surface_wind",
                "max_acceleration",
                "max_acceleration_time",
                "max_dynamic_pressure_time",
                "max_dynamic_pressure",
                "max_mach_number_time",
                "max_mach_number",
                "max_reynolds_number_time",
                "max_reynolds_number",
                "max_speed_time",
                "max_speed",
                "max_total_pressure_time",
                "max_total_pressure",
                "t_final",
            }
        )
        if export_list:
            for attr in set(export_list):
                if not isinstance(attr, str):
                    raise TypeError("Variables in export_list must be strings.")

                # Checks if attribute is not valid
                if attr not in can_be_exported:
                    raise ValueError(
                        f"Attribute '{attr}' can not be exported. Check export_list."
                    )
        else:
            # No export list provided, using default list instead.
            export_list = standard_output

        return export_list

    def _check_data_collector(self, data_collector):
        """Check if data collector provided is a valid

        Parameters
        ----------
        data_collector : dict
            A dictionary whose keys are the names of the exported variables
            and the values are callback functions that receive a Flight object
            and returns a value of that variable
        """

        if data_collector is not None:
            if not isinstance(data_collector, dict):
                raise ValueError(
                    "Invalid 'data_collector' argument! "
                    "It must be a dict of callback functions."
                )

            for key, callback in data_collector.items():
                if not isinstance(key, str):
                    raise ValueError(
                        "Invalid 'data_collector' key! "
                        f"Keys must be strings, not {type(key).__name__}."
                    )
                if key in _RESERVED_RECORD_KEYS:
                    raise ValueError(
                        f"Invalid 'data_collector' key '{key}'! "
                        "That name is reserved for the record metadata that "
                        "pairs an inputs row with its outputs row."
                    )
                if key in self.export_list:
                    raise ValueError(
                        "Invalid 'data_collector' key! "
                        f"Variable names overwrites 'export_list' key '{key}'."
                    )
                if not callable(callback):
                    raise ValueError(
                        f"Invalid value in 'data_collector' for key '{key}'! "
                        "Values must be python callables (callback functions)."
                    )

    @property
    def input_file(self):
        """String representing the filepath of the input file"""
        return self._input_file

    @input_file.setter
    def input_file(self, value):
        """
        Setter for input_file. Sets/updates inputs_log.

        Parameters
        ----------
        value : str
            The filepath of the input file.

        Returns
        -------
        None
        """
        self._input_file = value
        self.set_inputs_log()

    @property
    def output_file(self):
        """String representing the filepath of the output file"""
        return self._output_file

    @output_file.setter
    def output_file(self, value):
        """
        Setter for output_file. Sets/updates outputs_log, num_of_loaded_sims,
        results, and processed_results.

        Parameters
        ----------
        value : str
            The filepath of the output file.

        Returns
        -------
        None
        """
        self._output_file = value
        self.set_outputs_log()
        self.set_num_of_loaded_sims()
        self.set_results()
        self.set_processed_results()

    @property
    def error_file(self):
        """String representing the filepath of the error file"""
        return self._error_file

    @error_file.setter
    def error_file(self, value):
        """
        Setter for error_file. Sets/updates errors_log.

        Parameters
        ----------
        value : str
            The filepath of the error file.

        Returns
        -------
        None
        """
        self._error_file = value
        self.set_errors_log()

    # File format helpers

    @staticmethod
    def _detect_file_format(filepath):
        """Detect file format from the file extension.

        Parameters
        ----------
        filepath : str or Path
            Path to the file.

        Returns
        -------
        str
            One of ``"jsonl"``, ``"csv"``, or ``"json"``.

        Raises
        ------
        ValueError
            If the file extension is not supported.
        """
        suffix = Path(filepath).suffix.lower()
        format_map = {".txt": "jsonl", ".csv": "csv", ".json": "json"}
        if suffix not in format_map:
            raise ValueError(
                f"Unsupported file extension '{suffix}'. "
                "Expected '.txt', '.csv', or '.json'."
            )
        return format_map[suffix]

    @staticmethod
    def _parse_csv_value(value):
        """Parse a string value from a CSV cell into its appropriate type.

        Parameters
        ----------
        value : str
            The raw string value from the CSV cell.

        Returns
        -------
        int, float, dict, list, or str
            The parsed value in its appropriate Python type.
        """
        if value == "":
            return value
        # Try parsing JSON objects/arrays
        if value.startswith(("{", "[")):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, ValueError):
                pass
        # Try numeric types
        try:
            int_val = int(value)
            # Ensure the string was truly an integer (not "1.0")
            if str(int_val) == value:
                return int_val
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value

    def _read_log_file(self, filepath):
        """Read a log file in any supported format and return a list of dicts.

        Parameters
        ----------
        filepath : str or Path
            Path to the log file. Format is detected from the extension.

        Returns
        -------
        list of dict
            A list of dictionaries, one per simulation record.
        """
        fmt = self._detect_file_format(filepath)
        result = []
        with open(filepath, mode="r", encoding="utf-8") as f:
            if fmt == "jsonl":
                for line in f:
                    line = line.strip()
                    if line:
                        result.append(json.loads(line))
            elif fmt == "json":
                content = f.read().strip()
                if content:
                    result = json.loads(content)
            elif fmt == "csv":
                reader = csv.DictReader(f)
                for row in reader:
                    result.append({k: self._parse_csv_value(v) for k, v in row.items()})
        return result

    @staticmethod
    def _write_log_to_csv(log_data, filepath, flatten=False):
        """Write a list of dicts to a CSV file.

        Parameters
        ----------
        log_data : list of dict
            The data to write. Each dict is one row.
        filepath : str or Path
            Output file path.
        flatten : bool, optional
            If True, non-scalar columns (dicts, lists) are omitted.
            If False (default), non-scalar values are serialized as JSON
            strings in the CSV cells.

        Raises
        ------
        ValueError
            If ``log_data`` is empty.
        """
        if not log_data:
            raise ValueError(
                "No data to export. Run a simulation first or import existing data."
            )
        # Collect all keys preserving insertion order
        all_keys = list(dict.fromkeys(k for row in log_data for k in row))

        if flatten:
            # Identify scalar-only keys
            scalar_keys = []
            for key in all_keys:
                if all(not isinstance(row.get(key), (dict, list)) for row in log_data):
                    scalar_keys.append(key)
            fieldnames = scalar_keys
        else:
            fieldnames = all_keys

        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in log_data:
                csv_row = {}
                for key in fieldnames:
                    value = row.get(key, "")
                    if isinstance(value, (dict, list)):
                        csv_row[key] = json.dumps(value)
                    else:
                        csv_row[key] = value
                writer.writerow(csv_row)

    def _write_log_to_json(self, log_data, filepath):
        """Write a list of dicts to a JSON file as a proper JSON array.

        Parameters
        ----------
        log_data : list of dict
            The data to write. Each dict becomes one element of the array.
        filepath : str or Path
            Output file path.

        Raises
        ------
        ValueError
            If ``log_data`` is empty.
        """
        if not log_data:
            raise ValueError(
                "No data to export. Run a simulation first or import existing data."
            )
        with open(filepath, mode="w", encoding="utf-8") as f:
            json.dump(log_data, f, cls=RocketPyEncoder, indent=2)

    # Setters for post simulation attributes

    def set_inputs_log(self):
        """
        Sets inputs_log from a file into an attribute for easy access.
        Supports .txt (JSONL), .csv, and .json file formats.

        Returns
        -------
        None
        """
        self.inputs_log = self._read_log_file(self.input_file)

    def set_outputs_log(self):
        """
        Sets outputs_log from a file into an attribute for easy access.
        Supports .txt (JSONL), .csv, and .json file formats.

        Returns
        -------
        None
        """
        self.outputs_log = self._read_log_file(self.output_file)

    def set_errors_log(self):
        """
        Sets errors_log from a file into an attribute for easy access.
        Supports .txt (JSONL), .csv, and .json file formats.

        Returns
        -------
        None
        """
        self.errors_log = self._read_log_file(self.error_file)

    def set_num_of_loaded_sims(self):
        """
        Determines the number of simulations loaded from output_file being
        currently used. Supports .txt (JSONL), .csv, and .json formats.

        Returns
        -------
        None
        """
        fmt = self._detect_file_format(self.output_file)
        with open(self.output_file, mode="r", encoding="utf-8") as outputs:
            if fmt == "jsonl":
                self.num_of_loaded_sims = sum(1 for _ in outputs)
            elif fmt == "csv":
                # Subtract 1 for the header row
                self.num_of_loaded_sims = max(0, sum(1 for _ in outputs) - 1)
            elif fmt == "json":
                content = outputs.read().strip()
                if content:
                    self.num_of_loaded_sims = len(json.loads(content))
                else:
                    self.num_of_loaded_sims = 0

    def set_results(self):
        """
        Monte Carlo results organized in a dictionary where the keys are the
        names of the saved attributes, and the values are lists with all the
        result numbers of the respective attributes. For instance:

            .. code-block:: python

                {
                    'apogee': [1000, 1001, 1002, ...],
                    'max_speed': [100, 101, 102, ...],
                }

        Returns
        -------
        None
        """
        self.results = {}
        for result in self.outputs_log:
            for key, value in result.items():
                if key in self.results:
                    self.results[key].append(value)
                else:
                    self.results[key] = [value]

    def set_processed_results(self):
        """
        Create summary statistics for scalar, real-valued results.

        Structured and non-numeric results remain available in ``results``.
        Their entry in ``processed_results`` contains five ``None`` values
        because a scalar mean, median, standard deviation, and prediction
        interval are not defined for those values.

        Returns
        -------
        None
        """
        self.processed_results = {}
        for result, values in self.results.items():
            if not values or not all(
                isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
                for value in values
            ):
                self.processed_results[result] = (None, None, None, None, None)
                continue

            mean = np.mean(values)
            stdev = np.std(values)
            pi_low = np.quantile(values, 0.025)
            pi_high = np.quantile(values, 0.975)
            median = np.median(values)
            self.processed_results[result] = (mean, median, stdev, pi_low, pi_high)

    # Import methods

    def import_outputs(self, filename=None):
        """
        Import Monte Carlo results from a file and save it into a dictionary.
        Supports .txt (JSONL), .csv, and .json file formats.

        Parameters
        ----------
        filename : str, optional
            Name or directory path to the file to be imported. If none,
            self.filename will be used with the default .outputs.txt suffix.
            Files with .csv or .json extensions are also accepted.

        Returns
        -------
        None

        Notes
        -----
        Notice that you can import the outputs, inputs, and errors from a
        file without the need to run simulations. You can use previously saved
        files to process analyze the results or to continue a simulation.
        """
        filepath = filename if filename else self.filename.with_suffix(".outputs.txt")

        try:
            with open(filepath, "r+", encoding="utf-8"):
                self.output_file = filepath
        except FileNotFoundError:
            with open(filepath, "w+", encoding="utf-8"):
                self.output_file = filepath

        print(
            f"A total of {self.num_of_loaded_sims} simulation results were "
            f"loaded from: {self.output_file}"
        )

    def import_inputs(self, filename=None):
        """
        Import Monte Carlo inputs from a file and save it into a dictionary.
        Supports .txt (JSONL), .csv, and .json file formats.

        Parameters
        ----------
        filename : str, optional
            Name or directory path to the file to be imported. If none,
            self.filename will be used with the default .inputs.txt suffix.
            Files with .csv or .json extensions are also accepted.

        Returns
        -------
        None
        """
        filepath = filename if filename else self.filename.with_suffix(".inputs.txt")

        try:
            with open(filepath, "r+", encoding="utf-8"):
                self.input_file = filepath
        except FileNotFoundError:
            with open(filepath, "w+", encoding="utf-8"):
                self.input_file = filepath

        print(f"The following input file was imported: {self.input_file}")

    def import_errors(self, filename=None):
        """
        Import Monte Carlo errors from a file and save it into a dictionary.
        Supports .txt (JSONL), .csv, and .json file formats.

        Parameters
        ----------
        filename : str, optional
            Name or directory path to the file to be imported. If none,
            self.filename will be used with the default .errors.txt suffix.
            Files with .csv or .json extensions are also accepted.

        Returns
        -------
        None
        """
        filepath = filename if filename else self.filename.with_suffix(".errors.txt")

        try:
            with open(filepath, "r+", encoding="utf-8"):
                self.error_file = filepath
        except FileNotFoundError:
            with open(filepath, "w+", encoding="utf-8"):
                self.error_file = filepath

        print(f"The following error file was imported: {self.error_file}")

    def import_results(self, filename=None):
        """
        Import Monte Carlo results from a file and save it into a dictionary.

        Parameters
        ----------
        filename : str, optional
            Name or directory path to the file to be imported. If ``None``,
            self.filename will be used.

        Returns
        -------
        None
        """
        self.import_outputs(filename=filename)
        self.import_inputs(filename=filename)
        self.import_errors(filename=filename)

    # Export methods

    def export_ellipses_to_kml(  # pylint: disable=too-many-statements
        self,
        filename,
        origin_lat,
        origin_lon,
        type="all",  # TODO: Don't use "type" as a parameter name, it's a reserved word  # pylint: disable=redefined-builtin
        resolution=100,
        colors=("ffff0000", "ff00ff00"),  # impact, apogee
    ):
        """
        Generates a KML file with the ellipses on the impact point, which can be
        used to visualize the dispersion ellipses on Google Earth.

        Parameters
        ----------
        filename : str
            Name to the KML exported file.
        origin_lat : float
            Latitude coordinate of Ellipses' geometric center, in degrees.
        origin_lon : float
            Longitude coordinate of Ellipses' geometric center, in degrees.
        type : str, optional
            Type of ellipses to be exported. Options are: 'all', 'impact' and
            'apogee'. Default is 'all', it exports both apogee and impact ellipses.
        resolution : int, optional
            Number of points to be used to draw the ellipse. Default is 100. You
            can increase this number to make the ellipse smoother, but it will
            increase the file size. It is recommended to keep it below 1000.
        colors : tuple[str, str], optional
            Colors of the ellipses. Default is ['ffff0000', 'ff00ff00'], which
            are blue and green, respectively. The first element is the color of
            the impact ellipses, and the second element is the color of the
            apogee. The colors are in hexadecimal format (aabbggrr).

        Returns
        -------
        None

        Notes
        -----
        - For further understanding on .kml files, see the official documentation:\
            https://developers.google.com/kml/documentation/kmlreference
        - You can set a pair of origin coordinates different from the launch site\
            to visualize the dispersion as if the rocket was launched from that\
            point. This is useful to visualize the dispersion ellipses in a\
            different location. However, this approach is not accurate for\
            large distances offsets, as the atmospheric conditions may change.
        """
        # TODO: The lat and lon should be optional arguments, we can get it from the env
        # Retrieve monte carlo data por apogee and impact XY position
        if type not in ["all", "impact", "apogee"]:
            raise ValueError("Invalid type. Options are 'all', 'impact' and 'apogee'")

        apogee_x = np.array([])
        apogee_y = np.array([])
        impact_x = np.array([])
        impact_y = np.array([])
        if type in ["all", "apogee"]:
            try:
                apogee_x = np.array(self.results["apogee_x"])
                apogee_y = np.array(self.results["apogee_y"])
            except KeyError as e:
                raise KeyError("No apogee data found. Skipping apogee ellipses.") from e

        if type in ["all", "impact"]:
            try:
                impact_x = np.array(self.results["x_impact"])
                impact_y = np.array(self.results["y_impact"])
            except KeyError as e:
                raise KeyError("No impact data found. Skipping impact ellipses.") from e

        apogee_ellipses, impact_ellipses = generate_monte_carlo_ellipses(
            impact_x,
            impact_y,
            apogee_x,
            apogee_y,
        )

        outputs = []

        if type in ["all", "impact"]:
            outputs.extend(
                generate_monte_carlo_ellipses_coordinates(
                    impact_ellipses, origin_lat, origin_lon, resolution=resolution
                )
            )

        if type in ["all", "apogee"]:
            outputs.extend(
                generate_monte_carlo_ellipses_coordinates(
                    apogee_ellipses, origin_lat, origin_lon, resolution=resolution
                )
            )

        if all(isinstance(output, list) for output in outputs):
            kml_data = [
                [(coord[1], coord[0]) for coord in output] for output in outputs
            ]
        else:
            raise ValueError("Each element in outputs must be a list")

        kml = simplekml.Kml()

        for i, points in enumerate(kml_data):
            if i < len(impact_ellipses):
                name = f"Impact Ellipse {i + 1}"
                ellipse_color = colors[0]  # default is blue
            else:
                name = f"Apogee Ellipse {i + 1 - len(impact_ellipses)}"
                ellipse_color = colors[1]  # default is green

            mult_ell = kml.newmultigeometry(name=name)
            mult_ell.newpolygon(
                outerboundaryis=points,
                name=name,
            )
            # Setting ellipse style
            mult_ell.tessellate = 1
            mult_ell.visibility = 1
            mult_ell.style.linestyle.color = ellipse_color
            mult_ell.style.linestyle.width = 3
            mult_ell.style.polystyle.color = simplekml.Color.changealphaint(
                80, ellipse_color
            )

        kml.newpoint(
            name="Launch Pad",
            coords=[(origin_lon, origin_lat)],
            description="Flight initial position",
        )

        kml.save(filename)

    def info(self):
        """
        Print information about the Monte Carlo simulation.

        Returns
        -------
        None
        """
        self.prints.all()

    def all_info(self):
        """
        Print and plot information about the Monte Carlo simulation and its results.

        Returns
        -------
        None
        """
        self.info()
        self.plots.ellipses()
        self.plots.all()

    def compare_info(self, other_monte_carlo):
        """
        Prints the comparison of the information  of the Monte Carlo simulation
        against the information of another Monte Carlo simulation.
        Parameters
        ----------
        other_monte_carlo : MonteCarlo
            MonteCarlo object which the current one will be compared to.
        Returns
        -------
        None
        """
        self.prints.print_comparison(other_monte_carlo)

    def compare_plots(self, other_monte_carlo):
        """
        Plots the comparison of the information of the Monte Carlo simulation
        against the information of another Monte Carlo simulation.
        Parameters
        ----------
        other_monte_carlo : MonteCarlo
            MonteCarlo object which the current one will be compared to.
        Returns
        -------
        None
        """
        self.plots.plot_comparison(other_monte_carlo)

    def compare_ellipses(self, other_monte_carlo, **kwargs):
        """
        Plots the comparison of the ellipses of the Monte Carlo simulation
        against the ellipses of another Monte Carlo simulation.
        Parameters
        ----------
        other_monte_carlo : MonteCarlo
            MonteCarlo object which the current one will be compared to.
        Returns
        -------
        None
        """
        self.plots.ellipses_comparison(other_monte_carlo, **kwargs)

    # CSV and JSON export methods

    def export_outputs_to_csv(self, filename):
        """Export simulation outputs to a CSV file.

        Each row represents one simulation. All output values are scalar,
        so the CSV is directly usable in spreadsheet applications.

        Parameters
        ----------
        filename : str
            Path to the output CSV file.

        Raises
        ------
        ValueError
            If no output data is available to export.
        """
        self._write_log_to_csv(self.outputs_log, filename)

    def export_outputs_to_json(self, filename):
        """Export simulation outputs to a JSON file as an array of objects.

        Parameters
        ----------
        filename : str
            Path to the output JSON file.

        Raises
        ------
        ValueError
            If no output data is available to export.
        """
        self._write_log_to_json(self.outputs_log, filename)

    def export_inputs_to_csv(self, filename, flatten=False):
        """Export simulation inputs to a CSV file.

        Parameters
        ----------
        filename : str
            Path to the output CSV file.
        flatten : bool, optional
            If True, columns with non-scalar values (dicts, lists) are
            omitted from the CSV. If False (default), non-scalar values
            are serialized as JSON strings within the CSV cells.

        Raises
        ------
        ValueError
            If no input data is available to export.
        """
        self._write_log_to_csv(self.inputs_log, filename, flatten=flatten)

    def export_inputs_to_json(self, filename):
        """Export simulation inputs to a JSON file as an array of objects.

        Parameters
        ----------
        filename : str
            Path to the output JSON file.

        Raises
        ------
        ValueError
            If no input data is available to export.
        """
        self._write_log_to_json(self.inputs_log, filename)

    def export_errors_to_csv(self, filename, flatten=False):
        """Export simulation errors to a CSV file.

        Parameters
        ----------
        filename : str
            Path to the output CSV file.
        flatten : bool, optional
            If True, columns with non-scalar values (dicts, lists) are
            omitted from the CSV. If False (default), non-scalar values
            are serialized as JSON strings within the CSV cells.

        Raises
        ------
        ValueError
            If no error data is available to export.
        """
        self._write_log_to_csv(self.errors_log, filename, flatten=flatten)

    def export_errors_to_json(self, filename):
        """Export simulation errors to a JSON file as an array of objects.

        Parameters
        ----------
        filename : str
            Path to the output JSON file.

        Raises
        ------
        ValueError
            If no error data is available to export.
        """
        self._write_log_to_json(self.errors_log, filename)


def _recorded_indices(label, path):
    """``({index: how many rows carry it}, [rows that carry no usable index])``.

    Damage is returned rather than raised on. Whether a torn row matters
    depends on which run wrote it, and only the caller knows the range this
    run claimed: an ``append`` run is recovering from a file some earlier run
    damaged, which is the whole reason it is appending.

    ``type(...) is int`` and not ``isinstance``: ``True`` and ``1.0`` both
    compare equal to ``1`` and would otherwise pass for it.
    """
    written, damaged = {}, []
    with open(path, mode="r", encoding="utf-8") as rows:
        for number, line in enumerate(rows, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                damaged.append(f"{label} row {number} is not readable JSON")
                continue
            index = record.get("index") if isinstance(record, dict) else None
            # isinstance is the wrong tool here, see the docstring: bool is a
            # subclass of int, so True would pass for the index 1.
            # pylint: disable-next=unidiomatic-typecheck
            if type(index) is not int or index < 0:  # noqa: E721
                damaged.append(f"{label} row {number} carries no simulation index")
                continue
            written[index] = written.get(index, 0) + 1
    return written, damaged


# Written beside the output log so a later run can tell which scheme produced
# the rows. Index shape cannot: the previous release numbered parallel runs from
# 0 as well, and those rows came from per-worker entropy, shared component seeds
# and a different sampling call sequence.
_MANIFEST_SCHEMA_VERSION = 1
_SAMPLING_SCHEME = "per-index-seed-v1"


def _manifest_path(output_file):
    """Where the manifest for a given output log lives."""
    return Path(output_file).with_suffix(".manifest.json")


def _jsonable_entropy(entropy):
    """``SeedSequence`` entropy as something ``json`` will take.

    It may be an int, a sequence of ints or an ndarray, and only the first of
    those survives ``json.dumps`` unhelped.
    """
    if isinstance(entropy, (int, np.integer)):
        return int(entropy)
    if entropy is None:
        return None
    return [int(part) for part in np.asarray(entropy).ravel()]


def _write_run_manifest(output_file, root_state, seed_chosen):
    """Record the scheme and the root the rows in this log came from.

    Best effort on the way out: a manifest that cannot be written is worth a
    warning, not the loss of a finished run. The next append refuses without it,
    which is the safe direction.
    """
    entropy, spawn_key, pool_size, base = root_state
    document = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "sampling_scheme": _SAMPLING_SCHEME,
        "log_format": "jsonl-v1",
        "seed_chosen": bool(seed_chosen),
        "root_state": {
            "entropy": _jsonable_entropy(entropy),
            "spawn_key": [int(key) for key in spawn_key],
            "pool_size": int(pool_size),
            "n_children_spawned": int(base),
        },
    }
    _best_effort(
        lambda: _manifest_path(output_file).write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8"
        ),
        "run manifest",
    )


def _read_run_manifest(output_file):
    """The manifest beside a log, or ``None`` when there is not a usable one."""
    path = _manifest_path(output_file)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def _fingerprint_from_manifest(output_file):
    """``(fingerprint, whether the seed was chosen)`` for the log's own root.

    The object that wrote the rows is usually gone by the time an append runs,
    so this is what makes the lineage check survive a fresh interpreter.
    ``None`` when there is no manifest or it does not describe a root.
    """
    manifest = _read_run_manifest(output_file)
    if manifest is None:
        return None
    recorded = manifest.get("root_state")
    if not isinstance(recorded, dict):
        return None
    try:
        # Rebuilt for its words only. n_children_spawned is read-only on a
        # SeedSequence, so it is carried across as the recorded number instead.
        root = np.random.SeedSequence(
            entropy=recorded["entropy"],
            spawn_key=tuple(recorded["spawn_key"]),
            pool_size=recorded["pool_size"],
        )
        fingerprint = (
            tuple(int(word) for word in root.generate_state(4, dtype=np.uint32)),
            tuple(int(key) for key in recorded["spawn_key"]),
            int(recorded["pool_size"]),
            int(recorded["n_children_spawned"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    return fingerprint, bool(manifest.get("seed_chosen", False))


def _refuse_a_checkpoint_from_another_scheme(output_file, resume_at):
    """Refuse rows this release cannot have written, however they are numbered.

    The previous release numbered parallel runs from 0 too, so a clean one of
    those passes every check on index shape while its rows came from per-worker
    entropy and a different sampling order. Only something written alongside
    them can tell, so a checkpoint with nothing in it is refused rather than
    guessed at.
    """
    if resume_at <= 0:
        return
    manifest = _read_run_manifest(output_file)
    if manifest is None:
        raise ValueError(
            f"cannot append to {output_file}: no {_manifest_path(output_file).name} "
            f"beside it, so the rows cannot be shown to come from this release. "
            f"Runs before per-index seeding numbered parallel results from 0 as "
            f"well, and appending to one would put two sampling schemes in one "
            f"file. Re-run the study to start a checkpoint this release owns."
        )
    scheme = manifest.get("sampling_scheme")
    version = manifest.get("schema_version")
    if scheme != _SAMPLING_SCHEME or version != _MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"cannot append to {output_file}: it was written by sampling scheme "
            f"{scheme!r} at schema version {version!r}, and this release writes "
            f"{_SAMPLING_SCHEME!r} at {_MANIFEST_SCHEMA_VERSION}. Re-run the "
            f"study rather than continuing one scheme with another."
        )


def _check_the_checkpoint_supports_appending(input_file, output_file, resume_at):
    """Everything that can be judged from the files, before a worker starts.

    ``num_of_loaded_sims`` counts lines rather than indices, so a blank line or
    a torn row moves the resume point past an index that was never run. The run
    then skips it, and a check scoped to the new range calls that a success.
    Measured: two rows plus one blank line resume at 3, plus two blanks at 4,
    while the file holds only 0 and 1 either way.

    Held here rather than after the run so a checkpoint that cannot be resumed
    costs no simulations and is left exactly as it was found.

    A file with a hole in it is refused rather than repaired. Filling holes
    needs the workers to claim from a plan instead of counting on from the end,
    which is #1075; until then, refusing loudly beats resuming in the wrong
    place quietly.
    """
    for label, path in (("inputs", input_file), ("outputs", output_file)):
        written, damaged = _recorded_indices(label, path)
        if damaged:
            raise ValueError(
                f"cannot append to {path}: {len(damaged)} row(s) cannot be "
                f"read, so the simulations they held cannot be accounted for: "
                f"{damaged[:3]}."
            )
        _refuse_a_checkpoint_that_does_not_line_up(label, path, written, resume_at)

    _refuse_a_checkpoint_from_another_scheme(output_file, resume_at)

    inputs, _ = _recorded_indices("inputs", input_file)
    outputs, _ = _recorded_indices("outputs", output_file)
    if inputs != outputs:
        raise ValueError(
            f"cannot append: the input and output files hold different "
            f"simulations, {sorted(set(inputs) - set(outputs))[:5]} against "
            f"{sorted(set(outputs) - set(inputs))[:5]}. Appending would build "
            f"on a checkpoint that is already inconsistent."
        )


def _refuse_a_checkpoint_that_does_not_line_up(label, path, written, resume_at):
    """One file's indices have to be 0..resume_at-1, with nothing repeated."""
    repeated = sorted(index for index, count in written.items() if count > 1)
    if repeated:
        raise ValueError(
            f"cannot append to {path}: {label} hold {len(repeated)} index(es) "
            f"more than once {repeated[:5]}."
        )

    indices = set(written)
    if indices == set(range(1, len(indices) + 1)) and indices:
        # The serial path used to number from 1. Named rather than reported as
        # an off-by-one, because the fix is to re-baseline, not to retry.
        raise ValueError(
            f"cannot append to {path}: the {label} are numbered from 1, which "
            f"is how versions before per-index seeding wrote serial runs. This "
            f"release numbers from 0, so the two cannot be continued into each "
            f"other. Re-run the study. Renumbering the rows would line the "
            f"indices up without lining the seeds up: those rows came from the "
            f"old sequential scheme, not from the per-index derivation this "
            f"release would use for the same indices."
        )
    if indices != set(range(resume_at)):
        missing = sorted(set(range(resume_at)) - indices)
        extra = sorted(indices - set(range(resume_at)))
        raise ValueError(
            f"cannot append to {path}: the run would start at index "
            f"{resume_at}, but the {label} are not the {resume_at} before it. "
            f"Missing {missing[:5]}, unexpected {extra[:5]}."
        )


def _is_whole_number(value):
    """A Python or NumPy integer, and not a bool.

    ``type(value) in (int, np.integer)`` rejected every NumPy integer, because
    ``type(np.int64(2))`` is ``np.int64``. ``True`` still has to go: it is an
    ``int`` to ``isinstance`` and would quietly run one simulation.
    """
    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, np.integer))


def _validate_simulation_count(number_of_simulations):
    """A count has to be a whole non-negative number, checked before any file.

    A float ran ``int(count)`` simulations and then failed the completeness
    check with a range it could never have satisfied.
    """
    if not _is_whole_number(number_of_simulations):
        raise TypeError(
            f"number_of_simulations must be an integer, not "
            f"{type(number_of_simulations).__name__}."
        )
    if number_of_simulations < 0:
        raise ValueError(
            f"number_of_simulations must not be negative, got {number_of_simulations}."
        )


_WORKER_SHUTDOWN_GRACE = 5.0
# One sleep per round, not per worker, so the fleet size does not set how soon
# an error is noticed.
_WORKER_POLL_INTERVAL = 0.05


def _report_a_cancelled_simulation(error_file, sim_idx, inputs_json):
    """Note a simulation dropped because a peer had already failed.

    Runs in a worker spawned via multiprocessing, where logging handlers
    configured in the parent are not guaranteed to be inherited (Windows
    "spawn"), so this has to print to stay visible.
    """
    _SimMonitor.reprint(f"Simulation interrupt. Files from simulation {sim_idx} saved.")
    with open(error_file, "a", encoding="utf-8") as handle:
        handle.write(_build_unfinished_record(inputs_json, "cancelled"))


def _forget_the_last_draw(models):
    """Drop what the models drew, once it is on disk or about to be replaced.

    ``_inputs_drawn_so_far`` reads these, so they have to hold the simulation in
    flight and nothing else: a row already recorded would otherwise be recovered
    again and reported as the cause of a later failure.
    """
    for model in models:
        model.last_rnd_dict = {}


def _inputs_drawn_so_far(models, sim_idx, export_config):
    """Whatever the models had drawn when a simulation stopped early.

    The whole row is only built once the flight is, so a failure inside
    ``create_object`` or ``Flight`` itself left the error row carrying a
    traceback and nothing about the inputs that produced it. Each model fills
    its own ``last_rnd_dict`` as it goes, so what did get drawn is already
    there. Marked partial: the draws that never happened are absent, not null.

    Module level for the same reason as ``_record_simulation``: the run paths
    are driven by stub objects in the tests, which carry no private methods.
    """
    try:
        drawn = dict(item for model in models for item in model.last_rnd_dict.items())
        if not drawn:
            return ""
        drawn["index"] = sim_idx
        drawn["partial_inputs"] = True
        return json.dumps(drawn, cls=RocketPyEncoder, **export_config) + "\n"
    except Exception:  # pylint: disable=broad-exception-caught
        # A diagnostic must not replace the failure it is describing.
        return ""


def _record_simulation(input_file, output_file, inputs_json, outputs_json, models):
    """Append one simulation's inputs and outputs to their logs.

    Module level rather than a method: the run paths are driven directly by
    stub objects in the tests, and a private method is not reachable on those.
    """
    with open(input_file, "a", encoding="utf-8") as f:
        f.write(inputs_json)
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(outputs_json)
    # The pair is on disk, so what produced it is no longer in flight and
    # must not be recovered again for a later failure.
    _forget_the_last_draw(models)


def _record_failure(error_file, sim_idx, inputs_json, details):
    """Append the failure being handled, the way the workers record theirs.

    Best effort, and it says so rather than going quiet: an unwritable error
    file must not become the exception the caller sees in place of the one it
    was called to record.
    """
    try:
        with open(error_file, "a", encoding="utf-8") as handle:
            handle.write(_build_error_record(sim_idx, inputs_json, details))
    except Exception as reporting_error:  # pylint: disable=broad-exception-caught
        _say_so_without_raising(
            f"The simulation failed and its error record could not be written: "
            f"{reporting_error!r}"
        )


def _build_unfinished_record(inputs_json, status):
    """One simulation that stopped before it could either fail or finish.

    Marked, because an unmarked row reads as a simulation with no error, and a
    reader cannot otherwise tell a peer's crash from a keyboard interrupt.

    Nothing in flight means nothing to report, so an empty payload stays empty
    rather than becoming a row about a simulation that never started.
    """
    if not inputs_json:
        return ""
    try:
        record = json.loads(inputs_json)
    except (TypeError, ValueError):
        record = {}
    record["status"] = status
    return json.dumps(record) + "\n"


def _build_error_record(sim_idx, inputs_json, details):
    """One failed simulation as a row: what it drew, and what went wrong.

    The traceback goes onto the inputs rather than replacing them, so the file
    the run tells the user to read says both which inputs failed and why.
    """
    try:
        record = json.loads(inputs_json) if inputs_json else {"index": sim_idx}
    except (TypeError, ValueError):
        record = {"index": sim_idx}
    record["error"] = details
    return json.dumps(record) + "\n"


def _start_the_fleet(multiprocess, target, n_workers, args, started_processes):
    """Start the workers, appending each as it starts.

    Appended one at a time so a ``start()`` that fails part way through leaves
    the caller holding exactly those already running.
    """
    for _ in range(n_workers):
        sim_producer = multiprocess.Process(target=target, args=args)
        sim_producer.start()
        started_processes.append(sim_producer)


def _close_the_fleet_down(started_processes, error_event, error_file):
    """Let the fleet finish its writes, stop the rest, then check the logs.

    The event asks workers to stop, it does not stop them, and without this
    window one part way through a write is cut off and leaves exactly the torn
    row the check reports. Not ``_bring_the_fleet_down``: that sets the event,
    which on a clean run is what the crash check reads next.
    """
    # Best effort, so that housekeeping cannot report itself in place of the
    # worker result below, which is the authoritative verdict on the run.
    _best_effort(
        lambda: _wait_for_workers(started_processes, timeout=_WORKER_SHUTDOWN_GRACE),
        "graceful worker wait",
    )
    _best_effort(
        lambda: _stop_any_worker_still_running(started_processes),
        "forced worker shutdown",
    )
    _fail_if_a_worker_did_not_finish(started_processes, error_event, error_file)


def _bring_the_fleet_down(started_processes, error_event):
    """Stop everything, without raising over the failure being handled.

    Every step is best effort, not only the event: the manager may be the thing
    that died, and a shutdown that raises would replace the failure that started
    it. Bounded window to notice, then whatever is left gets stopped.
    """
    _best_effort(error_event.set, "error notification")
    _best_effort(
        lambda: _wait_for_workers(started_processes, timeout=_WORKER_SHUTDOWN_GRACE),
        "graceful worker wait",
    )
    _best_effort(
        lambda: _stop_any_worker_still_running(started_processes),
        "forced worker shutdown",
    )


def _write_unfinished_inputs(error_file, inputs_json):
    """Module level for the same reason as ``_record_simulation``: the run paths
    are driven by stub objects in the tests, which carry no private methods."""
    with open(error_file, "a", encoding="utf-8") as f:
        f.write(_build_unfinished_record(inputs_json, "interrupted"))


def _say_so_without_raising(message):
    """Report a secondary failure in a way that cannot become the primary one.

    ``warnings.warn`` raises when the caller has turned ``RuntimeWarning`` into
    an error, which is exactly how a diagnostic ends up replacing the failure it
    describes. The filter is overridden for this one warning only.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always", RuntimeWarning)
            warnings.warn(message, RuntimeWarning, stacklevel=3)
    except Exception:  # pylint: disable=broad-exception-caught
        pass


def _best_effort(action, description):
    """Run one shutdown step, reporting a failure rather than raising it."""
    try:
        action()
    except Exception as cleanup_error:  # pylint: disable=broad-exception-caught
        _say_so_without_raising(
            f"Worker cleanup failed during {description}: {cleanup_error!r}"
        )


@contextmanager
def _manager_mutex(mutex):
    """Hold a manager lock without letting its release replace a failure.

    The proxy can die while the lock is held, and a raw release in ``finally``
    then becomes the exception the caller sees rather than the one already on
    its way out. A release that fails with nothing in flight is still raised.
    """
    mutex.acquire()
    try:
        yield
    except BaseException:
        _best_effort(mutex.release, "manager mutex release")
        raise
    else:
        mutex.release()


def _read_error_event(error_event):
    """Whether a worker reported an error, and what went wrong asking.

    An unreachable proxy is itself a reason to stop and to fail the run, so it
    reads as reported rather than letting the exception past the crash list.
    """
    try:
        return bool(error_event.is_set()), None
    except Exception as event_error:  # pylint: disable=broad-exception-caught
        return True, event_error


def _workers_that_crashed(started_processes):
    """Those already known to have exited abnormally.

    ``join(timeout=0)`` first: an unjoined child has no exit code yet, so it
    would read as ``None`` and pass for one still running.
    """
    crashed = []
    for process in started_processes:
        process.join(timeout=0)
        if process.exitcode not in (None, 0):
            crashed.append(process)
    return crashed


def _wait_for_workers(started_processes, error_event=None, timeout=None):
    """Wait for the fleet, giving up early once one of them reports an error.

    Joining each worker in turn waits on them in the order they were started. A
    worker stuck in a native call held the parent on the first join while
    another had already set the event, so neither the error nor the cleanup
    after it was ever reached.

    No overall deadline on the normal path: a run with no error and one worker
    still going is a long simulation, and that is not for this to cut short.

    The wait itself is one sleep per round rather than a blocking join on each
    worker in turn, so how soon the event is noticed does not grow with the
    fleet. Blocking 0.1 s per worker meant 100 of them delayed the next check by
    10 s.
    """
    deadline = None if timeout is None else monotonic() + timeout
    while any(process.is_alive() for process in started_processes):
        if error_event is not None and _read_error_event(error_event)[0]:
            break
        # A worker killed outright sets no event. If it died holding the shared
        # lock its siblings never return either, and only the unbounded wait
        # has nothing else to end it.
        if deadline is None and _workers_that_crashed(started_processes):
            break
        if deadline is None:
            sleep(_WORKER_POLL_INTERVAL)
            continue
        remaining = deadline - monotonic()
        if remaining <= 0:
            break
        sleep(min(_WORKER_POLL_INTERVAL, remaining))

    # Reap whatever has already finished. A worker that was gone before the
    # loop started was never joined by it, and an unjoined child has no exit
    # code yet, so the crash check downstream would read None and call it one.
    for process in started_processes:
        process.join(timeout=0)


def _join_until(processes, deadline):
    """Wait on the fleet against one clock rather than one clock each.

    A full grace per worker made the wait scale with the fleet: eight stubborn
    ones could hold the parent for eight times what the grace period promised,
    and twice over, once for terminate and once for kill.
    """
    for process in processes:
        process.join(timeout=max(0.0, deadline - monotonic()))


def _stop_any_worker_still_running(started_processes, grace=_WORKER_SHUTDOWN_GRACE):
    """Whatever is still going here is not going to stop on its own.

    Signal every worker before waiting on any of them. Terminating one and
    joining it before reaching the next let a worker that ignores the signal
    keep the rest of the fleet, the manager and the open files alive behind it.
    """
    alive = [process for process in started_processes if process.is_alive()]
    for process in alive:
        process.terminate()
    _join_until(alive, monotonic() + grace)

    # terminate is a request. SIGKILL is not, and a worker that sat through the
    # first one would otherwise keep the manager and the files open for good.
    stubborn = [process for process in alive if process.is_alive()]
    for process in stubborn:
        process.kill()
    _join_until(stubborn, monotonic() + grace)


def _fail_if_a_worker_did_not_finish(started_processes, error_event, error_file):
    """Raise unless every worker finished and none of them reported an error.

    A worker can die without ever setting the event: SystemExit, ``os._exit``, a
    segfault in a native extension, a target that will not unpickle under spawn,
    or the error handler itself failing. ``join()`` returns None whatever
    happened, so the exit status is the only thing that separates a crash from a
    clean finish.
    """
    crashed = [
        f"{sim_producer.name} exited with {sim_producer.exitcode}"
        for sim_producer in started_processes
        if sim_producer.exitcode != 0
    ]
    reported, event_error = _read_error_event(error_event)
    if event_error is not None:
        crashed.append(f"the worker error event became unavailable: {event_error!r}")
    if reported or crashed:
        raise RuntimeError(
            "An error occurred during the simulation. \n"
            + (f"Workers that did not exit cleanly: {crashed}. \n" if crashed else "")
            + f"Check the logs and error file {error_file} for more information."
        )


def _claim_next_index(sim_monitor, mutex):
    """Atomically claim the next 0-based simulation index, or ``None`` if done.

    ``keep_simulating()`` and ``increment()`` are two separate manager calls, so
    the shared ``mutex`` has to be held across both. Without it, two workers can
    each pass the ``count < number_of_simulations`` check at the tail before
    either increments, and both then claim an index, running more simulations
    than were requested (and duplicating a simulation index).
    """
    with _manager_mutex(mutex):
        if not sim_monitor.keep_simulating():
            return None
        return sim_monitor.increment() - 1


def _import_multiprocess():
    """Import the necessary modules and submodules for the
    multiprocess library.

    Returns
    -------
    tuple
        Tuple containing the imported modules.
    """
    multiprocess = import_optional_dependency("multiprocess")
    managers = import_optional_dependency("multiprocess.managers")

    return multiprocess, managers


def _create_multiprocess_manager(multiprocess, managers):
    """Creates a manager for the multiprocess control of the
    Monte Carlo simulation.

    Parameters
    ----------
    multiprocess : module
        Multiprocess module.
    managers : module
        Managing submodules of the multiprocess module.

    Returns
    -------
    MonteCarloManager
        Subclass of BaseManager with the necessary classes registered.
    """

    class MonteCarloManager(managers.BaseManager):
        """Custom manager for shared objects in the Monte Carlo simulation."""

        def __init__(self):
            super().__init__()
            self.register("Lock", multiprocess.Lock)
            self.register("Queue", multiprocess.Queue)
            self.register("Event", multiprocess.Event)
            self.register("_SimMonitor", _SimMonitor)

    return MonteCarloManager()


class _SimMonitor:
    """Class to monitor the simulation progress and display the status."""

    _last_print_len = 0

    def __init__(self, initial_count, n_simulations, start_time):
        self.initial_count = initial_count
        self.count = initial_count
        self.n_simulations = n_simulations
        self.start_time = start_time
        self.completed_count = 0

    def keep_simulating(self):
        return self.count < self.n_simulations

    def increment(self):
        self.count += 1
        return self.count

    def print_update_status(self):
        """Prints a message on the same line as the previous one and replaces
        the previous message with the new one, deleting the extra characters
        from the previous message. This method increments the completed_count
        to track how many simulations have finished (thread-safe when called
        within a mutex-protected section).

        Returns
        -------
        None
        """
        self.completed_count += 1

        average_time = (time() - self.start_time) / self.completed_count
        remaining = self.n_simulations - self.initial_count - self.completed_count
        estimated_time = int(remaining * average_time)

        msg = f"Iterations completed: {self.completed_count:06d}"
        msg += f" | Average Time per Iteration: {average_time:.3f} s"
        msg += f" | Estimated time left: {estimated_time} s"

        _SimMonitor.reprint(msg, end="\r", flush=True)

    def print_final_status(self):
        """Prints the final status of the simulation."""
        print()
        msg = f"Completed {self.count - self.initial_count} iterations."
        msg += f" In total, {self.count} simulations are exported.\n"
        msg += f"Total wall time: {time() - self.start_time:.1f} s"
        _SimMonitor.reprint(msg, end="\n", flush=True)

    @staticmethod
    def reprint(msg, end="\n", flush=True):
        """Prints a message replacing the previous line to avoid cluttering
        the terminal output during concurrent simulation progress updates.

        Parameters
        ----------
        msg : str
            Message to be printed.
        end : str, optional
            String appended after the message. Default is a new line.
        flush : bool, optional
            If True, the output is flushed. Default is True.

        Returns
        -------
        None
        """
        padding = ""
        if len(msg) < _SimMonitor._last_print_len:
            padding = " " * (_SimMonitor._last_print_len - len(msg))
        print(msg + padding, end=end, flush=flush)
        _SimMonitor._last_print_len = len(msg)
