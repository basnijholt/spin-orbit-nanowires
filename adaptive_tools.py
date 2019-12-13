import asyncio
import copy
import gzip
import math
import os
import pickle
import re
from glob import glob

import adaptive
import toolz


class Learner1D(adaptive.Learner1D):
    def save(self, folder, fname, compress=True):
        os.makedirs(folder, exist_ok=True)
        fname = os.path.join(folder, fname)
        _open = gzip.open if compress else open
        with _open(fname, "wb") as f:
            pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, folder, fname, compress=True):
        _open = gzip.open if compress else open
        fname = os.path.join(folder, fname)
        try:
            with _open(fname, "rb") as f:
                self.data = pickle.load(f)
        except FileNotFoundError:
            pass


class Learner2D(adaptive.Learner2D):
    def save(self, folder, fname, compress=True):
        os.makedirs(folder, exist_ok=True)
        fname = os.path.join(folder, fname)
        _open = gzip.open if compress else open
        with _open(fname, "wb") as f:
            pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, folder, fname, compress=True):
        _open = gzip.open if compress else open
        fname = os.path.join(folder, fname)
        try:
            with _open(fname, "rb") as f:
                self.data = pickle.load(f)
                self.refresh_stack()
        except FileNotFoundError:
            pass

    def refresh_stack(self):
        # Remove points from stack if they already exist
        for point in copy.copy(self._stack):
            if point in self.data:
                self._stack.pop(point)


class BalancingLearner(adaptive.BalancingLearner):
    def save(self, folder, fname_pattern="data_learner_{}.pickle", compress=True):
        os.makedirs(folder, exist_ok=True)
        for i, learner in enumerate(self.learners):
            fname = fname_pattern.format(f"{i:04d}")
            learner.save(folder, fname, compress=compress)

    def load(self, folder, fname_pattern="data_learner_{}.pickle", compress=True):
        for i, learner in enumerate(self.learners):
            fname = fname_pattern.format(f"{i:04d}")
            learner.load(folder, fname, compress=compress)

    async def _periodic_saver(self, runner, folder, fname_pattern, interval, compress):
        while runner.status() == "running":
            await asyncio.sleep(interval)
            self.save(folder, fname_pattern, compress)

    def start_periodic_saver(
        self,
        runner,
        folder,
        fname_pattern="data_learner_{}.pickle",
        interval=3600,
        compress=True,
    ):
        saving_coro = self._periodic_saver(
            runner, folder, fname_pattern, interval, compress
        )
        return runner.ioloop.create_task(saving_coro)


###################################################
# Running multiple runners, each on its own core. #
###################################################


def run_learner_in_ipyparallel_client(
    learner, goal, profile, folder, fname_pattern, periodic_save, timeout, save_interval
):
    import hpc05
    import zmq
    import adaptive
    import asyncio

    client = hpc05.Client(profile=profile, context=zmq.Context(), timeout=timeout)
    client[:].use_cloudpickle()
    loop = asyncio.new_event_loop()
    runner = adaptive.Runner(learner, executor=client, goal=goal, ioloop=loop)

    if periodic_save:
        try:
            learner.start_periodic_saver(runner, folder, fname_pattern, save_interval)
        except AttributeError:
            raise Exception(f"Cannot auto-save {type(learner)}.")

    loop.run_until_complete(runner.task)
    return learner


def split_learners_in_executor(
    learners,
    executor,
    profile,
    ncores,
    goal=None,
    folder="tmp-{}",
    fname_pattern="data_learner_{}.pickle",
    periodic_save=True,
    timeout=300,
    save_interval=3600,
):
    if goal is None:
        if not periodic_save:
            raise Exception("Turn on periodic saving if there is no goal.")
        goal = lambda l: False  # noqa: E731

    futs = []
    for i, _learners in enumerate(split(learners, ncores)):
        learner = BalancingLearner(_learners)
        fut = executor.submit(
            run_learner_in_ipyparallel_client,
            learner,
            goal,
            profile,
            folder.format(f"{i:04d}"),
            fname_pattern,
            periodic_save,
            timeout,
            save_interval,
        )
        futs.append(fut)
    return futs


def combine_learners_from_folders(
    learners, file_pattern="tmp-*/*", save_folder=None, save_fname_pattern=None
):
    fnames = sorted(glob(file_pattern), key=alphanum_key)
    assert len(fnames) == len(learners)
    for learner, fname in zip(learners, fnames):
        learner.load(*os.path.split(fname))

    if save_folder is not None:
        BalancingLearner(learners).save(save_folder, save_fname_pattern)


######################
# General functions. #
######################


def split(lst, n_parts):
    n = math.ceil(len(lst) / n_parts)
    return toolz.partition_all(n, lst)


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    keys = []
    for _s in re.split("([0-9]+)", s):
        try:
            keys.append(int(_s))
        except Exception:
            keys.append(_s)
    return keys
