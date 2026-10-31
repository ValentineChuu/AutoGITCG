"""Starts the rollout actor processes, broadcasts weights to them and gathers their steps.

The results queue holds about one update's worth of steps, so actors keep playing through a learner update
but can't run further ahead than that.
"""

import multiprocessing
import queue
import time
from dataclasses import replace

from autogitcg.training.actor_process import ActorFailed, ActorSettings, StepsMessage, WeightsMessage, run_actor
from autogitcg.training.collection_pool import Collection
from autogitcg.training.open_file_limit import raise_open_file_limit
from autogitcg.training.rollout_stats import RolloutStats
from autogitcg.training.trajectory import Step


class ActorGroup:
    def __init__(self, actor_count: int, template: ActorSettings, collections: list[Collection], steps_per_update: int):
        raise_open_file_limit()  # before spawning, so the actors inherit it too
        context = multiprocessing.get_context("spawn")  # CUDA and V8 state must not be forked
        self.stop_event = context.Event()
        self.results_queue = context.Queue(maxsize=steps_per_update // template.chunk_steps + actor_count)
        self.weights_queues = [context.Queue() for _ in range(actor_count)]
        self.processes = [
            context.Process(target=run_actor, daemon=True, args=(
                replace(template, actor_index=index, seed=template.seed * 1000 + index),
                collections, self.weights_queues[index], self.results_queue, self.stop_event))
            for index in range(actor_count)
        ]
        for process in self.processes:
            process.start()

    def broadcast(self, version: int, learner_state: dict, snapshot_for_actor) -> None:
        """snapshot_for_actor(): a (possibly different) league snapshot for each actor, or None."""
        cpu_state = {name: tensor.detach().cpu() for name, tensor in learner_state.items()}
        for weights_queue in self.weights_queues:
            weights_queue.put(WeightsMessage(version, cpu_state, snapshot_for_actor()))

    def gather(self, step_count: int) -> tuple[list[Step], RolloutStats, list[dict]]:
        steps: list[Step] = []
        stats = RolloutStats()
        records: list[dict] = []
        while len(steps) < step_count:
            try:
                message = self.results_queue.get(timeout=60)
            except queue.Empty:
                dead = [index for index, process in enumerate(self.processes) if not process.is_alive()]
                if dead:
                    raise RuntimeError(f"actors {dead} exited without reporting an error")
                continue
            if isinstance(message, ActorFailed):
                raise RuntimeError(f"actor {message.actor_index} failed:\n{message.message}")
            assert isinstance(message, StepsMessage)
            steps.extend(message.steps)
            stats.merge(message.stats)
            records.extend(message.game_records)
        return steps, stats, records

    def close(self, seconds: float = 30) -> None:
        """Ask the actors to stop, keep draining their results meanwhile (a process can't exit while its queued
        data is unread), and terminate any still running after `seconds` in total. Nothing is read after a
        terminate: a process killed mid-send leaves half a message, and reading it would block forever."""
        self.stop_event.set()
        deadline = time.monotonic() + seconds
        while any(process.is_alive() for process in self.processes) and time.monotonic() < deadline:
            try:
                self.results_queue.get(timeout=0.2)
            except queue.Empty:
                pass
        for process in self.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)

    def __enter__(self) -> "ActorGroup":
        return self

    def __exit__(self, *exception_info) -> None:
        self.close()
