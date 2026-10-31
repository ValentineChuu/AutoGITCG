"""Runs several label-recording processes (label_actor.py) until enough searched labels exist, logging every shard
they write in <run>/labels/index.jsonl. Counts from earlier sessions of the same run are included, so a stopped
recording continues where it left off."""

import multiprocessing
import queue
import time
from dataclasses import replace
from pathlib import Path

from experiments.lookahead_students.label_actor import LabelActorSettings, run_label_actor
from experiments.lookahead_students.label_shards import append_index, read_index
from autogitcg.training.collection_pool import Collection
from autogitcg.training.open_file_limit import raise_open_file_limit


def recorded_searched_labels(run: Path) -> int:
    return sum(entry["searched"] for entry in read_index(run))


def record_labels(template: LabelActorSettings, actor_count: int, collections: list[Collection], target_searched: int,
                  on_shard=lambda entry, total: None) -> int:
    """Returns the run's searched-label total once it reaches `target_searched`. on_shard(entry, total) per shard."""
    total = recorded_searched_labels(template.run)
    if total >= target_searched:
        return total
    raise_open_file_limit()
    context = multiprocessing.get_context("spawn")  # CUDA and V8 state must not be forked
    stop_event = context.Event()
    progress_queue = context.Queue()
    processes = [context.Process(target=run_label_actor, daemon=True, args=(
                     replace(template, actor_index=index, seed=template.seed * 1000 + index), collections, progress_queue, stop_event))
                 for index in range(actor_count)]
    for process in processes:
        process.start()
    try:
        while total < target_searched:
            try:
                entry = progress_queue.get(timeout=60)
            except queue.Empty:
                dead = [index for index, process in enumerate(processes) if not process.is_alive()]
                if dead:
                    codes = [processes[index].exitcode for index in dead]  # negative: killed by that signal
                    raise RuntimeError(f"label actors {dead} exited without reporting an error (exit codes {codes})")
                continue
            if "error" in entry:
                raise RuntimeError(f"label actor {entry['actor']} failed:\n{entry['error']}")
            append_index(template.run, entry)
            total += entry["searched"]
            on_shard(entry, total)
    finally:
        stop_event.set()
        deadline = time.monotonic() + 30
        while any(process.is_alive() for process in processes) and time.monotonic() < deadline:
            try:
                progress_queue.get(timeout=0.2)
            except queue.Empty:
                pass
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    return total
