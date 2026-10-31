"""Everything a run keeps for later analysis, under its run directory:

    metrics.jsonl                    one line per update (losses, win rates, throughput, evaluations)
    games/update_NNNNN.jsonl.gz      every game finished during that update (training/game_record.py)
    weights/update_NNNNN.pt          network weights every few updates
    replays/update_NNNNN.jsonl.gz    full evaluation games (states, options, probabilities)
    benchmarks/update_NNNNN.jsonl.gz one row per benchmark game (matchup, deck order, seat, result, reactions)
    failures.log                     first full report of each engine failure
"""

import gzip
import json
from pathlib import Path

import numpy as np
import torch


STORED_FOLDERS = ("games", "weights", "replays", "benchmarks")


class RunStorage:
    def __init__(self, run_directory: Path):
        self.root = run_directory
        for folder in STORED_FOLDERS:
            (self.root / folder).mkdir(parents=True, exist_ok=True)

    def append_metrics(self, record: dict) -> None:
        with (self.root / "metrics.jsonl").open("a") as metrics_file:
            metrics_file.write(json.dumps(record) + "\n")

    def write_games(self, update: int, records: list[dict]) -> None:
        _write_json_lines(self.root / "games" / f"update_{update:05d}.jsonl.gz", records)

    def write_replays(self, update: int, replays: list[dict]) -> None:
        _write_json_lines(self.root / "replays" / f"update_{update:05d}.jsonl.gz", replays)

    def write_benchmark(self, update: int, rows: list[dict]) -> None:
        _write_json_lines(self.root / "benchmarks" / f"update_{update:05d}.jsonl.gz", rows)

    def write_weights(self, update: int, network: torch.nn.Module) -> None:
        torch.save(network.state_dict(), self.root / "weights" / f"update_{update:05d}.pt")

    def write_failures(self, update: int, examples: dict[str, str]) -> None:
        if not examples:
            return
        with (self.root / "failures.log").open("a") as failures_file:
            for summary, report in examples.items():
                failures_file.write(f"=== update {update}: {summary}\n{report}\n\n")

    def discard_after(self, update: int) -> None:
        """On resume: drop everything recorded after the checkpoint's update, so update numbers stay unique."""
        metrics_path = self.root / "metrics.jsonl"
        if metrics_path.exists():
            kept = [line for line in metrics_path.read_text().splitlines() if line and json.loads(line)["update"] <= update]
            metrics_path.write_text("".join(line + "\n" for line in kept))
        for folder in STORED_FOLDERS:
            for path in (self.root / folder).glob("update_*"):
                if int(path.name.split("_")[1].split(".")[0]) > update:
                    path.unlink()


def _write_json_lines(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, separators=(",", ":"), default=_json_default) + "\n")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)}")
