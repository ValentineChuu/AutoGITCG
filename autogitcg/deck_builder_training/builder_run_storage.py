"""Files of a deck builder run.

    metrics.jsonl                    one line per update (training metrics, benchmark when run)
    decks/update_NNNNN.jsonl.gz      every built deck: task, deck, reward, wins, games
    benchmark/random_decks.json      the random-deck baseline on the benchmark (computed once per run)
    benchmark/update_NNNNN.json      the builder's greedy decks on the benchmark, one row per task
    checkpoint_latest.pt, checkpoint_NNNNN.pt
"""

import gzip
import json
from pathlib import Path


class BuilderRunStorage:
    def __init__(self, run_directory: Path):
        self.root = run_directory
        for folder in ("decks", "benchmark"):
            (run_directory / folder).mkdir(parents=True, exist_ok=True)

    @property
    def baseline_path(self) -> Path:
        return self.root / "benchmark" / "random_decks.json"

    def append_metrics(self, record: dict) -> None:
        with (self.root / "metrics.jsonl").open("a") as metrics:
            metrics.write(json.dumps(record) + "\n")

    def discard_metrics_after(self, update: int) -> None:
        """On resume: drop lines from updates after the checkpoint (they will be redone)."""
        path = self.root / "metrics.jsonl"
        if path.exists():
            kept = [line for line in path.read_text().splitlines() if json.loads(line)["update"] <= update]
            path.write_text("".join(line + "\n" for line in kept))

    def write_decks(self, update: int, rows: list[dict]) -> None:
        with gzip.open(self.root / "decks" / f"update_{update:05d}.jsonl.gz", "wt") as output:
            for row in rows:
                output.write(json.dumps(row) + "\n")

    def write_benchmark(self, update: int, summary: dict, rows: list[dict]) -> None:
        (self.root / "benchmark" / f"update_{update:05d}.json").write_text(json.dumps({"summary": summary, "rows": rows}))

    def write_baseline(self, summary: dict, rows: list[dict]) -> None:
        self.baseline_path.write_text(json.dumps({"summary": summary, "rows": rows}))

    def read_baseline(self) -> dict:
        return json.loads(self.baseline_path.read_text())["summary"]
