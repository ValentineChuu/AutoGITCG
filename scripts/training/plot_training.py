"""Training curves of a run, as one PNG (<run>/plots/training_curves.png), from its metrics.jsonl.

Panels, over the number of decisions trained on: benchmark win rates (against the scripted heuristic and random
play), win rates in training games (against the heuristic, past versions of itself and random play), policy
entropy, update size (KL divergence and clip fraction), learning rate, PPO epochs per update, the value output's loss
and explained variance, game health (games reaching the round limit, policy lag) and decisions per second. Good for
checking a run is learning and stable; how strong it got is better measured with scripts/evaluation/judge_duel.py.

Usage:
    python scripts/training/plot_training.py --run runs/theta_01
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be chosen first)

PANELS = [
    ("Benchmark win rate", [("evaluation_heuristic.win_rate", "vs heuristic"), ("evaluation_random.win_rate", "vs random")]),
    ("Training win rate", [("win_rates.heuristic", "vs heuristic"), ("win_rates.snapshot", "vs past self"),
                           ("win_rates.random", "vs random")]),
    ("Policy entropy", [("entropy", "entropy")]),
    ("Update size", [("approx_kl", "approx KL"), ("clip_fraction", "clip fraction")]),
    ("Learning rate", [("learning_rate", "learning rate")]),
    ("PPO epochs per update (fewer = KL early stop)", [("epochs_completed", "epochs")]),
    ("Value head", [("value_loss", "value loss"), ("explained_variance", "explained variance")]),
    ("Game health", [("round_limit_rate", "round-limit games"), ("policy_lag", "policy lag (updates)")]),
    ("Throughput", [("decisions_per_second", "decisions / s")]),
]


def _lookup(record: dict, dotted: str):
    value = record
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    arguments = parser.parse_args()

    records = [json.loads(line) for line in (arguments.run / "metrics.jsonl").read_text().splitlines() if line]
    folder = arguments.run / "plots"
    folder.mkdir(exist_ok=True)
    figure, axes = plt.subplots(len(PANELS), 1, figsize=(10, 3 * len(PANELS)), sharex=True)
    for axis, (title, series) in zip(axes, PANELS):
        for key, label in series:
            points = [(record["learner_steps"], _lookup(record, key)) for record in records]
            points = [(steps, value) for steps, value in points if value is not None]
            if points:
                axis.plot(*zip(*points), label=label, marker="o" if "evaluation" in key else None, markersize=3)
        axis.set_title(title)
        axis.grid(alpha=0.3)
        if axis.get_lines():
            axis.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("learner decisions trained on")
    figure.tight_layout()
    figure.savefig(folder / "training_curves.png", dpi=120)
    print(f"wrote {folder / 'training_curves.png'}")


if __name__ == "__main__":
    main()
