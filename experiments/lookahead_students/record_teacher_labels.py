"""Records look-ahead labels for training a student network (experiments/lookahead_students/notes.md, step 1).

Several processes each keep many look-ahead games running on training matchups; every searched decision
becomes a label (look-ahead's pick), plus a share of the confident ones. Rerunning with the same --run continues
until --searched-labels in total.

Usage:
    python experiments/lookahead_students/record_teacher_labels.py --run runs/teacher_01 --searched-labels 50000 --actors 8 --workers-per-actor 3
    CUDA_VISIBLE_DEVICES="" python experiments/lookahead_students/record_teacher_labels.py --run runs/teacher_test --searched-labels 300 --actors 2
    ... --blind-share 0.5   # half the games blind, copies dealing from Chudledee v3's deck guesses
    # the deep judge of 2026-10-21 (ground truth: 1.8 points lost per hard decision, beta 10.8 9.7):
    ... --candidates 8 --every-move-kind --redeals 16 --horizon 24 --margin 0 --score-by value --own-view-only
"""

import argparse
import signal
import sys
import time
from pathlib import Path

from experiments.lookahead_students.label_actor import LabelActorSettings
from experiments.lookahead_students.label_recording import record_labels
from autogitcg.progression.stage_catalog import build_stage_catalog
from autogitcg.search.lookahead_settings import LookaheadSettings
from autogitcg.training.collection_pool import simulate_collections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--network", type=Path, default=Path("models/beta-10.8.pt"), help="the searching network")
    parser.add_argument("--opponent", type=Path, default=Path("models/beta-10.8.pt"))
    parser.add_argument("--searched-labels", type=int, required=True, help="stop once the run has this many searched labels")
    parser.add_argument("--actors", type=int, default=4, help="recording processes")
    parser.add_argument("--workers-per-actor", type=int, default=3, help="engine processes per recording process")
    parser.add_argument("--concurrent-games", type=int, default=48, help="real games per recording process")
    parser.add_argument("--previews", action="store_true", help="record with action previews in the observations (for preview students)")
    parser.add_argument("--unsearched-share", type=float, default=0.25, help="share of confident decisions kept as anchor labels")
    parser.add_argument("--shard-labels", type=int, default=2000)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--redeals", type=int, default=LookaheadSettings.redeals)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--confident-probability", type=float, default=LookaheadSettings.confident_probability)
    parser.add_argument("--margin", type=float, default=LookaheadSettings.margin)
    parser.add_argument("--minimum-probability", type=float, default=LookaheadSettings.minimum_probability)
    parser.add_argument("--every-move-kind", action="store_true", help="also try the most likely move of every other kind")
    parser.add_argument("--score-by", choices=("win", "value"), default=LookaheadSettings.score_by)
    parser.add_argument("--own-view-only", action="store_true", help="score copies only at the searcher's own decisions")
    parser.add_argument("--guess-builder", type=Path, default=Path("models/chudledee-v3.4.pt"),
                        help="Chudledee checkpoint that guesses blind opponents' decks")
    parser.add_argument("--blind-share", type=float, default=0.0, help="share of games where the searcher isn't told the opponent's deck")
    parser.add_argument("--seed", type=int, default=1)
    arguments = parser.parse_args()

    settings = LookaheadSettings(arguments.candidates, arguments.redeals, arguments.horizon,
                                 arguments.confident_probability, arguments.margin, arguments.minimum_probability,
                                 every_move_kind=arguments.every_move_kind, score_by=arguments.score_by,
                                 own_view_only=arguments.own_view_only)
    session = time.strftime("%Y%m%d_%H%M%S")
    template = LabelActorSettings(0, arguments.run, arguments.network, arguments.opponent, settings, arguments.workers_per_actor,
                                  arguments.concurrent_games, arguments.unsearched_share, arguments.shard_labels,
                                  arguments.seed * 7919 + int(time.time()) % 100000, session,
                                  arguments.guess_builder, arguments.blind_share, previews=arguments.previews)
    stages, _ = build_stage_catalog()
    collections = simulate_collections(stages, 64, 8, arguments.seed)

    # A plain `kill` (SIGTERM) stops like Ctrl+C, so the cleanup runs (recording processes stopped).
    # Exit code 143 (SIGTERM) marks a deliberate stop, so a wrapper that restarts crashed runs leaves it stopped.
    signal.signal(signal.SIGTERM, lambda signal_number, frame: _stop_deliberately(signal_number))
    started = time.perf_counter()
    session_searched, games, wins = 0, 0, 0

    def on_shard(entry: dict, total: int) -> None:
        nonlocal session_searched, games, wins
        session_searched += entry["searched"]
        games += entry["games"]
        wins += entry["searcher_wins"]
        per_hour = session_searched / (time.perf_counter() - started) * 3600
        message = f"{per_hour:,.0f} searched labels/h, look-ahead wins {wins / max(games, 1):.1%} of {games} games"
        print(f"{total:,} searched labels | {message}", flush=True)

    total = record_labels(template, arguments.actors, collections, arguments.searched_labels, on_shard)
    print(f"done: {total:,} searched labels")


def _stop_deliberately(signal_number: int) -> None:
    print(f"stopped by signal {signal_number}", file=sys.stderr, flush=True)
    raise SystemExit(128 + signal_number)


if __name__ == "__main__":
    main()
