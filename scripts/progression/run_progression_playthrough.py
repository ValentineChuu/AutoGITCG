"""Play simulated accounts through the whole progression, from the starter deck until they own every character
(or hit the match cap).

Arms (the same account personalities in each, by seed and account number):
    chudledee     Chudledee builds a stage deck from the current collection, Chudledum plays it
    chudledee_usable_weapons  like chudledee, but Chudledee is never offered weapons its lineup can't equip
    random_decks  a random legal deck from the collection, Chudledum plays it
    estimated     fast_forward's estimated results, no engine (what the training collections assume)
    lookahead     like chudledee, but the account's side plays with look-ahead (NPC side: plain Chudledum)
Engine arms split the accounts over --processes spawned processes (observation encoding is CPU-bound).
Matches go to runs/<name>/matches_<first account>.jsonl.gz; summarize with scripts/progression/summarize_progression_playthrough.py.

Usage:
    python scripts/progression/run_progression_playthrough.py --arm chudledee --run runs/playthrough_chudledee --accounts 96
"""

import argparse
import json
import multiprocessing
import time
from pathlib import Path

from autogitcg.progression_playthrough.playthrough_shard import ShardSettings, run_shard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["chudledee", "chudledee_usable_weapons", "random_decks", "estimated", "lookahead"], required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--accounts", type=int, default=96)
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--workers-per-process", type=int, default=2)
    parser.add_argument("--max-matches", type=int, default=2500, help="per account; accounts also stop once they own every character")
    parser.add_argument("--builder", type=Path, default=Path("models/chudledee-v1.6.pt"))
    parser.add_argument("--combat", type=Path, default=Path("models/alpha-2.40.pt"))
    parser.add_argument("--seed", type=int, default=1)
    arguments = parser.parse_args()
    arguments.run.mkdir(parents=True, exist_ok=True)
    (arguments.run / "options.json").write_text(json.dumps({name: str(value) for name, value in vars(arguments).items()}, indent=2))

    processes = 1 if arguments.arm == "estimated" else arguments.processes
    ranges = [(first, len(range(first, arguments.accounts, processes))) for first in range(processes)]
    context = multiprocessing.get_context("spawn")  # CUDA and V8 state must not be forked
    messages = context.Queue()
    shards = [context.Process(target=run_shard, daemon=True, args=(ShardSettings(
        arguments.arm, arguments.run, first, processes, count, arguments.max_matches, arguments.builder, arguments.combat,
        arguments.workers_per_process, arguments.seed), messages)) for first, count in ranges]
    for shard in shards:
        shard.start()

    started, played, wins, accounts_over, failures, running = time.perf_counter(), 0, 0, 0, 0, len(shards)
    while running:
        kind, value = messages.get()
        if kind == "match":
            played += 1
            wins += value
        elif kind == "account_over":
            accounts_over += 1
        elif kind == "failure":
            failures += 1
            print(f"  failed game: {value}", flush=True)
        elif kind == "error":
            raise RuntimeError(f"a playthrough shard failed:\n{value}")
        elif kind == "shard_done":
            running -= 1
        if kind == "match" and played % 5000 == 0:
            print(f"{played} matches, {wins / played:.1%} won, {accounts_over} accounts done, {played / (time.perf_counter() - started):.1f}/s", flush=True)
    print(f"done: {played} matches in {time.perf_counter() - started:.0f}s, {wins / max(played, 1):.1%} won, {failures} failed games")


if __name__ == "__main__":
    main()
