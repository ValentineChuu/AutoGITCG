"""The deck builder training loop: build decks for sampled tasks, judge each with games played by the frozen
combat network, then a PPO update; benchmark and checkpoint regularly.

Restarting with the same run directory resumes from checkpoint_latest.pt.
"""

from collections.abc import Callable
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from autogitcg.data.npc_decks import load_npc_decks
from autogitcg.deck_builder.builder_network import copy_card_knowledge
from autogitcg.deck_builder.builder_policy import BuilderPolicy
from autogitcg.deck_builder.builder_setup import BuilderSetup, load_builder_checkpoint, save_builder_checkpoint
from autogitcg.deck_builder.building_task import PVP_MODE
from autogitcg.deck_building.deck_validation import deck_problems
from autogitcg.deck_builder_training.builder_benchmark import BuilderBenchmark
from autogitcg.deck_builder_training.builder_run_storage import BuilderRunStorage
from autogitcg.deck_builder_training.builder_training_config import BuilderTrainingConfig
from autogitcg.deck_builder_training.builder_update import builder_update
from autogitcg.deck_builder_training.deck_games import deck_games
from autogitcg.deck_builder_training.deck_diversity import lineup_diversity
from autogitcg.deck_builder_training.deck_rewards import deck_result
from autogitcg.deck_builder_training.game_player_group import GamePlayerGroup
from autogitcg.deck_builder_training.game_player_process import PlayerSettings
from autogitcg.deck_builder_training.human_example_decks import human_example_decks
from autogitcg.deck_builder_training.human_imitation import HumanImitation
from autogitcg.deck_builder_training.imitation_update import imitation_pass
from autogitcg.deck_builder_training.pvp_opponents import PvpOpponents
from autogitcg.deck_builder_training.task_sampler import TaskSampler
from autogitcg.progression.stage_catalog import build_stage_catalog
from autogitcg.training.checkpoints import load_checkpoint
from autogitcg.training.collection_pool import simulate_collections
from autogitcg.training.machine_history import record_machine
from autogitcg.training.deck_pools import load_deck_pools

LATEST_CHECKPOINT = "checkpoint_latest.pt"


def train_builder(config: BuilderTrainingConfig, on_update: Callable[[dict], None] | None = None) -> None:
    """on_update: called with each update's metrics record (e.g. to report progress)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)
    setup = BuilderSetup()
    stages, _ = build_stage_catalog()
    collections = simulate_collections(stages, config.accounts, config.snapshots_per_account, config.seed)
    pools = load_deck_pools(collections)
    legal_npc_deck_ids = sorted(deck_id for deck_id, deck in load_npc_decks().items() if not deck_problems(deck))

    checkpoint_path = config.run_directory / LATEST_CHECKPOINT
    checkpoint = load_builder_checkpoint(checkpoint_path) if checkpoint_path.exists() else None
    starting_point = load_builder_checkpoint(config.initialize_from) if config.initialize_from and not checkpoint else None
    if checkpoint or starting_point:
        network = setup.network_from(checkpoint or starting_point)
    else:
        network = setup.new_network(width=config.network_width, layers=config.network_layers)
        copy_card_knowledge(network, load_checkpoint(config.combat_checkpoint)["network"])
    network.to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.learning_rate)
    progress = checkpoint["progress"] if checkpoint else {"update": 0, "decks_built": 0}
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"resumed at update {progress['update']}")
    run_seed = config.seed * 1_000_003 + progress["update"]  # a resumed run never replays its start
    rng = random.Random(run_seed)
    update_rng = np.random.default_rng(run_seed)
    generator = torch.Generator(device).manual_seed(run_seed)
    tasks = TaskSampler(collections, stages, config.stage_share, rng)
    opponents = PvpOpponents(pools, legal_npc_deck_ids, rng)
    benchmark = BuilderBenchmark(stages, pools, legal_npc_deck_ids, config.evaluation_tasks_per_mode, config.evaluation_games_per_deck)
    imitation = HumanImitation(human_example_decks(), collections, stages, setup.encoder, rng) if config.imitation_weight > 0 else None
    print(f"builder network: {sum(parameter.numel() for parameter in network.parameters()):,} weights on {device}")

    storage = BuilderRunStorage(config.run_directory)
    storage.discard_metrics_after(progress["update"])
    record_machine(config.run_directory, progress["update"])
    (config.run_directory / "options.json").write_text(json.dumps(
        {name: str(value) if isinstance(value, Path) else value for name, value in asdict(config).items()}, indent=2))
    builder = BuilderPolicy(network, setup.encoder, device)
    greedy_builder = BuilderPolicy(network, setup.encoder, device, greedy=True)
    player_settings = PlayerSettings(config.combat_checkpoint, config.workers_per_process, config.concurrent_games_per_process)
    with GamePlayerGroup(config.game_processes, player_settings) as players:
        if not storage.baseline_path.exists():
            summary, rows = benchmark.evaluate(players, benchmark.random_decks())
            storage.write_baseline(summary, rows)
            print(f"random-deck baseline: {_format_benchmark(summary)}")
        baseline = storage.read_baseline()
        while progress["update"] < config.updates:
            started = time.perf_counter()
            built = builder.build([tasks.sample() for _ in range(config.decks_per_update)], generator)
            build_seconds = time.perf_counter() - started
            schedules = [deck_games(deck.task, deck.deck, config.games_per_deck, rng, opponents.sample) for deck in built]
            outcomes = players.play([game for schedule in schedules for game in schedule])
            play_seconds = time.perf_counter() - started - build_seconds
            results, position = [], 0
            for schedule in schedules:
                results.append(deck_result(schedule, outcomes[position:position + len(schedule)]))
                position += len(schedule)
            opponents.add_built([deck.deck for deck in built if deck.task.mode == PVP_MODE])

            for group in optimizer.param_groups:
                group["lr"] = scheduled_learning_rate(config, progress["update"])
            judged = [(deck, result.reward) for deck, result in zip(built, results) if result.reward is not None]
            metrics = builder_update(network, optimizer, judged, config, device, update_rng)
            if imitation is not None:
                anchor = imitation_pass(network, imitation.steps(config.imitation_decks_per_update), config.training_token_budget,
                                        device, optimizer, update_rng, config.imitation_weight)
                metrics.update({"imitation_loss": anchor["loss"], "imitation_top1_in_deck": anchor["top1_in_deck"]})
            progress["update"] += 1
            progress["decks_built"] += len(built)
            update = progress["update"]

            record = {"update": update, "decks_built": progress["decks_built"], "learning_rate": optimizer.param_groups[0]["lr"],
                      **metrics, **_deck_metrics(built, results), "build_seconds": build_seconds, "play_seconds": play_seconds,
                      "update_seconds": time.perf_counter() - started - build_seconds - play_seconds}
            storage.write_decks(update, [_deck_row(deck, result) for deck, result in zip(built, results)])
            evaluate = config.evaluate_every and (update % config.evaluate_every == 0 or update == config.updates)  # 0: never
            if evaluate:
                summary, rows = benchmark.evaluate(players, [deck.deck for deck in greedy_builder.build(benchmark.tasks)])
                storage.write_benchmark(update, summary, rows)
                record["benchmark"] = summary
                record["benchmark_random_decks"] = baseline
            keep = update % config.keep_checkpoint_every == 0
            if evaluate or keep or update == config.updates:
                save_builder_checkpoint(checkpoint_path, network, optimizer, progress)
            if keep:
                save_builder_checkpoint(config.run_directory / f"checkpoint_{update:05d}.pt", network, optimizer, progress)
            storage.append_metrics(record)
            _print_summary(record)
            if on_update:
                on_update(record)


def scheduled_learning_rate(config: BuilderTrainingConfig, update: int) -> float:
    fraction = min(update / max(config.updates - 1, 1), 1.0)
    return config.learning_rate + fraction * (config.final_learning_rate - config.learning_rate)


def _deck_metrics(built: list, results: list) -> dict:
    metrics = {}
    for mode in sorted({deck.task.mode for deck in built}):
        mode_results = [result for deck, result in zip(built, results) if deck.task.mode == mode]
        games = sum(result.games for result in mode_results)
        rewards = [result.reward for result in mode_results if result.reward is not None]
        metrics[f"{mode}_win_rate"] = sum(result.wins for result in mode_results) / max(games, 1)
        metrics[f"{mode}_mean_reward"] = sum(rewards) / max(len(rewards), 1)
    metrics["game_failures"] = sum(result.failures for result in results)
    metrics["pick_entropy"] = sum(deck.entropy for deck in built) / len(built)
    metrics["distinct_cards_per_deck"] = sum(len(set(deck.deck.cards)) for deck in built) / len(built)
    metrics.update(lineup_diversity([(deck.task.mode, len(deck.task.collection.characters), deck.deck.characters) for deck in built]))
    return metrics


def _deck_row(deck, result) -> dict:
    return {"task": deck.task.origin, "mode": deck.task.mode, "matches_played": deck.task.collection.matches_played,
            "deck": {"characters": deck.deck.characters, "cards": deck.deck.cards}, "reward": result.reward,
            "wins": result.wins, "games": result.games, "failures": result.failures, "mean_rounds": result.mean_rounds,
            "pick_entropy": round(deck.entropy, 4)}


def _format_benchmark(summary: dict) -> str:
    return " ".join(f"{mode} win {values['win_rate']:.3f} reward {values['mean_reward']:+.3f}"
                    for mode, values in summary.items() if mode != "lineup_diversity")


def _print_summary(record: dict) -> None:
    line = (f"update {record['update']:4d} | stage win {record.get('stage_win_rate', float('nan')):.3f} "
            f"pvp win {record.get('pvp_win_rate', float('nan')):.3f} | entropy {record['pick_entropy']:.2f} "
            f"kl {record['approx_kl']:.4f} value {record['value_loss']:.3f} ev {record['explained_variance']:.2f} "
            f"| build {record['build_seconds']:.0f}s play {record['play_seconds']:.0f}s update {record['update_seconds']:.0f}s")
    if "near_complete_pvp_decks" in record:
        line += (f" | full-collection PvP lineups {record['near_complete_pvp_distinct_lineups']}/{record['near_complete_pvp_decks']}"
                 f" (top {record['near_complete_pvp_top_lineup_share']:.0%})")
    if "imitation_loss" in record:
        line += f" | human-deck imitation loss {record['imitation_loss']:.2f} top-1 {record['imitation_top1_in_deck']:.0%}"
    if record["game_failures"]:
        line += f" | failures {record['game_failures']}"
    if "benchmark" in record:
        line += (f"\n  benchmark: {_format_benchmark(record['benchmark'])}"
                 f"\n  random decks: {_format_benchmark(record['benchmark_random_decks'])}")
    print(line, flush=True)
