"""The training loop: gather rollouts from the actors, run a PPO update, broadcast weights, add league
snapshots, evaluate, checkpoint, and record everything for later analysis (see run_storage.py).

Restarting with the same run directory resumes from checkpoint_latest.pt.
"""

import dataclasses
import json
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.model.combat_network import CombatNetwork
from autogitcg.model.network_config import network_config_for
from autogitcg.observation.observation_encoder import ObservationEncoder
from autogitcg.policies.network_policy import NetworkPolicy
from autogitcg.progression.stage_catalog import build_stage_catalog
from autogitcg.training.actor_group import ActorGroup
from autogitcg.training.actor_process import ActorSettings
from autogitcg.training.benchmark_matchups import TOURNAMENT_BENCHMARK_FILE, load_benchmark_matchups
from autogitcg.policies.reply_selection import ReplySelection
from autogitcg.training.checkpoints import load_checkpoint, network_config_from, save_checkpoint
from autogitcg.training.collection_pool import simulate_collections
from autogitcg.training.evaluation import evaluate
from autogitcg.training.game_result import SPEED_PENALTY_PER_ROUND
from autogitcg.training.kickstarting import teacher_weight
from autogitcg.training.machine_history import record_machine
from autogitcg.training.opponent_league import HEURISTIC, RANDOM, REFERENCE, OpponentLeague
from autogitcg.training.ppo_config import PPOConfig
from autogitcg.training.ppo_update import ppo_update
from autogitcg.training.run_storage import RunStorage

EVALUATION_SEED = 12345
LATEST_CHECKPOINT = "checkpoint_latest.pt"


@dataclass
class TrainingOptions:
    run_directory: Path
    updates: int = 1000
    actors: int = 4
    workers_per_actor: int = 3
    concurrent_games_per_actor: int = 64
    chunk_steps: int = 1024
    evaluation_workers: int = 4
    evaluate_every: int = 20  # 0 = never (slow with --replies: the evaluation pool has only evaluation_workers)
    checkpoint_every: int = 20  # checkpoint_latest.pt; checkpoint_<update>.pt every keep_checkpoint_every regardless
    snapshot_every: int = 10
    weights_every: int = 5  # network weights alone (the release's 50-update milestones come from these)
    replay_matchups: int = 4  # evaluation matchups recorded as full replays (x2 seats x2 opponents)
    keep_checkpoint_every: int = 50
    evaluation_matchups: int = 150  # vs the heuristic; x2 deck orders x2 seats = 600 games
    random_evaluation_matchups: int = 50  # vs random play (a sanity floor)
    tournament_evaluation_matchups: int = 100  # held-out tournament decks vs the reference; x2 deck orders x2 seats = 400 games
    reference: Path | None = Path("models/alpha-2.40.pt")  # frozen alpha 2.40: mirror-pair difficulty and the tournament benchmark
    accounts: int = 64
    snapshots_per_account: int = 8
    network_width: int = 128
    network_layers: int = 4
    seed: int = 0
    initialize_from: Path | None = None  # another run's checkpoint: start a fresh run from its network and snapshots
    previews: bool = False  # the network reads action previews (fresh runs; resumed runs keep their setting)
    replies: bool = False  # it also reads the opponent's replies to each action (implies previews; fresh runs)
    preview_scale: float = 1.0  # preview and reply inputs times this (NetworkConfig.preview_scale; fresh runs)
    reply_selection: ReplySelection | None = None  # learned reply selector in the actors (policies/reply_selection.py)
    actor_device: str = ""  # "" = the second GPU when there are two or more (learner on the first), else the learner's
    teacher: Path | None = None  # kickstarting (training/kickstarting.py): a frozen network the learner imitates at first
    teacher_weight: float = 1.0  # starting weight of the imitation loss
    teacher_fade_updates: int = 100  # the weight fades linearly to 0 by this update
    distill_updates: int = 0  # the first updates are pure imitation of the teacher (no PPO loss, no KL stop)
    distill_learning_rate: float = 3e-4
    opponent_weights: dict[str, float] | None = None  # override the league's opponent mix, e.g. {"reference": 1.0}
    source_weights: dict[str, float] | None = None  # override the deck-source mix (matchup_categories.SOURCE_WEIGHTS)
    human_deck_opponents: dict[str, float] | None = None  # opponents of tournament/Chudledee decks (matchup_sampler.HUMAN_LIKE_OPPONENTS)
    tournament_from: str | None = None  # training tournament decks only from this version on (e.g. "6-4"); None = all
    card_play_bonus: float = 0.0  # immediate reward per action card played (training/card_play_bonus.py)
    card_kind_bonus: dict[str, float] | None = None  # extra reward per card kind played (data/card_kinds.py)
    exploration: float = 0.0  # learner's uniform-mixture share at action decisions (policy_distribution.sample_choices)
    speed_penalty: float = SPEED_PENALTY_PER_ROUND  # per round of a win's training reward; 0 = plain win/loss
    shaping_weight: float = 0.0  # reward shaping by the position score (training/position_potential.py)
    compile_learner: bool = True  # torch.compile the learner's forward pass: about 1.2x faster PPO updates (measured
    # 16.6 -> 13.7 s per 32k-step update on an RTX 3070), after a one-time compile of about 45 s. Actors gain nothing.


def actor_device_for(options: TrainingOptions) -> str:
    if options.actor_device:
        return options.actor_device
    return "cuda:1" if torch.cuda.device_count() > 1 else ""


def train(options: TrainingOptions, ppo: PPOConfig, on_update: Callable[[dict], None] | None = None) -> None:
    """on_update: called with each update's metrics record (e.g. to report progress)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(options.seed)
    encoder = ObservationEncoder()

    stages, _ = build_stage_catalog()
    started = time.perf_counter()
    collections = simulate_collections(stages, options.accounts, options.snapshots_per_account, options.seed)
    print(f"{len(collections)} simulated collections in {time.perf_counter() - started:.0f}s")
    evaluation_matchups = load_benchmark_matchups()[:options.evaluation_matchups]  # frozen: comparable across runs
    tournament_matchups = load_benchmark_matchups(TOURNAMENT_BENCHMARK_FILE)[:options.tournament_evaluation_matchups]

    checkpoint_path = options.run_directory / LATEST_CHECKPOINT
    checkpoint = load_checkpoint(checkpoint_path) if checkpoint_path.exists() else None
    starting_point = load_checkpoint(options.initialize_from) if options.initialize_from and not checkpoint else None
    config_source = checkpoint or starting_point
    config = network_config_from(config_source) if config_source else network_config_for(
        encoder, width=options.network_width, layers=options.network_layers)
    if (options.previews or options.replies) and not checkpoint:
        config = dataclasses.replace(config, reads_previews=True, reads_replies=options.replies, preview_scale=options.preview_scale)
    network = CombatNetwork(config, encoder.catalog_features.table).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=ppo.learning_rate)
    progress = checkpoint["progress"] if checkpoint else {"update": 0, "learner_steps": 0}
    # Seeds depend on the starting update, so a resumed run never replays the games of its start.
    run_seed = options.seed * 1_000_003 + progress["update"]
    rng = random.Random(run_seed)
    update_rng = np.random.default_rng(run_seed)
    league = OpponentLeague(rng)
    if checkpoint:
        network.load_state_dict(checkpoint["network"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        league.snapshots = checkpoint["snapshots"]
        print(f"resumed at update {progress['update']}")
    elif starting_point:
        network.load_state_dict(starting_point["network"])
        league.snapshots = starting_point.get("snapshots", [])  # its past snapshots make strong early opponents (frozen models have none)
        print(f"started from {options.initialize_from} (its update {starting_point['progress']['update']})")
    print(f"network: {sum(parameter.numel() for parameter in network.parameters()):,} weights on {device}")

    storage = RunStorage(options.run_directory)
    storage.discard_after(progress["update"])
    record_machine(options.run_directory, progress["update"])
    (options.run_directory / "options.json").write_text(json.dumps(
        {"training": {**asdict(options), "run_directory": str(options.run_directory),
                      "initialize_from": str(options.initialize_from) if options.initialize_from else None,
                      "reference": str(options.reference) if options.reference else None,
                      "teacher": str(options.teacher) if options.teacher else None},
         "ppo": asdict(ppo)}, indent=2))
    if options.compile_learner and device.type == "cuda":
        network.forward = torch.compile(network.forward, dynamic=True)  # state_dict and checkpoints are unaffected
    learner = NetworkPolicy(network, device)
    reference = None
    if options.reference is not None:
        frozen = load_checkpoint(options.reference)
        reference_network = CombatNetwork(network_config_from(frozen), encoder.catalog_features.table).to(device)
        reference_network.load_state_dict(frozen["network"])
        reference = NetworkPolicy(reference_network, device)
    teacher = None
    if options.teacher is not None:
        frozen_teacher = load_checkpoint(options.teacher)
        teacher = CombatNetwork(network_config_from(frozen_teacher), encoder.catalog_features.table).to(device)
        teacher.load_state_dict(frozen_teacher["network"])
    benchmarks = [(HEURISTIC, evaluation_matchups, None), (RANDOM, evaluation_matchups[:options.random_evaluation_matchups], None)]
    if reference is not None:
        benchmarks.append((REFERENCE, tournament_matchups, reference))
    actor_template = ActorSettings(0, options.workers_per_actor, options.concurrent_games_per_actor, options.chunk_steps,
                                   ppo.gae_lambda, config, run_seed, options.reference, actor_device_for(options),
                                   options.opponent_weights, options.exploration, options.speed_penalty, options.shaping_weight,
                                   options.source_weights, options.human_deck_opponents, options.card_play_bonus,
                                   options.card_kind_bonus, options.reply_selection, options.tournament_from)
    print(f"learner on {device}, actors on {actor_device_for(options) or device}")
    with ActorGroup(options.actors, actor_template, collections, ppo.steps_per_update) as actors, \
            EnginePool(options.evaluation_workers, previews=config.reads_previews, replies=config.reads_replies) as evaluation_pool:
        if progress["update"] == 0:
            storage.write_weights(0, network)
        actors.broadcast(progress["update"], network.state_dict(), league.random_snapshot)
        iteration_started = time.perf_counter()
        while progress["update"] < options.updates:
            steps, stats, game_records = actors.gather(ppo.steps_per_update)
            update_started = time.perf_counter()
            distilling = teacher is not None and progress["update"] < options.distill_updates
            learning_rate = (options.distill_learning_rate if distilling
                             else scheduled_learning_rate(ppo, progress["update"], options.updates))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            weight = teacher_weight(options.teacher_weight, options.teacher_fade_updates, progress["update"]) if teacher else 0.0
            if distilling:
                weight = 1.0
            metrics = ppo_update(network, optimizer, steps, ppo, device, update_rng, teacher, weight, imitation_only=distilling)
            if teacher:
                metrics["teacher_weight"] = weight
            progress["update"] += 1
            progress["learner_steps"] += len(steps)
            update = progress["update"]
            if update % options.snapshot_every == 0:
                league.add_snapshot(network, update)
            actors.broadcast(update, network.state_dict(), league.random_snapshot)

            record = {
                "update": update, "learner_steps": progress["learner_steps"], "learning_rate": learning_rate, **metrics,
                "decisions_per_second": stats.decisions / (time.perf_counter() - iteration_started),
                "update_seconds": time.perf_counter() - update_started,
                "games": dict(stats.games), "win_rates": stats.win_rates(),
                "round_limit_rate": stats.round_limit / max(sum(stats.games.values()), 1),
                "failures": sum(stats.failures.values()),
                "policy_lag": sum(update - 1 - step.policy_version for step in steps) / len(steps),
            }
            storage.write_games(update, game_records)
            if update % options.weights_every == 0:
                storage.write_weights(update, network)
            if options.evaluate_every and update % options.evaluate_every == 0:
                replays, benchmark_rows = [], []
                for opponent, matchups, opponent_network in benchmarks:
                    record[f"evaluation_{opponent}"], opponent_replays, rows, failure_reports = evaluate(
                        evaluation_pool, encoder, learner, opponent, matchups, EVALUATION_SEED,
                        options.replay_matchups, opponent_network)
                    replays += opponent_replays
                    benchmark_rows += [{**row, "opponent": opponent} for row in rows]
                    storage.write_failures(update, failure_reports)
                storage.write_replays(update, replays)
                storage.write_benchmark(update, benchmark_rows)
            if update % options.checkpoint_every == 0 or update % options.keep_checkpoint_every == 0:
                save_checkpoint(checkpoint_path, network, optimizer, league, progress)
            if update % options.keep_checkpoint_every == 0:
                save_checkpoint(options.run_directory / f"checkpoint_{update:05d}.pt", network, optimizer, league, progress)
            if stats.failure_examples:
                storage.write_failures(update, stats.failure_examples)
            storage.append_metrics(record)
            _print_summary(record, stats)
            if on_update:
                on_update(record)
            iteration_started = time.perf_counter()


def scheduled_learning_rate(ppo: PPOConfig, update: int, total_updates: int) -> float:
    """Linear from learning_rate at update 0 to final_learning_rate at the last update."""
    fraction = min(update / max(total_updates - 1, 1), 1.0)
    return ppo.learning_rate + fraction * (ppo.final_learning_rate - ppo.learning_rate)


def _print_summary(record: dict, stats) -> None:
    line = (f"update {record['update']:4d} steps {record['learner_steps']:,} | {record['decisions_per_second']:.0f} dec/s "
            f"| entropy {record['entropy']:.3f} kl {record['approx_kl']:.4f} (update {record.get('update_kl', 0):.4f}) value {record['value_loss']:.3f} "
            f"ev {record['explained_variance']:.2f} lag {record['policy_lag']:.2f} | win " + " ".join(f"{name} {rate:.2f}" for name, rate in record["win_rates"].items())
            + f" | limit {record['round_limit_rate']:.2f}")
    for name in (f"evaluation_{HEURISTIC}", f"evaluation_{RANDOM}", f"evaluation_{REFERENCE}"):
        if name in record:
            line += f" | {name} {record[name]['win_rate']:.3f}"
    if stats.failures:
        line += f" | failures {dict(stats.failures.most_common(3))}"
    print(line, flush=True)
