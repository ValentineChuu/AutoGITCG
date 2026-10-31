"""Engine-backed same-yardstick comparisons and a Bradley-Terry checkpoint ladder."""

import json
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.bridge.messages import Decision, GameFailed, GameFinished
from autogitcg.model.combat_network import CombatNetwork
from autogitcg.observation.match_recorder import MatchRecorder
from autogitcg.observation.observation_encoder import ObservationEncoder
from autogitcg.policies.network_policy import NetworkPolicy
from autogitcg.progression.stage_catalog import build_stage_catalog
from autogitcg.training.checkpoints import load_checkpoint, network_config_from
from autogitcg.training.collection_pool import simulate_collections
from autogitcg.training.deck_pools import load_deck_pools
from autogitcg.training.evaluation import evaluate
from autogitcg.training.matchup_sampler import Matchup, MatchupSampler, swapped
from autogitcg.training.opponent_league import HEURISTIC

EVALUATION_SEED = 12345


def evaluation_matchups() -> list[Matchup]:
    stages, _ = build_stage_catalog()
    collections = simulate_collections(stages, 64, 8, 0)
    sampler = MatchupSampler(load_deck_pools(collections), random.Random(EVALUATION_SEED))
    return [sampler.sample() for _ in range(150)]


def load_policy(run: Path, update: int, encoder: ObservationEncoder, device: torch.device,
                config) -> NetworkPolicy:
    network = CombatNetwork(config, encoder.catalog_features.table).to(device)
    weights = torch.load(run / "weights" / f"update_{update:05d}.pt", map_location="cpu", weights_only=True)
    missing, unexpected = network.load_state_dict(weights, strict=False)
    if any(not name.startswith("win_head.") for name in missing) or unexpected:
        raise RuntimeError(f"incompatible weights {run} update {update}: missing={missing}, unexpected={unexpected}")
    return NetworkPolicy(network, device)


def play_networks(pool: EnginePool, encoder: ObservationEncoder, first: NetworkPolicy, second: NetworkPolicy,
                  matchups: list[Matchup], seed: int) -> list[dict]:
    """Four games per matchup: both deck orders and both assignments of the two networks to seats."""
    games = {}
    for number, original in enumerate(matchups):
        for deck_order, matchup in enumerate((original, swapped(original))):
            for first_seat in (0, 1):
                game_id = pool.create_game(*matchup.decks, seed=seed + number, equipment=matchup.equipment)
                games[game_id] = ({"matchup": number, "deck_order": deck_order, "first_seat": first_seat,
                                   "seed": seed + number},
                                  MatchRecorder(matchup.decks, matchup.opponent_deck_known))
    while pool.has_active_games():
        pending = {0: [], 1: []}
        for event in pool.poll():
            row, recorder = games[event.game_id]
            if isinstance(event, Decision):
                tracker = recorder.on_decision(event)
                policy_index = 0 if event.who == row["first_seat"] else 1
                pending[policy_index].append((event, encoder.encode(event, tracker)))
            elif isinstance(event, GameFinished):
                row.update(winner=event.winner, first_won=event.winner == row["first_seat"], rounds=event.rounds)
            elif isinstance(event, GameFailed):
                row["failure"] = event.message.splitlines()[0]
        for policy_index, policy in enumerate((first, second)):
            for (decision, _), choice in zip(pending[policy_index], policy.choose(pending[policy_index])):
                games[decision.game_id][1].on_response(decision, choice.response)
                pool.respond(decision, choice.response)
    return [row for row, _ in games.values()]


def _summary(rows: list[dict]) -> dict:
    finished = [row for row in rows if "failure" not in row]
    wins = sum(row["first_won"] for row in finished)
    low, high = _wilson(wins, len(finished))
    return {"games": len(rows), "finished": len(finished), "failures": len(rows) - len(finished),
            "first_wins": wins, "first_win_rate": wins / len(finished), "low": low, "high": high,
            "mean_rounds": sum(row["rounds"] for row in finished) / len(finished)}


def _wilson(successes: int, count: int) -> tuple[float, float]:
    z = 1.96
    rate = successes / count
    denominator = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / denominator
    return center - half, center + half


def bradley_terry(updates: list[int], pair_rows: list[dict]) -> list[dict]:
    index = {update: position for position, update in enumerate(updates)}
    pair_counts = defaultdict_pair_counts(pair_rows)
    ability = np.zeros(len(updates), dtype=float)
    for _ in range(50):
        gradient = np.zeros(len(updates))
        information = np.zeros((len(updates), len(updates)))
        for (first_update, second_update), (wins, games) in pair_counts.items():
            # Jeffreys half-win/half-loss smoothing keeps an undefeated or winless checkpoint finite.
            wins, games = wins + 0.5, games + 1.0
            first, second = index[first_update], index[second_update]
            probability = 1 / (1 + math.exp(-(ability[first] - ability[second])))
            gradient[first] += wins - games * probability
            gradient[second] -= wins - games * probability
            weight = games * probability * (1 - probability)
            information[first, first] += weight
            information[second, second] += weight
            information[first, second] -= weight
            information[second, first] -= weight
        delta = np.linalg.solve(information + np.eye(len(updates)) * 1e-6, gradient)
        delta -= delta.mean()
        ability += delta
        ability -= ability.mean()
        if np.max(np.abs(delta)) < 1e-9:
            break
    covariance = np.linalg.pinv(information)
    scale = 400 / math.log(10)
    return [{"update": update, "elo": float(scale * ability[position]),
             "half_width": float(1.96 * scale * math.sqrt(max(covariance[position, position], 0)))}
            for position, update in enumerate(updates)]


def defaultdict_pair_counts(rows: list[dict]) -> dict[tuple[int, int], list[int]]:
    counts = {}
    for row in rows:
        key = (row["first_update"], row["second_update"])
        value = counts.setdefault(key, [0, 0])
        value[0] += int(row["first_won"])
        value[1] += 1
    return counts


def run_evaluations(alpha_1: Path, alpha_2: Path, workers: int, ladder_matchups: int,
                    cache: Path, figures: Path, output: Path) -> dict:
    torch.manual_seed(20261014)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ObservationEncoder()
    config = network_config_from(load_checkpoint(alpha_2 / "checkpoint_latest.pt"))
    matchups = evaluation_matchups()
    final_01 = load_policy(alpha_1, 600, encoder, device, config)
    final_02 = load_policy(alpha_2, 2000, encoder, device, config)
    with EnginePool(workers) as pool:
        metrics_01, _, benchmark_01, failures_01 = evaluate(
            pool, encoder, final_01, HEURISTIC, matchups, EVALUATION_SEED)
        final_rows = play_networks(pool, encoder, final_02, final_01, matchups, EVALUATION_SEED + 100_000)

        updates = list(range(0, 2001, 200))
        policies = {update: load_policy(alpha_2, update, encoder, device, config) for update in updates}
        ladder_rows = []
        for first_index, first_update in enumerate(updates):
            for second_update in updates[first_index + 1:]:
                print(f"ladder {first_update} vs {second_update}", flush=True)
                rows = play_networks(pool, encoder, policies[first_update], policies[second_update],
                                     matchups[:ladder_matchups], EVALUATION_SEED + 200_000)
                ladder_rows += [{**row, "first_update": first_update, "second_update": second_update}
                                for row in rows if "failure" not in row]
    ratings = bradley_terry(updates, ladder_rows)
    result = {
        "device": str(device),
        "main_01_vs_heuristic": metrics_01,
        "main_01_vs_heuristic_failures": failures_01,
        "main_01_benchmark_rows": len(benchmark_01),
        "main_02_final_vs_main_01_final": _summary(final_rows),
        "ladder_matchups_per_pair": ladder_matchups,
        "ladder_games": len(ladder_rows),
        "ratings": ratings,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"final_head_to_head": final_rows, "ladder": ladder_rows}) + "\n")
    output.write_text(json.dumps(result, indent=2) + "\n")
    figures.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.errorbar([row["update"] for row in ratings], [row["elo"] for row in ratings],
                  yerr=[row["half_width"] for row in ratings], marker="o", capsize=3)
    axis.axhline(0, color="gray", linewidth=1)
    axis.set(title="alpha 2 checkpoint ladder (Bradley-Terry ratings)", xlabel="update", ylabel="rating (Elo scale)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(figures / "checkpoint_ladder.png", dpi=160)
    plt.close(figure)
    return result
