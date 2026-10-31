"""Would a critic that sees the opponent's hidden hand and dice judge positions better? (docs: chudledum-v3-plan.md part D)

Plays beta 10.8 self-play games on training matchups, keeps a share of all decisions with both encodings (the
player's normal view and the privileged view), splits by game, and fine-tunes two copies of beta 10.8's value output on the
same decisions: one on each view. Reports held-out squared error against the actual final reward, overall and by
round, next to untouched beta 10.8. A clearly lower error for the privileged view is the go signal for a training-only
privileged critic in PPO.

Usage:
    python experiments/privileged_critic/compare_privileged_value.py --games 1500 --output runs/privileged_value_check.json
"""

import argparse
import copy
import dataclasses
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.bridge.messages import Decision, GameFailed, GameFinished
from autogitcg.model.combat_network_file import load_combat_network
from autogitcg.model.combat_network import CombatNetwork
from autogitcg.model.observation_batch import compact, network_dtypes, stack, upload
from autogitcg.observation.match_recorder import MatchRecorder
from autogitcg.observation.observation_encoder import ObservationEncoder
from experiments.privileged_critic.privileged_view import PRIVILEGED_GLOBAL_EXTRA, encode_privileged
from autogitcg.policies.network_policy import NetworkPolicy
from autogitcg.progression.stage_catalog import build_stage_catalog
from autogitcg.training.collection_pool import simulate_collections
from autogitcg.training.deck_pools import load_deck_pools
from autogitcg.training.game_result import final_reward
from autogitcg.training.matchup_sampler import MatchupSampler


def record_games(network: CombatNetwork, encoder: ObservationEncoder, game_count: int, keep_share: float, workers: int,
                 device: torch.device, seed: int) -> list[dict]:
    """One row per kept decision: both views (compacted), game id, player, round; reward filled at game end."""
    rng = random.Random(seed)
    stages, _ = build_stage_catalog()
    matchups = MatchupSampler(load_deck_pools(simulate_collections(stages, 16, 4, seed)), rng, adaptive=False)
    policy = NetworkPolicy(network, device)
    rows_by_game: dict[int, list[dict]] = {}
    recorders: dict[int, MatchRecorder] = {}
    finished_rows: list[dict] = []
    started = finished = 0
    with EnginePool(workers) as pool:
        while finished < game_count:
            while started < game_count and len(recorders) < 64:
                matchup = matchups.sample()
                game_id = pool.create_game(*matchup.decks, seed=rng.randrange(2**31), equipment=matchup.equipment, privileged=True)
                recorders[game_id] = MatchRecorder(matchup.decks, matchup.opponent_deck_known)
                rows_by_game[game_id] = []
                started += 1
            pending = []
            for event in pool.poll():
                if isinstance(event, Decision):
                    tracker = recorders[event.game_id].on_decision(event)
                    encoded = encoder.encode(event, tracker)
                    pending.append((event, encoded))
                    if rng.random() < keep_share:
                        rows_by_game[event.game_id].append({
                            "normal": compact(encoded.arrays),
                            "privileged": compact(encode_privileged(encoder, event, tracker).arrays),
                            "game": event.game_id, "who": event.who, "round": event.state.get("roundNumber", 0)})
                elif isinstance(event, (GameFinished, GameFailed)):
                    rows = rows_by_game.pop(event.game_id)
                    del recorders[event.game_id]
                    finished += 1
                    if isinstance(event, GameFinished):
                        for row in rows:
                            row["reward"] = final_reward(event, row["who"])
                        finished_rows.extend(rows)
            for (decision, _), choice in zip(pending, policy.choose(pending)):
                recorders[decision.game_id].on_response(decision, choice.response)
                pool.respond(decision, choice.response)
    return finished_rows


def privileged_copy(network: CombatNetwork, encoder: ObservationEncoder) -> CombatNetwork:
    """beta 10.8 with PRIVILEGED_GLOBAL_EXTRA more global inputs, starting at zero weight (so it starts as beta 10.8's value)."""
    widths = {**network.config.feature_widths, "global": network.config.feature_widths["global"] + PRIVILEGED_GLOBAL_EXTRA}
    grown = CombatNetwork(dataclasses.replace(network.config, feature_widths=widths), encoder.catalog_features.table)
    grown.load_state_dict(network.state_dict())
    return grown.to(next(network.parameters()).device)


def value_errors(network: CombatNetwork, rows: list[dict], view: str, device: torch.device) -> dict[str, float]:
    network.eval()
    errors, rounds = [], []
    with torch.no_grad():
        for start in range(0, len(rows), 512):
            chunk = rows[start:start + 512]
            batch = network_dtypes(upload(stack([row[view] for row in chunk]), device))
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                values = network(batch).value.float().cpu().numpy()
            errors.extend((values - np.array([row["reward"] for row in chunk])) ** 2)
            rounds.extend(row["round"] for row in chunk)
    errors, rounds = np.array(errors), np.array(rounds)
    return {"all": float(errors.mean()), "rounds 1-2": float(errors[rounds <= 2].mean()),
            "rounds 3-4": float(errors[(rounds >= 3) & (rounds <= 4)].mean()), "rounds 5+": float(errors[rounds >= 5].mean())}


def fine_tune_value(network: CombatNetwork, rows: list[dict], test: list[dict], view: str, epochs: int, learning_rate: float,
                    device: torch.device, seed: int) -> list[dict[str, float]]:
    """Held-out errors after each epoch (the first run, 1e-4 x 3 epochs on 1,200 games, overfit both views)."""
    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    per_epoch = []
    for _ in range(epochs):
        network.train()
        order = rng.permutation(len(rows))
        for start in range(0, len(order) - 255, 256):
            chunk = [rows[index] for index in order[start:start + 256]]
            batch = network_dtypes(upload(stack([row[view] for row in chunk]), device))
            targets = torch.tensor([row["reward"] for row in chunk], dtype=torch.float32, device=device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                values = network(batch).value.float()
            loss = (values - targets).pow(2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 0.5)
            optimizer.step()
        per_epoch.append(value_errors(network, test, view, device))
        print(view, "epoch", len(per_epoch), per_epoch[-1], flush=True)
    return per_epoch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=Path, default=Path("models/beta-10.8.pt"))
    parser.add_argument("--games", type=int, default=1500)
    parser.add_argument("--keep-share", type=float, default=0.3, help="share of decisions recorded")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(arguments.seed)
    encoder = ObservationEncoder()
    network = load_combat_network(arguments.network, encoder, device)
    started = time.perf_counter()
    rows = record_games(network, encoder, arguments.games, arguments.keep_share, arguments.workers, device, arguments.seed)
    games = sorted({row["game"] for row in rows})
    random.Random(arguments.seed).shuffle(games)
    test_games = set(games[:len(games) // 5])
    training = [row for row in rows if row["game"] not in test_games]
    test = [row for row in rows if row["game"] in test_games]
    print(f"{len(rows):,} decisions from {len(games)} games ({len(training):,} training, {len(test):,} test), "
          f"{time.perf_counter() - started:.0f}s", flush=True)

    results = {"untouched beta 10.8 (normal view)": value_errors(network, test, "normal", device)}
    for view in ("normal", "privileged"):
        critic = privileged_copy(network, encoder) if view == "privileged" else copy.deepcopy(network)
        results[f"fine-tuned on the {view} view, by epoch"] = fine_tune_value(
            critic, training, test, view, arguments.epochs, arguments.learning_rate, device, arguments.seed)
    summary = {"games": len(games), "decisions": len(rows), "test_decisions": len(test), "squared_error": results}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
