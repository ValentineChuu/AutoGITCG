"""One game-player process: its own engine workers, observation encoder and a frozen combat network on the GPU.

Runs in a spawned process. It pulls single games from the shared job queue whenever it has room for more,
plays both seats with the same frozen network (sampling, as in training), and reports each outcome.
"""

import queue
import traceback
from dataclasses import dataclass
from pathlib import Path

import torch

from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.bridge.messages import Decision, GameFailed, GameFinished
from autogitcg.model.combat_network import CombatNetwork
from autogitcg.observation.match_recorder import MatchRecorder
from autogitcg.observation.observation_encoder import ObservationEncoder
from autogitcg.policies.network_policy import NetworkPolicy
from autogitcg.training.checkpoints import load_checkpoint, network_config_from
from autogitcg.deck_builder_training.deck_games import ScheduledGame
from autogitcg.deck_builder_training.game_outcome import GameOutcome


@dataclass
class PlayerSettings:
    combat_checkpoint: Path
    workers: int
    concurrent_games: int


@dataclass
class PlayerFailed:
    message: str


def run_game_player(settings: PlayerSettings, jobs, results, stop_event) -> None:
    try:
        _run(settings, jobs, results, stop_event)
    except Exception:
        results.put(PlayerFailed(traceback.format_exc()))


def _run(settings: PlayerSettings, jobs, results, stop_event) -> None:
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(settings.combat_checkpoint)
    encoder = ObservationEncoder()
    network = CombatNetwork(network_config_from(checkpoint), encoder.catalog_features.table).to(device)
    network.load_state_dict(checkpoint["network"])
    policy = NetworkPolicy(network, device)
    active: dict[int, tuple[int, MatchRecorder]] = {}  # engine game id -> (job, recorder)
    with EnginePool(settings.workers) as pool:
        while not stop_event.is_set():
            while len(active) < settings.concurrent_games:
                try:
                    job, game = jobs.get(timeout=0.5 if not active else 0)
                except queue.Empty:
                    break
                game: ScheduledGame
                game_id = pool.create_game(*game.matchup.decks, seed=game.seed, equipment=game.matchup.equipment)
                active[game_id] = (job, MatchRecorder(game.matchup.decks, game.matchup.opponent_deck_known))
            if not active:
                continue
            pending = []
            for event in pool.poll(timeout=1):
                job, recorder = active[event.game_id]
                if isinstance(event, Decision):
                    pending.append((event, encoder.encode(event, recorder.on_decision(event))))
                elif isinstance(event, GameFinished):
                    results.put(GameOutcome(job, event.winner, event.rounds))
                    del active[event.game_id]
                elif isinstance(event, GameFailed):
                    results.put(GameOutcome(job, failure=event.message.splitlines()[0][:200]))
                    del active[event.game_id]
            for (decision, _), choice in zip(pending, policy.choose(pending)):
                active[decision.game_id][1].on_response(decision, choice.response)
                pool.respond(decision, choice.response)
