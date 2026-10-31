"""One rollout actor: its own engine workers, encoder and GPU copies of the learner, a snapshot and the reference.

Runs in a spawned process. It waits for the first weights, then plays continuously, sending finished steps
in chunks and switching to newer weights whenever the learner broadcasts them.
"""

import queue
import random
import traceback
from dataclasses import dataclass
from pathlib import Path

import torch

from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.model.combat_network import CombatNetwork
from autogitcg.model.network_config import NetworkConfig
from autogitcg.observation.observation_encoder import ObservationEncoder
from autogitcg.policies.network_policy import NetworkPolicy
from autogitcg.policies.reply_selection import ReplySelection
from autogitcg.training.checkpoints import load_checkpoint, network_config_from
from autogitcg.training.collection_pool import Collection
from autogitcg.training.deck_pools import load_deck_pools
from autogitcg.training.game_result import SPEED_PENALTY_PER_ROUND
from autogitcg.training.matchup_sampler import MatchupSampler
from autogitcg.training.opponent_league import OpponentLeague
from autogitcg.training.rollout_collector import RolloutCollector


@dataclass
class ActorSettings:
    actor_index: int
    workers: int
    concurrent_games: int
    chunk_steps: int
    gae_lambda: float
    network_config: NetworkConfig
    seed: int
    reference: Path | None = None  # frozen opponent for mirror-pair skill measurement (models/alpha-2.40.pt)
    device: str = ""  # e.g. "cuda:1" so game play doesn't queue behind the learner's GPU work; "" = default GPU (or CPU)
    opponent_weights: dict[str, float] | None = None  # override opponent_league.OPPONENT_WEIGHTS (diagnostics)
    exploration: float = 0.0  # learner's uniform-mixture share at action decisions (policy_distribution.sample_choices)
    speed_penalty: float = SPEED_PENALTY_PER_ROUND  # per round of a win's reward; 0 = plain win/loss
    shaping_weight: float = 0.0  # reward shaping by the position score (training/position_potential.py)
    source_weights: dict[str, float] | None = None  # override matchup_categories.SOURCE_WEIGHTS
    human_deck_opponents: dict[str, float] | None = None  # override matchup_sampler.HUMAN_LIKE_OPPONENTS
    card_play_bonus: float = 0.0  # immediate reward per action card played (training/card_play_bonus.py)
    card_kind_bonus: dict[str, float] | None = None  # extra per card kind (data/card_kinds.py)
    reply_selection: ReplySelection | None = None  # learned reply selector (policies/reply_selection.py); None = every action
    tournament_from: str | None = None  # training tournament decks only from this version on (e.g. "6-4"); None = all


@dataclass
class WeightsMessage:
    version: int
    learner: dict[str, torch.Tensor]
    snapshot: tuple[int, dict[str, torch.Tensor]] | None  # (update it was taken at, state dict)


@dataclass
class StepsMessage:
    actor_index: int
    steps: list
    stats: object
    game_records: list[dict]


@dataclass
class ActorFailed:
    actor_index: int
    message: str


def run_actor(settings: ActorSettings, collections: list[Collection], weights_queue, results_queue, stop_event) -> None:
    try:
        _run(settings, collections, weights_queue, results_queue, stop_event)
    except Exception:
        results_queue.put(ActorFailed(settings.actor_index, traceback.format_exc()))


def _run(settings: ActorSettings, collections: list[Collection], weights_queue, results_queue, stop_event) -> None:
    torch.set_num_threads(1)
    device = torch.device(settings.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rng = random.Random(settings.seed)
    encoder = ObservationEncoder()
    matchups = MatchupSampler(load_deck_pools(collections, settings.tournament_from), rng, source_weights=settings.source_weights,
                             human_deck_opponents=settings.human_deck_opponents)
    league = OpponentLeague(rng, settings.opponent_weights, has_reference=settings.reference is not None)
    learner_network = CombatNetwork(settings.network_config, encoder.catalog_features.table).to(device)
    snapshot_network = CombatNetwork(settings.network_config, encoder.catalog_features.table).to(device)
    reference_policy = None
    if settings.reference is not None:
        frozen = load_checkpoint(settings.reference)
        reference_network = CombatNetwork(network_config_from(frozen), encoder.catalog_features.table).to(device)
        reference_network.load_state_dict(frozen["network"])
        reference_policy = NetworkPolicy(reference_network, device)

    with EnginePool(settings.workers, previews=settings.network_config.reads_previews,
                    replies=settings.network_config.reads_replies) as pool:
        collector = RolloutCollector(pool, encoder, NetworkPolicy(learner_network, device, exploration=settings.exploration),
                                     NetworkPolicy(snapshot_network, device),
                                     matchups, league, settings.concurrent_games, settings.gae_lambda, rng, reference_policy,
                                     settings.reply_selection)
        collector.actor_index = settings.actor_index
        collector.speed_penalty = settings.speed_penalty
        collector.shaping_weight = settings.shaping_weight
        collector.card_play_bonus = settings.card_play_bonus
        collector.card_kind_bonus = settings.card_kind_bonus

        def apply(weights: WeightsMessage) -> None:
            learner_network.load_state_dict(weights.learner)
            league.snapshots = [weights.snapshot] if weights.snapshot is not None else []
            if weights.snapshot is not None:
                collector.snapshot_version, snapshot_state = weights.snapshot
                snapshot_network.load_state_dict(snapshot_state)
            collector.policy_version = weights.version

        apply(weights_queue.get())
        while not stop_event.is_set():
            latest = None
            try:
                while True:
                    latest = weights_queue.get_nowait()
            except queue.Empty:
                pass
            if latest is not None:
                apply(latest)
            steps, stats, records = collector.collect(settings.chunk_steps)
            while not stop_event.is_set():
                try:
                    results_queue.put(StepsMessage(settings.actor_index, steps, stats, records), timeout=1)
                    break
                except queue.Full:
                    continue
