"""One label-recording process: its own engine workers, encoder and network copies, keeping `concurrent_games`
look-ahead games running on training matchups and writing their labels to shards.

In a `blind_share` of the games the searcher isn't told the opponent's deck list, and copies deal the opponent's
hidden cards from Chudledee's guesses for their lineup (search/deck_guess.py); in the rest, and whenever the
lineup can't be guessed (not 3 different characters a player can own), it is told the list and copies deal from it. Matchups come from the training sampler, so the locked
test lineups never appear. Runs in a spawned process; reports each written shard to `progress_queue`.
"""

import dataclasses
import random
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import torch

from autogitcg.model.combat_network_file import load_combat_network

from autogitcg.bridge.engine_pool import EnginePool
from experiments.lookahead_students.label_collector import LabelCollector
from experiments.lookahead_students.label_shards import LABEL_FOLDER, write_shard
from autogitcg.observation.observation_encoder import ObservationEncoder
from autogitcg.policies.network_policy import NetworkPolicy
from autogitcg.search.deck_guess import DeckGuesser
from autogitcg.search.lookahead_settings import LookaheadSettings
from autogitcg.search.lookahead_table import LookaheadTable
from autogitcg.training.collection_pool import Collection
from autogitcg.training.deck_pools import load_deck_pools
from autogitcg.training.matchup_sampler import MatchupSampler


@dataclass
class LabelActorSettings:
    actor_index: int
    run: Path
    network: Path  # the searching network (also plays the copies)
    opponent: Path
    lookahead: LookaheadSettings
    workers: int
    concurrent_games: int
    unsearched_share: float
    shard_labels: int  # labels per shard file
    seed: int
    session: str  # names this recording session's shards, so a rerun never overwrites earlier ones
    guess_builder: Path | None = None  # Chudledee checkpoint for blind games' deck guesses
    blind_share: float = 0.0
    guesses_per_lineup: int = 4
    device: str = ""  # "" = the GPUs in turn by actor index, or CPU when there is none
    previews: bool = False  # games carry action previews, so the labels' observations include them (preview students)


def run_label_actor(settings: LabelActorSettings, collections: list[Collection], progress_queue, stop_event) -> None:
    try:
        _run(settings, collections, progress_queue, stop_event)
    except Exception:
        progress_queue.put({"actor": settings.actor_index, "error": traceback.format_exc()})


def default_device(actor_index: int) -> str:
    """Recording processes take turns over the GPUs (CPU when there is none)."""
    return f"cuda:{actor_index % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu"


def _run(settings: LabelActorSettings, collections: list[Collection], progress_queue, stop_event) -> None:
    torch.set_num_threads(1)
    device = torch.device(settings.device or default_device(settings.actor_index))
    rng = random.Random(settings.seed)
    encoder = ObservationEncoder()
    matchups = MatchupSampler(load_deck_pools(collections), rng, adaptive=False)
    network = load_combat_network(settings.network, encoder, device)
    opponent_network = network if settings.opponent == settings.network else load_combat_network(settings.opponent, encoder, device)
    collector = LabelCollector(settings.unsearched_share, rng)
    guesser = (DeckGuesser(settings.guess_builder, settings.guesses_per_lineup, settings.seed, device)
               if settings.guess_builder and settings.blind_share > 0 else None)
    labels, games, searcher_wins, shard_number = [], 0, 0, 0
    started = time.perf_counter()

    previews = settings.previews or network.config.reads_previews or opponent_network.config.reads_previews
    replies = network.config.reads_replies or opponent_network.config.reads_replies
    with EnginePool(settings.workers, previews=previews, replies=replies) as pool:
        table = LookaheadTable(pool, encoder, NetworkPolicy(network, device), NetworkPolicy(network, device, greedy=True),
                               NetworkPolicy(opponent_network, device), settings.lookahead, settings.seed,
                               deck_guesses=guesser, on_searcher_move=collector.on_searcher_move)
        while not stop_event.is_set():
            while table.game_count < settings.concurrent_games:
                matchup = matchups.sample()
                searcher_seat = rng.randrange(2)
                blind = (guesser is not None and rng.random() < settings.blind_share
                         and guesser.can_guess(tuple(matchup.decks[1 - searcher_seat].characters)))
                known = tuple(not blind if seat == searcher_seat else matchup.opponent_deck_known[seat] for seat in (0, 1))
                table.start_game(dataclasses.replace(matchup, opponent_deck_known=known), rng.randrange(2**31),
                                 searcher_seat, {"source": matchup.source, "blind": blind})
            for row in table.step():
                if "failure" not in row:
                    games += 1
                    searcher_wins += row["searcher_won"]
                labels.extend(collector.finish(row["game_id"], row))
            if len(labels) >= settings.shard_labels:
                path = settings.run / LABEL_FOLDER / f"{settings.session}_actor{settings.actor_index:02d}_{shard_number:05d}.pkl.gz"
                write_shard(path, labels)
                progress_queue.put({
                    "actor": settings.actor_index, "shard": path.name, "labels": len(labels),
                    "searched": sum(label.searched for label in labels),
                    "overridden": sum(label.overridden for label in labels),
                    "games": games, "searcher_wins": searcher_wins, "seconds": time.perf_counter() - started,
                    "counts": dict(table.counts),  # since this process started
                })
                labels, games, searcher_wins, shard_number = [], 0, 0, shard_number + 1
                started = time.perf_counter()
