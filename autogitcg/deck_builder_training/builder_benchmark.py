"""Fixed deck builder benchmark: the same tasks, opponents and seeds every time, on held-out collections.

Collections come from accounts simulated with their own seed, so they never appear in training. Each task's
games (opponents, seats, seeds) depend only on the task, never on the deck, so the builder's decks and the
random-deck baseline (today's training decks) play exactly the same games.
"""

import random
from collections import defaultdict

from autogitcg.data.deck import Deck
from autogitcg.deck_builder.building_task import BuildingTask
from autogitcg.deck_building.random_deck_builder import random_deck
from autogitcg.deck_builder_training.deck_diversity import lineup_diversity
from autogitcg.deck_builder_training.deck_games import ScheduledGame, deck_games
from autogitcg.deck_builder_training.deck_rewards import deck_result
from autogitcg.deck_builder_training.game_player_group import GamePlayerGroup
from autogitcg.deck_builder_training.task_sampler import TaskSampler
from autogitcg.progression.stage import Stage
from autogitcg.training.collection_pool import simulate_collections
from autogitcg.training.deck_pools import DeckPools

BENCHMARK_SEED = 424242
BENCHMARK_ACCOUNTS = 24
BENCHMARK_SNAPSHOTS_PER_ACCOUNT = 4
SMALL_COLLECTION_CHARACTERS = 20
MEDIUM_COLLECTION_CHARACTERS = 100


class BuilderBenchmark:
    def __init__(self, stages: list[Stage], pools: DeckPools, legal_npc_deck_ids: list[int], tasks_per_mode: int, games_per_deck: int):
        collections = simulate_collections(stages, BENCHMARK_ACCOUNTS, BENCHMARK_SNAPSHOTS_PER_ACCOUNT, BENCHMARK_SEED)
        rng = random.Random(BENCHMARK_SEED)
        stage_tasks = TaskSampler(collections, stages, 1.0, rng)
        pvp_tasks = TaskSampler(collections, stages, 0.0, rng)
        self.tasks = ([stage_tasks.sample() for _ in range(tasks_per_mode)] + [pvp_tasks.sample() for _ in range(tasks_per_mode)])
        self.tasks = [BuildingTask(task.collection, task.mode, task.stage, f"benchmark_{task.collection_origin}") for task in self.tasks]
        self.opponents = [(pools.synergy_decks[index], f"synergy:{index}") for index in range(len(pools.synergy_decks))]
        self.opponents += [(pools.npc_decks[deck_id], f"npc_deck:{deck_id}") for deck_id in legal_npc_deck_ids]
        self.games_per_deck = games_per_deck

    def random_decks(self) -> list[Deck]:
        """The baseline: a random legal deck per task, the same every time."""
        return [random_deck(list(task.collection.characters), task.collection.action_cards, random.Random(BENCHMARK_SEED + number))
                for number, task in enumerate(self.tasks)]

    def games(self, decks: list[Deck]) -> list[list[ScheduledGame]]:
        schedules = []
        for number, (task, deck) in enumerate(zip(self.tasks, decks)):
            rng = random.Random(BENCHMARK_SEED * 7 + number)
            schedules.append(deck_games(task, deck, self.games_per_deck, rng, lambda rng=rng: rng.choice(self.opponents)))
        return schedules

    def evaluate(self, players: GamePlayerGroup, decks: list[Deck]) -> tuple[dict, list[dict]]:
        """(summary per mode and overall, one row per task with its deck and result)."""
        schedules = self.games(decks)
        outcomes = players.play([game for schedule in schedules for game in schedule])
        rows, position = [], 0
        for task, deck, schedule in zip(self.tasks, decks, schedules):
            result = deck_result(schedule, outcomes[position:position + len(schedule)])
            position += len(schedule)
            rows.append({"task": task.origin, "mode": task.mode, "owned_characters": len(task.collection.characters), "deck": {"characters": deck.characters, "cards": deck.cards},
                         "reward": result.reward, "wins": result.wins, "games": result.games, "failures": result.failures})
        return summarize(rows), rows


def collection_size(owned_characters: int) -> str:
    """Small: early game, few real choices. Medium: real but constrained choices. Large: nearly everything owned."""
    return "small" if owned_characters <= SMALL_COLLECTION_CHARACTERS else "medium" if owned_characters <= MEDIUM_COLLECTION_CHARACTERS else "large"


def summarize(rows: list[dict]) -> dict:
    """Win rate and mean reward per mode, overall, and per collection size (all modes together)."""
    by_mode = defaultdict(list)
    for row in rows:
        by_mode[row["mode"]].append(row)
        by_mode["all"].append(row)
        by_mode[f"{collection_size(row['owned_characters'])}_collections"].append(row)
    summary = {}
    for mode, mode_rows in by_mode.items():
        games = sum(row["games"] for row in mode_rows)
        rewarded = [row["reward"] for row in mode_rows if row["reward"] is not None]
        summary[mode] = {"win_rate": sum(row["wins"] for row in mode_rows) / max(games, 1),
                         "mean_reward": sum(rewarded) / max(len(rewarded), 1), "games": games,
                         "failures": sum(row["failures"] for row in mode_rows)}
    summary["lineup_diversity"] = lineup_diversity([(row["mode"], row["owned_characters"], row["deck"]["characters"]) for row in rows])
    return summary

