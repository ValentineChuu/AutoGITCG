"""The games that judge one deck: against its stage (stage mode) or against sampled opponents (PvP mode).

The deck plays both seats equally often. In stage mode the deck's side knows the stage deck (as in a real NPC
fight); in PvP neither side knows the other's deck.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass

from autogitcg.data.deck import Deck
from autogitcg.deck_builder.building_task import BuildingTask
from autogitcg.training.matchup_sampler import Matchup, swapped


@dataclass(frozen=True)
class ScheduledGame:
    matchup: Matchup
    seed: int
    deck_seat: int  # where the judged deck sits


def deck_games(task: BuildingTask, deck: Deck, game_count: int, rng: random.Random,
               pvp_opponent: Callable[[], tuple[Deck, str]]) -> list[ScheduledGame]:
    games = []
    for number in range(game_count):
        if task.stage is not None:
            stage = task.stage
            matchup = Matchup("stage", (deck, stage.enemy_deck), stage.enemy_equipment, (True, False),
                              ("built", f"stage:{stage.level_id}"))
        else:
            opponent, origin = pvp_opponent()
            matchup = Matchup("pvp", (deck, opponent), (), (False, False), ("built", origin))
        deck_seat = number % 2
        games.append(ScheduledGame(swapped(matchup) if deck_seat else matchup, rng.randrange(2**31), deck_seat))
    return games
