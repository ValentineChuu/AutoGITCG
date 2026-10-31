"""The decks compared: Chudledee's favourite decks from late training, its own card choices for given lineups,
and its stage decks when it owns everything.
"""

import gzip
import json
from collections import Counter
from pathlib import Path

import torch

from autogitcg.data.deck import Deck
from autogitcg.deck_builder.builder_policy import BuilderPolicy
from autogitcg.deck_builder.builder_setup import BuilderSetup, load_builder_checkpoint
from autogitcg.deck_builder.building_task import PVP_MODE, STAGE_MODE, BuildingTask
from autogitcg.deck_building.deck_rules import max_copies
from autogitcg.deck_building.deck_validation import buildable_action_cards, buildable_characters
from autogitcg.progression.stage import Stage
from autogitcg.training.collection_pool import Collection


def favourite_decks(run: Path, first_update: int, last_update: int, minimum_matches: int, count: int) -> list[Deck]:
    """The most common exact deck of each of the `count` most common lineups built from big collections."""
    decks_by_lineup: dict[tuple, Counter] = {}
    for update in range(first_update, last_update + 1):
        with gzip.open(run / "decks" / f"update_{update:05d}.jsonl.gz", "rt") as file:
            for line in file:
                row = json.loads(line)
                if row["matches_played"] < minimum_matches:
                    continue
                lineup = tuple(sorted(row["deck"]["characters"]))
                decks_by_lineup.setdefault(lineup, Counter())[(tuple(row["deck"]["characters"]), tuple(sorted(row["deck"]["cards"])))] += 1
    lineups = sorted(decks_by_lineup, key=lambda lineup: -sum(decks_by_lineup[lineup].values()))[:count]
    favourites = []
    for lineup in lineups:
        (characters, cards), _ = decks_by_lineup[lineup].most_common(1)[0]
        favourites.append(Deck(characters=list(characters), cards=list(cards)))
    return favourites


def _every_card() -> Counter:
    return Counter({card: max_copies(card) for card in buildable_action_cards()})


def builder_cards_for(lineups: list[list[int]], builder_checkpoint: Path) -> list[Deck]:
    """For each lineup: the builder's PvP deck when it owns only those characters but every action card."""
    tasks = [BuildingTask(Collection(tuple(lineup), _every_card(), 0), PVP_MODE, collection_origin=f"lineup:{index}")
             for index, lineup in enumerate(lineups)]
    return _build(tasks, builder_checkpoint)


def builder_stage_decks(stages: list[Stage], builder_checkpoint: Path) -> list[Deck]:
    """For each stage: the builder's deck when it owns every character and every action card."""
    everything = Collection(tuple(sorted(buildable_characters())), _every_card(), 0)
    tasks = [BuildingTask(everything, STAGE_MODE, stage, "everything") for stage in stages]
    return _build(tasks, builder_checkpoint)


def _build(tasks: list[BuildingTask], builder_checkpoint: Path) -> list[Deck]:
    setup = BuilderSetup()
    network = setup.network_from(load_builder_checkpoint(builder_checkpoint))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    built = BuilderPolicy(network.to(device), setup.encoder, device, greedy=True).build(tasks)
    return [deck.deck for deck in built]
