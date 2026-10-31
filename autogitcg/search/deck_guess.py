"""Guessing a blind opponent's hidden cards for look-ahead copies (docs/search.md).

The opponent's characters are visible from the start, so Chudledee builds a few decks for that lineup (sampled,
owning every action card). A copy deals the opponent's hand and deck from one guess, minus the cards the opponent
has already revealed (played or tuned), so known cards aren't dealt twice.
"""

import random
from collections import Counter
from pathlib import Path

import torch

from autogitcg.deck_builder.builder_policy import BuilderPolicy
from autogitcg.deck_builder.builder_setup import BuilderSetup, load_builder_checkpoint
from autogitcg.deck_builder.building_task import PVP_MODE, BuildingTask
from autogitcg.deck_builder.lineup_feasibility import lineup_is_feasible
from autogitcg.deck_building.deck_rules import max_copies
from autogitcg.deck_building.deck_validation import buildable_action_cards, buildable_characters
from autogitcg.observation.game_tracker import GameTracker
from autogitcg.training.collection_pool import Collection

Lineup = tuple[int, ...]


class DeckGuesser:
    """Chudledee guesses for any lineup, built on first request and kept. Acts like a dict for LookaheadTable."""

    def __init__(self, builder_checkpoint: Path, guesses_per_lineup: int, seed: int, device: torch.device | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup = BuilderSetup()
        network = self.setup.network_from(load_builder_checkpoint(builder_checkpoint)).to(self.device)
        self.policy = BuilderPolicy(network, self.setup.encoder, self.device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.guesses_per_lineup = guesses_per_lineup
        self.every_card = Counter({card: max_copies(card) for card in buildable_action_cards()})
        self.known: dict[Lineup, list[Counter]] = {}

    def can_guess(self, lineup: Lineup) -> bool:
        """Chudledee builds PvP decks: 3 different characters a player can own, that can make a legal deck.
        (Some NPC decks repeat a character; building for them leaves no legal pick and the sampling fails.)"""
        return (len(lineup) == 3 and len(set(lineup)) == 3 and set(lineup) <= buildable_characters()
                and lineup_is_feasible(list(lineup), self.every_card))

    def build(self, lineups: list[Lineup]) -> None:
        """Builds guesses for every new lineup among `lineups` in one batch."""
        new = sorted({tuple(sorted(lineup)) for lineup in lineups} - self.known.keys())
        tasks = [BuildingTask(Collection(lineup, self.every_card, 0), PVP_MODE, collection_origin=f"guess:{index}")
                 for index, lineup in enumerate(new) for _ in range(self.guesses_per_lineup)]
        for task, deck in zip(tasks, self.policy.build(tasks, self.generator)):
            self.known.setdefault(task.collection.characters, []).append(Counter(deck.deck.cards))

    def get(self, lineup: Lineup) -> list[Counter] | None:
        key = tuple(sorted(lineup))
        if key not in self.known and self.can_guess(key):
            self.build([key])
        return self.known.get(key)


def build_guesses(lineups: list[Lineup], builder_checkpoint: Path, guesses_per_lineup: int, seed: int) -> dict[Lineup, list[Counter]]:
    """For each lineup: `guesses_per_lineup` sampled Chudledee decks (card counts), keyed by the sorted lineup."""
    guesser = DeckGuesser(builder_checkpoint, guesses_per_lineup, seed)
    guesser.build(lineups)
    return guesser.known


def hidden_cards(guess: Counter, tracker: GameTracker, hidden_count: int, rng: random.Random) -> list[int] | None:
    """`hidden_count` card ids for the opponent's hand and deck: the guess minus the revealed cards, trimmed or padded
    (with more cards from the guess) to the right count. None if the guess has no cards but some are hidden (e.g. a
    PvE deck without action cards holding cards created by effects)."""
    if hidden_count and not guess:
        return None
    remaining = guess - tracker.opponent_played_cards
    cards = list(remaining.elements()) or list(guess.elements())
    rng.shuffle(cards)
    while len(cards) < hidden_count:
        cards.append(rng.choice(list(guess.elements())))
    return cards[:hidden_count]
