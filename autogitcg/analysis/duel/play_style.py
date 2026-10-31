"""Play-style statistics per network from game logs (autogitcg/analysis/duel/game_log.py), e.g.
a duel's games/*.jsonl.gz: what each network does with its turns (to see why a network's results changed,
judged from how it plays). Counted per game, per network name from the log header.

- card plays per game, split by kind (talent, weapon, artifact, technique, food, resonance, support, other event);
- how often it plays a card when it could: action decisions with a playable card that end in a card play;
- tunings, switches, skills by type, rounds ended with a playable card still in hand;
- bursts: used when ready (decisions where a burst was valid that end in a burst);
- dice and hand size left when it ends the round; wins, and wins per seat; the most played cards.
"""

import collections
import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path

from autogitcg.analysis.definition_kinds import card_facts, skill_facts

CARD_KINDS = ("talent", "weapon", "artifact", "technique", "food", "resonance")  # finer than card_category, by tag


def card_kinds() -> dict[int, str]:
    """Deck card id -> one kind: the first matching tag in CARD_KINDS, else its category (support, event, equipment)."""
    return {card: next((kind for kind in CARD_KINDS if kind in tags), category) for card, (category, _, tags) in card_facts().items()}


@dataclass
class NetworkStyle:
    games: int = 0
    wins: int = 0
    games_by_seat: list[int] = field(default_factory=lambda: [0, 0])
    wins_by_seat: list[int] = field(default_factory=lambda: [0, 0])
    rounds: int = 0
    counts: collections.Counter = field(default_factory=collections.Counter)
    cards: collections.Counter = field(default_factory=collections.Counter)


def _valid(action: dict) -> bool:
    return action.get("validity", 0) == 0


def read_styles(log_files: list[Path]) -> dict[str, NetworkStyle]:
    kinds, skills = card_kinds(), {skill: facts[0] for skill, facts in skill_facts().items()}
    styles: dict[str, NetworkStyle] = collections.defaultdict(NetworkStyle)
    for path in log_files:
        with gzip.open(path, "rt") as file:
            records = [json.loads(line) for line in file]
        header, end = records[0], records[-1]
        if end.get("type") != "end" or end.get("winner") is None:
            continue
        names = header["names"]
        for seat, name in enumerate(names):
            style = styles[name]
            style.games += 1
            style.games_by_seat[seat] += 1
            style.rounds += end.get("rounds") or 0
            if end["winner"] == seat:
                style.wins += 1
                style.wins_by_seat[seat] += 1
        for move in records[1:-1]:
            if move.get("type") != "move" or move["kind"] != "action":
                continue
            style, counts = styles[names[move["who"]]], styles[names[move["who"]]].counts
            options = [entry for entry in move["request"]["action"] if _valid(entry)]
            chosen = move["chosen"] or {}
            case, value = chosen.get("$case"), chosen.get("value", {})
            counts["action decisions"] += 1
            card_playable = any(entry["action"]["$case"] == "playCard" for entry in options)
            burst_ready = any(entry["action"]["$case"] == "useSkill"
                              and skills.get(entry["action"]["value"]["skillDefinitionId"]) == "burst" for entry in options)
            if card_playable:
                counts["decisions with a playable card"] += 1
            if burst_ready:
                counts["decisions with a burst ready"] += 1
            if case == "playCard":
                card = value.get("cardDefinitionId")
                counts["card plays"] += 1
                counts[f"card plays: {kinds.get(card, 'event')}"] += 1
                style.cards[card] += 1
                if card_playable:
                    counts["played a card when it could"] += 1
            elif case == "elementalTuning":
                counts["tunings"] += 1
            elif case == "switchActive":
                counts["switches"] += 1
            elif case == "useSkill":
                skill_type = skills.get(value.get("skillDefinitionId"), "other")
                counts[f"skills: {skill_type}"] += 1
                if skill_type == "burst" and burst_ready:
                    counts["burst used when ready"] += 1
            elif case == "declareEnd":
                player = move["state"]["player"][move["who"]]
                counts["round ends"] += 1
                counts["dice left at round end"] += len(player.get("dice", []))
                counts["hand left at round end"] += len(player.get("handCard", []))
                if card_playable:
                    counts["round ends with a playable card"] += 1
    return dict(styles)
