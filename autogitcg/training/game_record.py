"""One finished training game as a JSON-ready dict, for later statistics (one line in games/*.jsonl.gz).

`moves` rows follow `move_columns` (stored in every record so old and new formats both export cleanly).
Network moves fill the last five columns; scripted moves leave them null. For rerolls and mulligans,
definition_id is the number of dice/cards changed and chosen_probability is of the whole keep/change pattern.
"""

import math
from dataclasses import asdict

from autogitcg.bridge.messages import GameFinished
from autogitcg.observation.enums import REACTION_NAMES
from autogitcg.policies.network_policy import NetworkChoice
from autogitcg.training.active_game import ActiveGame
from autogitcg.training.move_summary import MoveSummary

MOVE_COLUMNS = ["seat", "round", "decision_kind", "action_kind", "definition_id", "is_fast", "dice_paid",
                "chosen_probability", "entropy", "value", "win_probability", "option_count"]
NETWORK_COLUMNS = 5


def move_row(who: int, move: MoveSummary, choice: NetworkChoice | None) -> list:
    row = [who, move.round, move.decision_kind, move.action_kind, move.definition_id, int(move.is_fast), move.dice_paid]
    if choice is None:
        return row + [None] * NETWORK_COLUMNS
    return row + [round(math.exp(choice.log_prob), 4), round(choice.entropy, 4), round(choice.value, 4),
                  round(choice.win_probability, 4), len(choice.probabilities)]


def game_record(game: ActiveGame, finished: GameFinished, policy_version: int, actor_index: int) -> dict:
    matchup = game.matchup
    return {
        "policy_version": policy_version,
        "actor_index": actor_index,
        "source": matchup.source,
        "category": matchup.category,
        "focus_seat": matchup.focus_seat,
        "seats": list(game.seats),
        "opponent": game.opponent,
        "snapshot_version": game.snapshot_version,
        "pair_id": game.pair_id,  # reference games: the mirror pair (per actor)
        "deck_origins": list(matchup.deck_origins),
        "decks": [{"characters": list(deck.characters), "cards": list(deck.cards)} for deck in matchup.decks],
        "equipment": [asdict(item) for item in matchup.equipment],
        "opponent_deck_known": list(matchup.opponent_deck_known),
        "seed": game.seed,
        "winner": finished.winner,
        "rounds": finished.rounds,
        "defeated": list(finished.defeated),
        "damage_dealt": list(finished.damage_dealt),
        "reactions": [{REACTION_NAMES.get(code, str(code)): count for code, count in side.items()} for side in finished.reactions],
        "reaction_causes": [{_named_cause(cause): count for cause, count in side.items()} for side in finished.reaction_causes],
        "move_columns": MOVE_COLUMNS,
        "moves": game.moves,
    }


def _named_cause(cause: str) -> str:
    """"102|character|elemental" -> "vaporize|character|elemental"."""
    code, caller_type, skill_type = cause.split("|")
    return f"{REACTION_NAMES.get(int(code), code)}|{caller_type}|{skill_type}"
