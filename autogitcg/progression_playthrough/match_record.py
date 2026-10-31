"""One row per played match: where the account stood, what it played and how it went."""

from autogitcg.data.deck import Deck
from autogitcg.progression.account import Account
from autogitcg.progression.match_result import MatchResult
from autogitcg.progression.simulator import PlannedMatch


def match_record(account_index: int, account: Account, planned: PlannedMatch, result: MatchResult, deck: Deck | None) -> dict:
    """Taken before the result is recorded, so the account state is the one the match was played with."""
    stage = planned.stage
    return {"account": account_index, "match": account.matches_played, "week": planned.week, "kind": stage.kind,
            "level_id": stage.level_id, "stage": stage.name, "won": result.won, "rounds": result.rounds,
            "level": account.level, "characters": len(account.owned_characters()),
            "action_cards": sum(account.owned_action_cards().values()),
            "lineup": list(deck.characters) if deck else None}


def with_state_after(record: dict, account: Account) -> dict:
    """Adds what the account owns once the match's rewards are in (the last match of a playthrough completes it)."""
    return {**record, "characters_after": len(account.owned_characters()), "level_after": account.level}
