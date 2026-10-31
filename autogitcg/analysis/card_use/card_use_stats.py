"""Counts of how a side uses its action cards: plays, tunes, skills, and round ends with a useful card still playable."""

from collections import Counter

from autogitcg.analysis.card_use.play_quality import record_play_quality

PLAYABLE = 0  # ActionValidity.VALID


def useful_card_playable(options: list[dict]) -> bool:
    """A card play is legal now and the engine doesn't flag it as having no effect."""
    return any(option["action"]["$case"] == "playCard" and option.get("validity", PLAYABLE) == PLAYABLE
               and not option["action"]["value"].get("willBeEffectless") for option in options)


def record_action(stats: Counter, options: list[dict], chosen_index: int, me: dict | None = None) -> None:
    """me: the acting player's state, for the play-quality counts (play_quality.py)."""
    if me is not None:
        record_play_quality(stats, options[chosen_index], me)
    chosen = options[chosen_index]["action"]["$case"]
    stats[chosen] += 1
    if chosen == "declareEnd":
        stats["round ends"] += 1
        stats["round ends with a useful playable card"] += useful_card_playable(options)


def record_game(stats: Counter, won: bool, rounds: int) -> None:
    stats["games"] += 1
    stats["wins"] += won
    stats["rounds"] += rounds


def summary(stats: Counter) -> dict:
    games, rounds = max(stats["games"], 1), max(stats["rounds"], 1)
    return {"games": stats["games"], "win_rate": stats["wins"] / games, "rounds_per_game": stats["rounds"] / games,
            "cards_played_per_game": stats["playCard"] / games, "cards_played_per_round": stats["playCard"] / rounds,
            "cards_tuned_per_game": stats["elementalTuning"] / games, "skills_per_game": stats["useSkill"] / games,
            "round_ends_with_useful_playable_card": stats["round ends with a useful playable card"] / max(stats["round ends"], 1),
            "healing_food_per_game": stats["healing food plays"] / games,
            "healing_food_at_full_health": stats["healing food at full health"] / max(stats["healing food plays"], 1)}
