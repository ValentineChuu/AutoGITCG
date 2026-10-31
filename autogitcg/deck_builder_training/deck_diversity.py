"""Does the builder keep making the same deck? Measured on near-complete collections, where nothing forces variety.

For each mode: how many decks were built from near-complete collections, how many different lineups they use,
and the share of the most common lineup. A collapse onto one or two decks shows up as few distinct lineups and
a top share near 1. In stage mode a healthy builder adapts to each opponent, so lineups should vary.
"""

from collections import Counter

NEAR_COMPLETE_CHARACTERS = 140  # of 147 buildable characters


def lineup_diversity(decks: list[tuple[str, int, list[int]]]) -> dict:
    """decks: (mode, owned character count, lineup). Returns metrics named near_complete_<mode>_<measure>."""
    metrics = {}
    for mode in sorted({mode for mode, _, _ in decks}):
        lineups = Counter(tuple(sorted(lineup)) for deck_mode, owned, lineup in decks
                          if deck_mode == mode and owned >= NEAR_COMPLETE_CHARACTERS)
        total = sum(lineups.values())
        if not total:
            continue
        metrics[f"near_complete_{mode}_decks"] = total
        metrics[f"near_complete_{mode}_distinct_lineups"] = len(lineups)
        metrics[f"near_complete_{mode}_top_lineup_share"] = lineups.most_common(1)[0][1] / total
    return metrics
