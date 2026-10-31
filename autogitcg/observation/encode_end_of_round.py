"""The end-of-round input (engine/reference/bridge/end_of_round.mjs): if both players ended the round now, what the
end phase would do. Appended to the global token: per character, mine first then the opponent's, the HP change and
whether it would be defeated; then "computed", and whether the game would end with my win or my loss. Decisions
without it (games without replies, seats without a reply view, single-move decisions) get zeros."""

from autogitcg.observation.encode_board import CHARACTERS_PER_SIDE

END_OF_ROUND_WIDTH = 2 * CHARACTERS_PER_SIDE * 2 + 3


def end_of_round_features(end_of_round: dict | None, who: int) -> list[float]:
    if not end_of_round:
        return [0.0] * END_OF_ROUND_WIDTH
    features = []
    for seat in (who, 1 - who):
        characters = end_of_round["change"][seat][:CHARACTERS_PER_SIDE]
        for position in range(CHARACTERS_PER_SIDE):
            character = characters[position] if position < len(characters) else None
            features += [character["health"] / 10, float(character["defeated"])] if character else [0.0, 0.0]
    ended, winner = end_of_round.get("ended", False), end_of_round.get("winner")
    return features + [1.0, float(ended and winner == who), float(ended and winner == 1 - who)]
