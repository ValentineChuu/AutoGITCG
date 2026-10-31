"""The follow-ups input (engine/reference/bridge/follow_ups.mjs): for a checked move, what the rest of my turn after
it can reach, the opponent not acting. Appended to each action token after the replies: per opponent character the
most HP a line takes and whether a line defeats it, the most total damage, how many lines (scaled), the share that
stopped at a draw or random card, whether the search ran out of budget, and "computed". Moves without them (not
checked, or games without replies on demand) get zeros."""

from autogitcg.observation.encode_board import CHARACTERS_PER_SIDE

FOLLOW_UP_WIDTH = 2 * CHARACTERS_PER_SIDE + 5


def follow_up_features(follow_ups: dict | None) -> list[float]:
    if not follow_ups:
        return [0.0] * FOLLOW_UP_WIDTH
    most_lost = (follow_ups["mostLost"] + [0] * CHARACTERS_PER_SIDE)[:CHARACTERS_PER_SIDE]
    defeated = (follow_ups["defeated"] + [False] * CHARACTERS_PER_SIDE)[:CHARACTERS_PER_SIDE]
    lines = follow_ups["lines"]
    return ([lost / 10 for lost in most_lost] + [float(each) for each in defeated]
            + [follow_ups["mostDamage"] / 10, min(lines, 64) / 16, follow_ups["stoppedAtRandom"] / max(lines, 1),
               float(follow_ups["capped"]), 1.0])
