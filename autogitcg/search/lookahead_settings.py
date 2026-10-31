"""Settings for look-ahead search at action decisions (docs/search.md)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LookaheadSettings:
    candidates: int = 3  # the network's most likely moves that are tried
    redeals: int = 4  # game copies per candidate, each with the hidden cards dealt differently
    horizon: int = 8  # decisions (either player) a copy plays after the candidate before it is scored
    confident_probability: float = 0.9  # no search when the network's top move is at least this likely
    margin: float = 0.05  # a candidate replaces the network's own choice only if it scores this much better
    minimum_probability: float = 0.0  # candidates the network finds less likely than this aren't tried (their
    # copies reach positions it never plays into, where its value can be badly wrong)
    every_move_kind: bool = False  # also try the network's most likely move of each kind it hasn't tried yet (card play,
    # skill, switch, tuning, ending the round), so a plan the network rates low (a setup card) still gets tested
    horizon_rounds: int = 0  # > 0: a copy is scored once it reaches the round (decision's round + this), or after
    # `horizon` decisions, whichever comes first; 2 = to the end of the next round
    score_by: str = "win"  # "win": the win output; "value": the value output. The two players' win outputs disagree
    # badly (at the same position they summed to 1.33 on average, 0.2-2.0; the values summed to about 0)
    own_view_only: bool = False  # once the horizon is reached, play on to the searching player's next decision, so
    # every copy is scored by the same player's view
