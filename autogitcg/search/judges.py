"""The named judges: look-ahead settings by name (docs/search.md compares them). "none" never searches: the network
plays its own moves."""

from autogitcg.search.lookahead_settings import LookaheadSettings

JUDGES = {
    "none": LookaheadSettings(confident_probability=0.0),
    # the first look-ahead: scored by the (unreliable) win output, either player's view
    "classic": LookaheadSettings(candidates=4, redeals=4, horizon=24),
    "quick": LookaheadSettings(candidates=4, redeals=4, horizon=24, score_by="value", own_view_only=True),
    "cautious": LookaheadSettings(candidates=4, redeals=8, horizon=24, margin=0.0, every_move_kind=True, score_by="value", own_view_only=True),
    "wide": LookaheadSettings(candidates=8, redeals=4, horizon=24, margin=0.0, every_move_kind=True, score_by="value", own_view_only=True),
    "balanced": LookaheadSettings(candidates=8, redeals=8, horizon=24, margin=0.0, every_move_kind=True, score_by="value", own_view_only=True),
    "deep": LookaheadSettings(candidates=8, redeals=16, horizon=24, margin=0.0, every_move_kind=True, score_by="value", own_view_only=True),
}
