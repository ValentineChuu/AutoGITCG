"""Plays matchups between a look-ahead searcher and a network opponent, like the fixed benchmark (training/evaluation.py):
each matchup with the decks in both orders and the searcher in both seats. The search itself: lookahead_table.py.
"""

from collections import Counter

from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.observation.observation_encoder import ObservationEncoder
from autogitcg.policies.network_policy import NetworkPolicy
from autogitcg.search.deck_guess import Lineup
from autogitcg.search.lookahead_settings import LookaheadSettings
from autogitcg.search.lookahead_table import LookaheadTable
from autogitcg.training.matchup_sampler import Matchup, swapped


def play_lookahead_matches(pool: EnginePool, encoder: ObservationEncoder, searcher: NetworkPolicy, rollout: NetworkPolicy,
                           opponent: NetworkPolicy, matchups: list[Matchup], seed: int, settings: LookaheadSettings,
                           on_game_finished=lambda row: None,
                           deck_guesses: dict[Lineup, list[Counter]] | None = None, reply_selection=None) -> tuple[list[dict], Counter]:
    """searcher: samples like normal play; rollout: the same network, greedy. Returns (one row per game, search counts)."""
    table = LookaheadTable(pool, encoder, searcher, rollout, opponent, settings, seed, deck_guesses, reply_selection=reply_selection)
    rows = []
    for number, original in enumerate(matchups):
        for deck_order, matchup in enumerate((original, swapped(original))):
            for searcher_seat in (0, 1):
                row = {"matchup": number, "deck_order": deck_order}
                table.start_game(matchup, seed + number, searcher_seat, row)
                rows.append(row)
    while table.game_count:
        for row in table.step():
            on_game_finished(row)
    return rows, table.counts
