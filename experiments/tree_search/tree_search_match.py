"""Plays matchups between a tree searcher and a network opponent like lookahead_match.py: each matchup with the decks
in both orders and the searcher in both seats. The search itself: tree_search_table.py."""

from collections import Counter

from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.observation.observation_encoder import ObservationEncoder
from autogitcg.policies.network_policy import NetworkPolicy
from autogitcg.search.deck_guess import Lineup
from experiments.tree_search.tree_search_settings import TreeSearchSettings
from experiments.tree_search.tree_search_table import TreeSearchTable
from autogitcg.training.matchup_sampler import Matchup, swapped


def play_tree_search_matches(pool: EnginePool, encoder: ObservationEncoder, searcher: NetworkPolicy, rollout: NetworkPolicy,
                             opponent: NetworkPolicy, matchups: list[Matchup], seed: int, settings: TreeSearchSettings,
                             on_game_finished=lambda row: None, deck_guesses: dict[Lineup, list[Counter]] | None = None,
                             on_searcher_move=None) -> tuple[list[dict], TreeSearchTable]:
    """Returns (one row per game, the table: its counts and search times)."""
    extra = {"on_searcher_move": on_searcher_move} if on_searcher_move else {}
    table = TreeSearchTable(pool, encoder, searcher, rollout, opponent, settings, seed, deck_guesses, **extra)
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
    return rows, table
