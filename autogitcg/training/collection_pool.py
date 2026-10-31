"""Realistic card collections for training decks: fast-forwarded progression accounts, snapshotted along the way.

Snapshots are spread evenly over collection size: each one targets a character count drawn uniformly between
the starter collection and the full roster, and is taken when the account first owns that many. Spreading
them over match count instead (log-uniform, as at first) put half of them before match 44 (about 6
characters) and almost none in the 100-400 match stretch where collections grow fastest.
Fast-forwarding uses estimated results (no engine), which is good enough for collection shapes; see
autogitcg/progression/fast_forward.py.
"""

import random
from collections import Counter
from dataclasses import dataclass

from autogitcg.deck_building.deck_validation import buildable_characters
from autogitcg.progression.fast_forward import estimated_result
from autogitcg.progression.shop import Shop
from autogitcg.progression.simulator import ProgressionSimulator
from autogitcg.progression.stage import Stage

MAX_MATCHES = 3000  # accounts usually own everything after 900-2000 matches


@dataclass(frozen=True)
class Collection:
    characters: tuple[int, ...]
    action_cards: Counter
    matches_played: int


def simulate_collections(stages: list[Stage], account_count: int, snapshots_per_account: int, seed: int) -> list[Collection]:
    rng = random.Random(seed)
    shop = Shop()
    full_roster = len(buildable_characters())
    collections = []
    for _ in range(account_count):
        simulator = ProgressionSimulator(stages, shop, random.Random(rng.randrange(2**31)))
        starter = len(simulator.account.owned_characters())
        targets = sorted(rng.randint(starter, full_roster) for _ in range(snapshots_per_account))
        for target in targets:
            while (len(simulator.account.owned_characters()) < target and simulator.account.matches_played < MAX_MATCHES
                   and (planned := simulator.next_match()) is not None):
                simulator.record(planned, estimated_result(simulator.account, planned.stage, rng))
            account = simulator.account
            collections.append(Collection(tuple(account.owned_characters()), Counter(account.owned_action_cards()), account.matches_played))
    return collections
