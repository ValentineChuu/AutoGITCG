"""Progression simulator rules: stage catalog, starter account, objectives, rewards, deck legality, full runs."""

import random
from collections import Counter

import pytest

from autogitcg.deck_building.deck_rules import card_allowed, max_copies
from autogitcg.deck_building.random_deck_builder import random_deck
from autogitcg.progression.economy import LUCKY_COIN_CAP, MAX_COPIES_PER_CARD
from autogitcg.progression.fast_forward import estimated_result
from autogitcg.progression.items import LUCKY_COIN
from autogitcg.progression.match_result import MatchResult
from autogitcg.progression.objectives import objective_achieved
from autogitcg.progression.rewards import claim_objectives
from autogitcg.progression.shop import Shop
from autogitcg.progression.simulator import ProgressionSimulator
from autogitcg.progression.stage import (
    ASCENSION, INVITATION_FRIENDLY, INVITATION_SERIOUS, TAVERN_FRIENDLY, TAVERN_SERIOUS, WEEKLY_GUEST, Objective, Stage,
)
from autogitcg.progression.stage_catalog import build_stage_catalog
from autogitcg.progression.starter import starter_account, starter_deck
from autogitcg.data.deck import Deck


@pytest.fixture(scope="module")
def stages():
    supported, unsupported = build_stage_catalog()
    assert not unsupported, [stage.name for stage in unsupported]
    return supported


def test_catalog_has_every_stage_kind(stages):
    kinds = Counter(stage.kind for stage in stages)
    assert kinds[INVITATION_FRIENDLY] == 324 and kinds[INVITATION_SERIOUS] == 108
    assert kinds[TAVERN_FRIENDLY] == 39 and kinds[TAVERN_SERIOUS] == 39
    assert kinds[WEEKLY_GUEST] == 535 and kinds[ASCENSION] == 4


def test_starter_account_matches_the_wiki():
    deck = starter_deck()
    assert len(deck.characters) == 3 and len(deck.cards) == 30 and len(set(deck.cards)) == 30
    account = starter_account()
    assert len(account.owned_characters()) == 4  # the starter lineup plus Fischl
    assert account.letters == 1 and account.lucky_coins == 500 and account.level == 1


def _result(**overrides) -> MatchResult:
    values = dict(won=True, rounds=6, my_defeated=1, bursts_used=1, cards_played=())
    return MatchResult(**(values | overrides))


def test_objectives_follow_challenge_semantics():
    win = Objective("GCG_CHALLENGE_WIN", 1, (), ())
    within_six = Objective("GCG_CHALLENGE_ROUND_NUM", 6, (), ())
    fewer_than_two_defeated = Objective("GCG_CHALLENGE_DEAD_CHARACTER_NUM", 2, (), ())
    assert objective_achieved(win, _result()) and not objective_achieved(win, _result(won=False))
    assert objective_achieved(within_six, _result(rounds=6)) and not objective_achieved(within_six, _result(rounds=7))
    assert objective_achieved(fewer_than_two_defeated, _result(my_defeated=1))
    assert not objective_achieved(fewer_than_two_defeated, _result(my_defeated=2))
    assert objective_achieved(Objective("GCG_CHALLENGE_ELEMENT_REACTION_TIMES", 1, (), ()), _result()) is None


def test_coin_objectives_wait_while_at_the_cap():
    stage = Stage(1, "test", INVITATION_SERIOUS, 0, Deck([], []), (),
                  (Objective("GCG_CHALLENGE_WIN", 1, (), ((LUCKY_COIN, 1000),)),))
    account = starter_account()
    account.lucky_coins = LUCKY_COIN_CAP
    assert claim_objectives(account, stage, _result()) == []
    account.lucky_coins = LUCKY_COIN_CAP - 1
    assert claim_objectives(account, stage, _result()) == [0]
    assert account.lucky_coins == LUCKY_COIN_CAP + 999  # a reward that crosses the cap is kept in full


def test_talents_and_resonances_need_matching_lineups():
    ganyu, diona, keqing, kaeya = 1101, 1102, 1403, 1103
    ganyu_talent = 211011  # Undivided Heart
    cryo_resonance = 331101
    assert card_allowed(ganyu_talent, [ganyu, diona, keqing])
    assert not card_allowed(ganyu_talent, [kaeya, diona, keqing])
    assert card_allowed(cryo_resonance, [ganyu, diona, keqing])
    assert not card_allowed(cryo_resonance, [ganyu, keqing, 1301])


def test_random_decks_are_legal():
    rng = random.Random(0)
    account = starter_account()
    for _ in range(20):
        deck = random_deck(account.owned_characters(), account.owned_action_cards(), rng)
        assert len(deck.characters) == 3 and len(deck.cards) == 30
        assert all(card_allowed(card, deck.characters) for card in deck.cards)
        assert all(count <= min(max_copies(card), account.cards[card]) for card, count in Counter(deck.cards).items())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fast_forwarded_accounts_progress_to_level_ten(stages, seed):
    rng = random.Random(seed)
    simulator = ProgressionSimulator(stages, Shop(), rng)
    caps_seen = set()
    while simulator.account.matches_played < 3000 and (planned := simulator.next_match()) is not None:
        simulator.record(planned, estimated_result(simulator.account, planned.stage, rng))
        account = simulator.account
        caps_seen.add(account.level_cap)
        assert account.lucky_coins >= 0 and account.letters >= 0
        assert all(count <= MAX_COPIES_PER_CARD for count in account.shop_bought.values())
    assert simulator.account.level == 10
    assert caps_seen == {2, 4, 6, 9, 10}
    assert len(simulator.account.owned_characters()) > 40
