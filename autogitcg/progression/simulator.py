"""Simulates one player account from the post-tutorial starter deck onward.

The simulator decides what the account does next; the caller plays the match (with the bot, or with
`fast_forward.estimated_result` when only realistic collections are needed) and reports the result:

    simulator = ProgressionSimulator(stages, shop, rng)
    while (planned := simulator.next_match()) is not None:
        result = play(simulator.account, planned.stage)
        simulator.record(planned, result)
"""

import random
from dataclasses import dataclass

from autogitcg.progression.account import Account
from autogitcg.progression.account_policy import AccountPolicy
from autogitcg.progression.personality import Personality, random_personality
from autogitcg.progression.match_result import MatchResult
from autogitcg.progression.player_manual import claim_manual_rewards
from autogitcg.progression.rewards import claim_objectives, grant_level_rewards
from autogitcg.progression.shop import Shop
from autogitcg.progression.stage import ASCENSION, INVITATION_FRIENDLY, WEEKLY_GUEST, Stage
from autogitcg.progression.stage_availability import StageAvailability
from autogitcg.progression.starter import starter_account
from autogitcg.progression.weekly import record_guest_win, start_week

MAX_EMPTY_WEEKS = 3  # stop once nothing is left to do for this many weeks in a row


@dataclass(frozen=True)
class PlannedMatch:
    stage: Stage
    week: int


class ProgressionSimulator:
    def __init__(self, stages: list[Stage], shop: Shop, rng: random.Random, personality: Personality | None = None):
        self.rng = rng
        self.availability = StageAvailability(stages)
        invitation_characters = sorted({stage.group for stage in stages if stage.kind == INVITATION_FRIENDLY})
        self.personality = personality or random_personality(rng, invitation_characters)
        self.policy = AccountPolicy(self.personality, shop, rng)
        self.letter_invitations = [stage for stage in stages if stage.kind == INVITATION_FRIENDLY and stage.letter_cost]
        self.account: Account = starter_account()
        claim_manual_rewards(self.account)
        start_week(self.account, sorted(self.availability.guest_variants), rng)

    def next_match(self) -> PlannedMatch | None:
        """The next stage to play, or None when the account has nothing left to do."""
        empty_weeks = 0
        while True:
            account = self.account
            self.policy.go_shopping(account, self.policy.wants_letter(account, self.letter_invitations))
            available = self.availability.available(account)
            if available and account.matches_this_week < self.personality.matches_per_week:
                stage = self.policy.choose_stage(account, available)
                if stage.kind == INVITATION_FRIENDLY and stage.letter_cost and stage.group not in account.invited:
                    account.letters -= stage.letter_cost
                    account.invited.add(stage.group)
                return PlannedMatch(stage, account.week)
            empty_weeks = 0 if available else empty_weeks + 1
            if empty_weeks >= MAX_EMPTY_WEEKS:
                return None
            start_week(account, sorted(self.availability.guest_variants), self.rng)

    def record(self, planned: PlannedMatch, result: MatchResult) -> None:
        account, stage = self.account, planned.stage
        account.matches_played += 1
        account.matches_this_week += 1
        if result.won:
            if stage.level_id not in account.won_levels:
                account.won_levels.add(stage.level_id)
                account.cleared_stages[stage.kind] += 1
            if stage.kind == WEEKLY_GUEST and planned.week == account.week:
                record_guest_win(account, stage.group)
            if stage.kind == ASCENSION and stage.raises_level_cap_to and stage.raises_level_cap_to > account.level_cap:
                old_level = account.level
                account.level_cap = stage.raises_level_cap_to
                grant_level_rewards(account, old_level, account.level)
        claim_objectives(account, stage, result)
        claim_manual_rewards(account)
