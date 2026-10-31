"""Which stages an account can play right now, following the game's gating rules."""

from autogitcg.progression.account import Account
from autogitcg.progression.objectives import objective_measurable
from autogitcg.progression.stage import (
    ASCENSION, INVITATION_FRIENDLY, INVITATION_SERIOUS, TAVERN_FRIENDLY, TAVERN_SERIOUS, WEEKLY_GUEST, Stage,
)
from autogitcg.progression.weekly import guest_stage


def has_open_objectives(account: Account, stage: Stage) -> bool:
    """Unclaimed objectives that a match can actually achieve (unmeasurable ones would keep a stage open forever)."""
    return any((stage.level_id, index) not in account.completed_objectives and objective_measurable(objective)
               for index, objective in enumerate(stage.objectives))


class StageAvailability:
    def __init__(self, stages: list[Stage]):
        self.stages = [stage for stage in stages if stage.kind != WEEKLY_GUEST]
        self.guest_variants: dict[int, list[Stage]] = {}
        for stage in stages:
            if stage.kind == WEEKLY_GUEST:
                self.guest_variants.setdefault(stage.group, []).append(stage)
        self.friendly_level_ids: dict[tuple[str, int], set[int]] = {}
        for stage in stages:
            if stage.kind in (INVITATION_FRIENDLY, TAVERN_FRIENDLY):
                self.friendly_level_ids.setdefault((stage.kind, stage.group), set()).add(stage.level_id)

    def _cleared_friendly(self, account: Account, kind: str, group: int) -> bool:
        """Serious Showdowns open once the matching Friendly Fracas has been won."""
        return not self.friendly_level_ids.get((kind, group), set()).isdisjoint(account.won_levels)

    def available(self, account: Account) -> list[Stage]:
        level = account.level
        result = []
        for stage in self.stages:
            if stage.required_player_level > level:
                continue
            if stage.kind == ASCENSION:
                if stage.raises_level_cap_to and stage.raises_level_cap_to > account.level_cap:
                    result.append(stage)
                continue
            if not has_open_objectives(account, stage):
                continue
            if stage.kind == INVITATION_FRIENDLY and stage.letter_cost and stage.group not in account.invited and account.letters < stage.letter_cost:
                continue
            if stage.kind == INVITATION_SERIOUS and not self._cleared_friendly(account, INVITATION_FRIENDLY, stage.group):
                continue
            if stage.kind == TAVERN_SERIOUS and not self._cleared_friendly(account, TAVERN_FRIENDLY, stage.group):
                continue
            result.append(stage)
        for group in account.weekly_guests:
            if group not in account.weekly_guests_beaten and group in self.guest_variants:
                result.append(guest_stage(self.guest_variants[group], level))
        return result
