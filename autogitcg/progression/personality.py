"""Per-account random tendencies, so simulated accounts take different paths through the game."""

import random
from dataclasses import dataclass

from autogitcg.progression.stage import (
    ASCENSION, INVITATION_FRIENDLY, INVITATION_SERIOUS, OPEN_WORLD, TAVERN_FRIENDLY, TAVERN_SERIOUS, WEEKLY_GUEST,
)

CARD_TYPES = ["GCG_CARD_MODIFY", "GCG_CARD_ASSIST", "GCG_CARD_EVENT"]  # equipment, support, event


@dataclass(frozen=True)
class Personality:
    kind_weights: dict[str, float]  # how much the account likes each kind of stage
    target_characters: frozenset[int]  # invitations chased first ("forced" goals)
    card_type_weights: dict[str, float]  # shopping taste
    matches_per_week: int
    coins_kept: int  # the account shops only with coins above this reserve


def random_personality(rng: random.Random, invitation_characters: list[int]) -> Personality:
    kinds = [OPEN_WORLD, INVITATION_FRIENDLY, INVITATION_SERIOUS, TAVERN_FRIENDLY, TAVERN_SERIOUS, WEEKLY_GUEST, ASCENSION]
    return Personality(
        kind_weights={kind: rng.uniform(0.3, 2.0) for kind in kinds},
        target_characters=frozenset(rng.sample(invitation_characters, k=min(len(invitation_characters), rng.randint(3, 12)))),
        card_type_weights={card_type: rng.uniform(0.3, 2.0) for card_type in CARD_TYPES},
        matches_per_week=rng.randint(8, 40),
        coins_kept=rng.choice([0, 500, 1000, 2000, 4000]),
    )
