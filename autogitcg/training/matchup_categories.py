"""The kinds of training matchups, each with a base sampling weight.

A category names what the "focus" side plays and against what:
- "stage/<stage kind>/<synergy|collection|chudledee>": a player deck vs a PvE stage of that kind (invitations
  get half of the stage share, the other kinds split the rest by how many stages they have; "chudledee" is a
  deck Chudledee built for that stage);
- "tournament": a real tournament deck vs a tournament, Chudledee, synergy or collection deck;
- "chudledee": a Chudledee PvP deck vs the same mix;
- "synergy/<archetype>": a hand-built synergy deck vs a synergy, NPC or collection deck;
- "npc_decks", "collections".
Sources whose pool is empty are left out. Base weights sum to 1; AdaptiveDifficulty then shifts weight toward
the categories where the learner plays worst.
"""

from collections import Counter

from autogitcg.progression.stage import INVITATION_FRIENDLY, INVITATION_SERIOUS
from autogitcg.training.deck_pools import DeckPools

# alpha 2 and 3: {stage 0.4, synergy 0.2, npc_decks 0.15, collections 0.25}. beta 10.8 adds human decks.
SOURCE_WEIGHTS = {"stage": 0.35, "tournament": 0.2, "chudledee": 0.1, "synergy": 0.1, "npc_decks": 0.1, "collections": 0.15}
INVITATION_STAGE_SHARE = 0.5  # of stage games
INVITATION_KINDS = (INVITATION_FRIENDLY, INVITATION_SERIOUS)


def base_category_weights(pools: DeckPools, source_weights: dict[str, float] | None = None) -> dict[str, float]:
    """source_weights overrides SOURCE_WEIGHTS (missing sources count as 0)."""
    source_weights = SOURCE_WEIGHTS if source_weights is None else {source: source_weights.get(source, 0.0) for source in SOURCE_WEIGHTS}
    weights: dict[str, float] = {}
    stage_counts = Counter(stage.kind for stage in pools.stages)
    invitation_total = sum(stage_counts[kind] for kind in INVITATION_KINDS)
    other_total = sum(count for kind, count in stage_counts.items() if kind not in INVITATION_KINDS)
    for kind, count in stage_counts.items():
        if kind in INVITATION_KINDS:
            kind_share = INVITATION_STAGE_SHARE * count / invitation_total
        else:
            kind_share = (1 - INVITATION_STAGE_SHARE) * count / other_total
        player_types = ["collection"] + ["synergy"] * bool(pools.synergy_decks) + ["chudledee"] * bool(pools.chudledee.stage)
        for player_type in player_types:
            weights[f"stage/{kind}/{player_type}"] = source_weights["stage"] * kind_share / len(player_types)
    archetype_counts = Counter(pools.synergy_archetypes)
    for archetype, count in archetype_counts.items():
        weights[f"synergy/{archetype}"] = source_weights["synergy"] * count / len(pools.synergy_archetypes)
    if pools.tournament_decks:
        weights["tournament"] = source_weights["tournament"]
    if pools.chudledee.pvp:
        weights["chudledee"] = source_weights["chudledee"]
    weights["npc_decks"] = source_weights["npc_decks"]
    weights["collections"] = source_weights["collections"]
    total = sum(weights.values())
    return {category: weight / total for category, weight in weights.items() if weight > 0}
