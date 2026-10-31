"""Picks what each deck is built for: a simulated collection, and either a PvE stage (known opponent) or PvP.

Stages: invitations (the hardest real use case) get half of the stage tasks; the other kinds split the rest
by how many stages they have, as in combat training (training/matchup_categories.py).
"""

import random
from collections import Counter

from autogitcg.deck_builder.building_task import PVP_MODE, STAGE_MODE, BuildingTask
from autogitcg.progression.stage import Stage
from autogitcg.training.collection_pool import Collection
from autogitcg.training.matchup_categories import INVITATION_KINDS, INVITATION_STAGE_SHARE


class TaskSampler:
    def __init__(self, collections: list[Collection], stages: list[Stage], stage_share: float, rng: random.Random):
        self.collections = collections
        self.stages = stages
        self.stage_share = stage_share
        self.rng = rng
        kind_counts = Counter(stage.kind for stage in stages)
        invitation_total = sum(kind_counts[kind] for kind in INVITATION_KINDS)
        other_total = len(stages) - invitation_total
        self.stage_weights = [
            (INVITATION_STAGE_SHARE / invitation_total if stage.kind in INVITATION_KINDS else (1 - INVITATION_STAGE_SHARE) / other_total)
            for stage in stages
        ]

    def sample(self) -> BuildingTask:
        index = self.rng.randrange(len(self.collections))
        origin = f"collection:{index}"
        if self.rng.random() < self.stage_share:
            stage = self.rng.choices(self.stages, weights=self.stage_weights)[0]
            return BuildingTask(self.collections[index], STAGE_MODE, stage, origin)
        return BuildingTask(self.collections[index], PVP_MODE, None, origin)
