"""Choose a reproducible, dice-stratified set of situations for counterfactual rollout."""

import random
from collections import defaultdict

from experiments.card_play_diagnostic.card_types import CARD_TYPES


def select_situations(situations: list[dict], per_type: int, seed: int) -> list[dict]:
    random_generator = random.Random(seed)
    grouped = defaultdict(list)
    for situation in situations:
        grouped[(situation["card_type"], situation["dice_bucket"])].append(situation)
    selected = []
    for card_type in CARD_TYPES:
        buckets = {bucket: list(grouped[(card_type, bucket)]) for bucket in ("0-2", "3-5", "6+")}
        for values in buckets.values():
            random_generator.shuffle(values)
        target_by_bucket = {bucket: per_type // 3 for bucket in buckets}
        for bucket in list(target_by_bucket)[:per_type % 3]:
            target_by_bucket[bucket] += 1
        chosen = []
        for bucket, target in target_by_bucket.items():
            chosen.extend(buckets[bucket][:target])
        remaining = [item for bucket, values in buckets.items() for item in values[target_by_bucket[bucket]:]]
        random_generator.shuffle(remaining)
        chosen.extend(remaining[:max(0, per_type - len(chosen))])
        selected.extend(chosen)
    return selected
