"""Confidence intervals and grouped summaries for the card-play diagnostic."""

import math
from collections import defaultdict

import numpy as np

from experiments.card_play_diagnostic.card_types import CARD_TYPES
from autogitcg.analysis.game_text.english_names import card_name


def wilson_interval(successes: int, count: int) -> tuple[float, float]:
    if count == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    proportion = successes / count
    denominator = 1 + z * z / count
    centre = (proportion + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count * count)) / denominator
    return centre - radius, centre + radius


def situation_summaries(situations: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for situation in situations:
        grouped[(situation["card_type"], situation["dice_bucket"])].append(situation)
    rows = []
    for card_type in CARD_TYPES:
        for dice in ("all", "0-2", "3-5", "6+"):
            values = ([situation for situation in situations if situation["card_type"] == card_type]
                      if dice == "all" else grouped[(card_type, dice)])
            if not values:
                continue
            rows.append({
                "card_type": card_type, "dice_bucket": dice, "situations": len(values),
                "play_probability": _mean(values, "play_probability"),
                "tune_probability": _mean(values, "tune_probability"),
                "other_probability": 1 - _mean(values, "play_probability") - _mean(values, "tune_probability"),
                "value": _mean(values, "value"), "win_probability": _mean(values, "win_probability"),
            })
            for field in ("play_probability", "tune_probability", "value", "win_probability"):
                rows[-1][f"{field}_low"], rows[-1][f"{field}_high"] = _cluster_mean_interval(
                    values, field, "game_number", 1_000_000 + len(rows) * 10 + len(field))
    return rows


def branch_summaries(results: list[dict], bootstrap_seed: int = 20261015) -> list[dict]:
    grouped = defaultdict(list)
    for result in results:
        grouped[result["card_type"]].append(result)
    return [_branch_summary(card_type, grouped[card_type], bootstrap_seed + index)
            for index, card_type in enumerate(CARD_TYPES) if grouped[card_type]]


def branch_summaries_by_dice(results: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for result in results:
        grouped[(result["card_type"], result["dice_bucket"])].append(result)
    rows = []
    for type_index, card_type in enumerate(CARD_TYPES):
        for dice_index, dice in enumerate(("0-2", "3-5", "6+")):
            values = grouped[(card_type, dice)]
            if values:
                row = _branch_summary(card_type, values, 4_000_000 + type_index * 10 + dice_index)
                row["dice_bucket"] = dice
                rows.append(row)
    return rows


def card_summaries(results: list[dict], minimum_situations: int = 3) -> list[dict]:
    grouped = defaultdict(list)
    for result in results:
        grouped[(result["card_type"], result["card_id"])].append(result)
    rows = []
    for index, ((card_type, card_id), values) in enumerate(sorted(grouped.items())):
        situations = len({value["situation_id"] for value in values})
        if situations < minimum_situations:
            continue
        row = _branch_summary(card_type, values, 3_000_000 + index)
        row.update(card_id=card_id, card_name=card_name(card_id))
        rows.append(row)
    return sorted(rows, key=lambda row: row["win_rate_difference"], reverse=True)


def _branch_summary(card_type: str, values: list[dict], bootstrap_seed: int) -> dict:
    by_branch = {branch: [value for value in values if value["branch"] == branch] for branch in ("play", "tune")}
    row = {"card_type": card_type, "situations": len({value["situation_id"] for value in values})}
    for branch, branch_values in by_branch.items():
        wins = sum(value["won"] for value in branch_values)
        rates_by_situation = defaultdict(list)
        for value in branch_values:
            rates_by_situation[value["situation_id"]].append(float(value["won"]))
        low, high = _bootstrap_mean_interval(
            [float(np.mean(outcomes)) for outcomes in rates_by_situation.values()],
            bootstrap_seed + (0 if branch == "play" else 10_000))
        row.update({f"{branch}_wins": wins, f"{branch}_rollouts": len(branch_values),
                    f"{branch}_win_rate": wins / len(branch_values),
                    f"{branch}_win_rate_low": low, f"{branch}_win_rate_high": high,
                    f"{branch}_remaining_rounds": _mean(branch_values, "remaining_rounds"),
                    f"{branch}_draw_rate": sum(value["draw"] for value in branch_values) / len(branch_values)})
    win_differences = _situation_differences(values, "won")
    round_differences = _situation_differences(values, "remaining_rounds")
    row["win_rate_difference"] = float(np.mean(win_differences))
    row["win_rate_difference_low"], row["win_rate_difference_high"] = _bootstrap_mean_interval(
        win_differences, bootstrap_seed)
    row["remaining_round_difference"] = float(np.mean(round_differences))
    row["remaining_round_difference_low"], row["remaining_round_difference_high"] = _bootstrap_mean_interval(
        round_differences, bootstrap_seed + 1)
    return row


def _situation_differences(values: list[dict], field: str) -> list[float]:
    grouped = defaultdict(lambda: {"play": [], "tune": []})
    for value in values:
        grouped[value["situation_id"]][value["branch"]].append(float(value[field]))
    return [float(np.mean(branches["play"]) - np.mean(branches["tune"]))
            for branches in grouped.values() if branches["play"] and branches["tune"]]


def _bootstrap_mean_interval(values: list[float], seed: int, samples: int = 10_000) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 1:
        return float(array[0]), float(array[0])
    random_generator = np.random.default_rng(seed)
    means = np.empty(samples)
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        means[start:start + count] = array[random_generator.integers(0, len(array), size=(count, len(array)))].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, (0.025, 0.975)))


def _cluster_mean_interval(values: list[dict], field: str, cluster_field: str, seed: int,
                           samples: int = 10_000) -> tuple[float, float]:
    grouped = defaultdict(list)
    for value in values:
        grouped[value[cluster_field]].append(float(value[field]))
    cluster_means = [float(np.mean(items)) for items in grouped.values()]
    return _bootstrap_mean_interval(cluster_means, seed, samples)


def _mean(values: list[dict], field: str) -> float:
    return sum(float(value[field]) for value in values) / len(values)
