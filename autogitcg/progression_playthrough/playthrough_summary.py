"""Measures of one playthrough arm, from its match rows: how fast accounts progress and where they get stuck."""

import gzip
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from autogitcg.deck_building.deck_validation import buildable_characters

CHECKPOINT_MATCHES = (100, 300, 600, 1000)


def load_matches(run: Path) -> dict[int, list[dict]]:
    """Match rows per account, in play order."""
    accounts = defaultdict(list)
    for path in sorted(run.glob("matches*.jsonl.gz")):
        for row in _rows_until_cut(path):
            accounts[row["account"]].append(row)
    for rows in accounts.values():
        rows.sort(key=lambda row: row["match"])
    return dict(accounts)


def _rows_until_cut(path: Path):
    """Complete rows of one file; a run stopped by hand leaves the gzip stream (and maybe the last line) cut off."""
    with gzip.open(path, "rt") as file:
        try:
            for line in file:
                if line.endswith("\n"):
                    yield json.loads(line)
        except EOFError:
            return


def progress_measures(accounts: dict[int, list[dict]]) -> dict:
    full = len(buildable_characters())
    finished = {account: next((row for row in rows if row["characters_after"] >= full), None) for account, rows in accounts.items()}
    half = {account: next((row for row in rows if row["characters_after"] >= full / 2), None) for account, rows in accounts.items()}
    measures = {
        "accounts": len(accounts),
        "matches": sum(len(rows) for rows in accounts.values()),
        "win rate": sum(row["won"] for rows in accounts.values() for row in rows) / max(1, sum(len(rows) for rows in accounts.values())),
        "accounts reaching the full roster": sum(row is not None for row in finished.values()),
        "matches to half the roster (median)": _median([row["match"] for row in half.values() if row]),
        "matches to the full roster (median)": _median([row["match"] for row in finished.values() if row]),
        "matches to the full roster (90th percentile)": _percentile([row["match"] for row in finished.values() if row], 0.9),
        "weeks to the full roster (median)": _median([row["week"] for row in finished.values() if row]),
    }
    for checkpoint in CHECKPOINT_MATCHES:
        at = [rows[checkpoint - 1] for rows in accounts.values() if len(rows) >= checkpoint]
        if at:
            measures[f"characters at match {checkpoint} (median)"] = _median([row["characters"] for row in at])
            measures[f"level at match {checkpoint} (median)"] = _median([row["level"] for row in at])
    return measures


def stage_kind_measures(accounts: dict[int, list[dict]]) -> dict[str, dict]:
    """Per stage kind: win rate over all attempts, and over each account's first attempt at each stage."""
    attempts, wins, first_attempts, first_wins = Counter(), Counter(), Counter(), Counter()
    for rows in accounts.values():
        seen = set()
        for row in rows:
            attempts[row["kind"]] += 1
            wins[row["kind"]] += row["won"]
            if row["level_id"] not in seen:
                seen.add(row["level_id"])
                first_attempts[row["kind"]] += 1
                first_wins[row["kind"]] += row["won"]
    return {kind: {"matches": attempts[kind], "win rate": wins[kind] / attempts[kind],
                   "first-attempt win rate": first_wins[kind] / max(1, first_attempts[kind])} for kind in sorted(attempts)}


def hardest_stages(accounts: dict[int, list[dict]], count: int = 12) -> list[dict]:
    """Stages with the most losses before the first win, summed over accounts."""
    losses_before_win, never_beaten, names, kinds = Counter(), Counter(), {}, {}
    for rows in accounts.values():
        beaten, losses = set(), Counter()
        for row in rows:
            names[row["level_id"]], kinds[row["level_id"]] = row["stage"], row["kind"]
            if row["level_id"] in beaten:
                continue
            if row["won"]:
                beaten.add(row["level_id"])
            else:
                losses[row["level_id"]] += 1
        for level_id, lost in losses.items():
            losses_before_win[level_id] += lost
            never_beaten[level_id] += level_id not in beaten
    return [{"stage": names[level_id], "kind": kinds[level_id], "losses before first win": lost, "accounts never beating it": never_beaten[level_id]}
            for level_id, lost in losses_before_win.most_common(count)]


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _percentile(values: list[float], fraction: float) -> float | None:
    return sorted(values)[min(len(values) - 1, int(fraction * len(values)))] if values else None
