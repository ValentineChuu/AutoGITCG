"""The learned reply selector (docs/architecture.md): at a decision whose replies come on demand, the network
first scores it without replies; its reply-interest output picks the actions whose replies are computed. On a share
of decisions every action is checked, and those give the interest output its targets (how much each action's
probability changed once its replies were known), so the selector is corrected wherever it misses."""

import random
from dataclasses import dataclass

import numpy as np

from autogitcg.bridge.messages import Decision
from autogitcg.observation.encode_actions import MAX_ACTIONS


@dataclass(frozen=True)
class ReplySelection:
    checks: int = 4  # actions whose replies are computed per decision
    full_share: float = 0.15  # decisions where every action is checked (training targets)


@dataclass
class FirstLook:
    decision: Decision
    tracker: object  # the decision's GameTracker (MatchRecorder.on_decision runs once per decision)
    probabilities: np.ndarray  # per action token, without replies
    full: bool  # every action checked: the second look gives interest targets


def checked_actions(interest: np.ndarray, request_indices: list[int], checks: int, full: bool) -> list[int]:
    """Request indices of the actions to check: all of them, or the `checks` with the highest interest."""
    tokens = range(len(request_indices)) if full else np.argsort(-interest[:len(request_indices)])[:checks]
    return [request_indices[token] for token in tokens]


def choose_full(selection: ReplySelection, rng: random.Random) -> bool:
    return rng.random() < selection.full_share


def interest_target(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Per action token (padded to MAX_ACTIONS): how much its probability changed once every action's replies were known."""
    target = np.zeros(MAX_ACTIONS, dtype=np.float32)
    target[:len(second)] = np.abs(second - first[:len(second)])
    return target


def on_demand(view: dict | None) -> dict | None:
    """A seat's reply view with replies on demand (bridge/reply_views.py views are shared constants: copied)."""
    return None if view is None else {**view, "onDemand": True}
