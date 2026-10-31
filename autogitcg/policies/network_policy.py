"""Plays with a CombatNetwork: one batched forward pass answers many pending decisions at once."""

import os
import dataclasses
from dataclasses import dataclass

import numpy as np
import torch

from autogitcg.bridge.messages import Decision
from autogitcg.model.combat_network import CombatNetwork
from autogitcg.model.observation_batch import stack, to_device
from autogitcg.model.policy_distribution import sample_choices
from autogitcg.observation.encode_dice import MAX_DICE
from autogitcg.observation.encode_hand import MAX_HAND_PER_SIDE
from autogitcg.observation.enums import DECISION_KINDS
from autogitcg.observation.observation_encoder import EncodedDecision
from autogitcg.policies.network_responses import response_from_choice


@dataclass
class NetworkChoice:
    response: dict
    kind: int  # index in DECISION_KINDS
    choice_index: int
    choice_bits: np.ndarray  # float32 (CHOICE_BITS,)
    log_prob: float  # behaviour (with exploration: the mixture actually sampled from)
    policy_log_prob: float  # the policy's own
    value: float  # expected (speed-adjusted) reward
    win_probability: float  # plain win chance from the separate win output
    entropy: float
    probabilities: np.ndarray  # one per option: action tokens in request order, or each die / opening-hand card
    reply_interest: np.ndarray | None = None  # per action token: the learned reply selector's score (policies/reply_selection.py)


# Decisions per forward pass: look-ahead can have thousands pending, which ran a 12 GB GPU out of memory. Several
# searching processes sharing one GPU set AUTOGITCG_MAX_BATCH lower (128: about 1 GB each instead of 4-5).
MAX_BATCH = int(os.environ.get("AUTOGITCG_MAX_BATCH", 1024))


class NetworkPolicy:
    def __init__(self, network: CombatNetwork, device: torch.device, greedy: bool = False, exploration: float = 0.0,
                 move_value_weight: float = 0.0, move_value_candidates: int = 0):
        """exploration: see policy_distribution.sample_choices (training only; log_prob is then the behaviour's).
        move_value_weight: action-like choices are made from the policy's scores + weight x the move-value output, i.e.
        the policy times exp(weight x move value); 0 = the policy alone, large = the move with the highest value.
        move_value_candidates: > 0 = only the policy's this many most likely moves get the move-value bonus (the move
        values are trained only on the moves search tried, mostly the policy's top ones)."""
        self.network = network
        self.device = device
        self.greedy = greedy
        self.exploration = exploration
        self.move_value_weight = move_value_weight
        self.move_value_candidates = move_value_candidates

    def choose(self, pending: list[tuple[Decision, EncodedDecision]]) -> list[NetworkChoice]:
        return [choice for start in range(0, len(pending), MAX_BATCH) for choice in self._choose_batch(pending[start:start + MAX_BATCH])]

    @torch.no_grad()
    def _choose_batch(self, pending: list[tuple[Decision, EncodedDecision]]) -> list[NetworkChoice]:
        if not pending:
            return []
        self.network.eval()
        batch = to_device(stack([encoded.arrays for _, encoded in pending]), self.device)
        kinds = torch.tensor([DECISION_KINDS.index(decision.kind) for decision, _ in pending], device=self.device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            output = self.network(batch)
        if self.move_value_weight:
            output = dataclasses.replace(output, action_logits=_with_move_values(output, self.move_value_weight, self.move_value_candidates))
        sampled = sample_choices(output, batch, kinds, greedy=self.greedy, exploration=self.exploration)
        choice_indices = sampled.choice_index.cpu().numpy()
        choice_bits = sampled.choice_bits.cpu().numpy()
        log_probs = sampled.log_prob.cpu().numpy()
        policy_log_probs = sampled.policy_log_prob.cpu().numpy()
        values = output.value.float().cpu().numpy()
        win_probabilities = torch.sigmoid(output.win_logit.float()).cpu().numpy()
        entropies = sampled.entropy.cpu().numpy()
        probabilities = sampled.probabilities.cpu().numpy()
        interest = output.reply_interest.float().cpu().numpy() if output.reply_interest is not None else None
        return [
            NetworkChoice(response_from_choice(decision, encoded, int(choice_indices[row]), choice_bits[row].tolist()),
                          int(kinds[row]), int(choice_indices[row]), choice_bits[row], float(log_probs[row]), float(policy_log_probs[row]), float(values[row]),
                          float(win_probabilities[row]), float(entropies[row]), probabilities[row, :option_count(decision, encoded)],
                          interest[row, :len(encoded.request_indices)] if interest is not None else None)
            for row, (decision, encoded) in enumerate(pending)
        ]


def option_count(decision: Decision, encoded: EncodedDecision) -> int:
    """How many options the decision offers: action tokens, or dice / opening-hand cards for yes/no kinds."""
    me = decision.state["player"][decision.who]
    if decision.kind == "rerollDice":
        return min(len(me.get("dice", [])), MAX_DICE)
    if decision.kind == "switchHands":
        return min(len(me.get("handCard", [])), MAX_HAND_PER_SIDE)
    return len(encoded.request_indices)


def _with_move_values(output, weight: float, candidates: int) -> torch.Tensor:
    """The action logits plus weight x the move values, for every move or only the `candidates` most likely ones."""
    logits = output.action_logits.float()
    bonus = weight * output.move_values.float()
    if candidates > 0:
        top = logits.topk(min(candidates, logits.shape[1]), dim=1).indices
        bonus = torch.zeros_like(bonus).scatter(1, top, bonus.gather(1, top))
    return logits + bonus
