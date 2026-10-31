"""Turns network scores into a probability distribution over one decision's choices, for every decision kind.

- action / chooseActive / selectCard: pick one action token (categorical over the action mask).
- rerollDice: reroll each die or not (independent yes/no per die token).
- switchHands: swap each opening-hand card or not (independent yes/no per own hand token).

A choice is stored as (choice_index, choice_bits): the token index for categorical kinds, a 0/1 vector for
the yes/no kinds (padded to MAX_DICE). Batches can mix kinds.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from autogitcg.model.combat_network import NetworkOutput
from autogitcg.observation.encode_dice import MAX_DICE
from autogitcg.observation.encode_hand import MAX_HAND_PER_SIDE
from autogitcg.observation.enums import DECISION_KINDS

REROLL_KIND = DECISION_KINDS.index("rerollDice")
SWITCH_HANDS_KIND = DECISION_KINDS.index("switchHands")
CHOICE_BITS = MAX_DICE


@dataclass
class SampledChoices:
    choice_index: torch.Tensor  # (batch,) long
    choice_bits: torch.Tensor  # (batch, CHOICE_BITS) float 0/1
    log_prob: torch.Tensor  # (batch,) under the distribution actually sampled from (the behaviour)
    policy_log_prob: torch.Tensor  # (batch,) under the policy itself (differs from log_prob only with exploration)
    entropy: torch.Tensor  # (batch,)
    probabilities: torch.Tensor  # (batch, MAX_ACTIONS): per action token, or per die / hand card for yes/no kinds


def categorical_logits(output: NetworkOutput, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    mask = batch["action_mask"] > 0
    mask[:, 0] |= ~mask.any(dim=1)  # yes/no kinds have no action tokens; keep the softmax defined
    return output.action_logits.float().masked_fill(~mask, float("-inf"))


def yes_no_logits_and_mask(output: NetworkOutput, batch: dict[str, torch.Tensor], kinds: torch.Tensor):
    hand_logits = functional.pad(output.hand_logits.float(), (0, CHOICE_BITS - MAX_HAND_PER_SIDE))
    hand_mask = functional.pad(batch["hand_mask"][:, :MAX_HAND_PER_SIDE] > 0, (0, CHOICE_BITS - MAX_HAND_PER_SIDE))
    is_switch = (kinds == SWITCH_HANDS_KIND).unsqueeze(1)
    logits = torch.where(is_switch, hand_logits, output.dice_logits.float())
    mask = torch.where(is_switch, hand_mask, batch["dice_mask"] > 0)
    return logits, mask


def is_yes_no(kinds: torch.Tensor) -> torch.Tensor:
    return (kinds == REROLL_KIND) | (kinds == SWITCH_HANDS_KIND)


def sample_choices(output: NetworkOutput, batch: dict[str, torch.Tensor], kinds: torch.Tensor, greedy: bool = False,
                   exploration: float = 0.0) -> SampledChoices:
    """exploration: for categorical kinds, sample from (1 - exploration) x policy + exploration x uniform over the legal
    options, and return that mixture's log-probability (the behaviour PPO must correct for). Moves the policy almost
    never picks get tried now and then, so training can find out whether they're good."""
    action_logits = categorical_logits(output, batch)
    yes_no_logits, yes_no_mask = yes_no_logits_and_mask(output, batch, kinds)
    behaviour = None
    if greedy:
        choice_index = action_logits.argmax(dim=1)
        choice_bits = (yes_no_logits > 0).float() * yes_no_mask
    else:
        if exploration > 0:
            legal = (batch["action_mask"] > 0).float()
            behaviour = (1 - exploration) * action_logits.softmax(dim=1) + exploration * legal / legal.sum(dim=1, keepdim=True).clamp(min=1)
            choice_index = torch.distributions.Categorical(probs=behaviour).sample()
        else:
            choice_index = torch.distributions.Categorical(logits=action_logits).sample()
        choice_bits = torch.bernoulli(torch.sigmoid(yes_no_logits)) * yes_no_mask
    log_prob, entropy = evaluate_choices(output, batch, kinds, choice_index, choice_bits)
    policy_log_prob = log_prob
    if behaviour is not None:
        behaviour_log_prob = behaviour.gather(1, choice_index.unsqueeze(1)).squeeze(1).clamp(min=1e-12).log()
        log_prob = torch.where(is_yes_no(kinds), log_prob, behaviour_log_prob)
    yes_no_probabilities = functional.pad(torch.sigmoid(yes_no_logits) * yes_no_mask,
                                          (0, action_logits.shape[1] - CHOICE_BITS))
    probabilities = torch.where(is_yes_no(kinds).unsqueeze(1), yes_no_probabilities, action_logits.softmax(dim=1))
    return SampledChoices(choice_index, choice_bits, log_prob, policy_log_prob, entropy, probabilities)


def evaluate_choices(output: NetworkOutput, batch: dict[str, torch.Tensor], kinds: torch.Tensor,
                     choice_index: torch.Tensor, choice_bits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Log probability of the given choices and the entropy of each distribution."""
    categorical = torch.distributions.Categorical(logits=categorical_logits(output, batch))
    categorical_log_prob = categorical.log_prob(choice_index)
    categorical_entropy = categorical.entropy()

    yes_no_logits, yes_no_mask = yes_no_logits_and_mask(output, batch, kinds)
    yes_no = torch.distributions.Bernoulli(logits=yes_no_logits)
    yes_no_log_prob = (yes_no.log_prob(choice_bits) * yes_no_mask).sum(dim=1)
    yes_no_entropy = (yes_no.entropy() * yes_no_mask).sum(dim=1)

    yes_no_rows = is_yes_no(kinds)
    return (torch.where(yes_no_rows, yes_no_log_prob, categorical_log_prob),
            torch.where(yes_no_rows, yes_no_entropy, categorical_entropy))
