"""Kickstarting: a new (e.g. larger) network learns from a frozen teacher while it trains with PPO.

An extra loss pulls the student's choice distributions toward the teacher's on the student's own games
(KL(teacher || student): categorical for action-like kinds, per-item yes/no for rerolls and mulligans), and
its value toward the teacher's value. Its weight fades to zero over the first updates, so the student starts
from the teacher's skill and is then free to surpass it (Schmitt et al. 2018, "Kickstarting deep RL").
"""

import torch
import torch.nn.functional as functional

from autogitcg.model.combat_network import NetworkOutput
from autogitcg.model.policy_distribution import categorical_logits, is_yes_no, yes_no_logits_and_mask


def teacher_divergence(student: NetworkOutput, teacher: NetworkOutput, batch: dict[str, torch.Tensor],
                       kinds: torch.Tensor) -> torch.Tensor:
    """Per row: KL(teacher || student) of the decision's choice distribution."""
    student_log_probs = categorical_logits(student, batch).log_softmax(dim=1)
    teacher_log_probs = categorical_logits(teacher, batch).log_softmax(dim=1)
    teacher_probs = teacher_log_probs.exp()
    finite = torch.isfinite(teacher_log_probs)  # masked-out tokens: probability 0 for both
    categorical = torch.where(finite, teacher_probs * (teacher_log_probs - student_log_probs), 0.0).sum(dim=1)

    student_yes_no, mask = yes_no_logits_and_mask(student, batch, kinds)
    teacher_yes_no, _ = yes_no_logits_and_mask(teacher, batch, kinds)
    teacher_yes = torch.sigmoid(teacher_yes_no)
    per_item = (teacher_yes * (functional.logsigmoid(teacher_yes_no) - functional.logsigmoid(student_yes_no))
                + (1 - teacher_yes) * (functional.logsigmoid(-teacher_yes_no) - functional.logsigmoid(-student_yes_no)))
    yes_no = (per_item * mask).sum(dim=1)
    return torch.where(is_yes_no(kinds), yes_no, categorical)


def kickstarting_loss(student: NetworkOutput, teacher: NetworkOutput, batch: dict[str, torch.Tensor],
                      kinds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(mean policy divergence, mean squared value difference) to the teacher."""
    divergence = teacher_divergence(student, teacher, batch, kinds).mean()
    value_difference = (student.value.float() - teacher.value.float()).pow(2).mean()
    return divergence, value_difference


def teacher_weight(start: float, fade_updates: int, update: int) -> float:
    """Linear fade from `start` at update 0 to 0 at `fade_updates`."""
    return start * max(0.0, 1.0 - update / max(fade_updates, 1))
