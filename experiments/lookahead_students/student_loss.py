"""The student's losses on a batch of labels (experiments/lookahead_students/notes.md, step 2):

- imitation: cross-entropy toward look-ahead's move where the label rule says to learn it ("imitated");
- anchor: KL(searching network || student) everywhere else, so the student keeps playing like before where
  look-ahead had nothing (reliable) to add;
- value drift: the value output's squared change from the starting network's, so the student stays usable as the
  next searcher (training changes the shared layers even with no value loss: epsilon 2 drifted 0.59 -> 0.82);
- value toward the game's final reward, and the win-chance output toward the plain result (off by default);
- search win: the win-chance output toward the win chance search found (searched decisions only; off by default);
- move value: the move-value output toward each scored candidate's mean search score (off by default). Reported with
  how often its highest-valued candidate is search's best (at decisions where the candidates differ by 0.2 or more).

Also returns accuracies for reporting: how often the student's most likely move is look-ahead's pick, overall
and at the decisions where look-ahead overrode the network (the moves it has to newly learn).
"""

import torch
import torch.nn.functional as functional

from autogitcg.model.combat_network import NetworkOutput
from autogitcg.model.policy_distribution import categorical_logits


def _masked_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp(min=1)


SEPARATED_SPREAD = 0.2


def _move_value_terms(output: NetworkOutput, move_value_targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scored = torch.isfinite(move_value_targets)
    targets = move_value_targets.nan_to_num(0.0)
    values = output.move_values.float()
    loss = (values - targets).pow(2)[scored].mean() if scored.any() else values.sum() * 0
    best_target = targets.masked_fill(~scored, -float("inf"))
    worst_target = targets.masked_fill(~scored, float("inf"))
    separated = (scored.sum(dim=1) >= 2) & (best_target.amax(dim=1) - worst_target.amin(dim=1) >= SEPARATED_SPREAD)
    picks = (values.masked_fill(~scored, -float("inf")).argmax(dim=1) == best_target.argmax(dim=1)).float()
    return loss, _masked_mean(picks, separated.float())


def _centered_move_value_loss(output: NetworkOutput, move_value_targets: torch.Tensor) -> torch.Tensor:
    scored = torch.isfinite(move_value_targets)
    rows = scored.sum(dim=1) >= 2
    if not rows.any():
        return output.move_values.float().sum() * 0
    weights = scored.float()
    values = output.move_values.float() * weights
    targets = move_value_targets.nan_to_num(0.0) * weights
    count = weights.sum(dim=1, keepdim=True).clamp(min=1)
    errors = (values - values.sum(dim=1, keepdim=True) / count) - (targets - targets.sum(dim=1, keepdim=True) / count)
    return (errors.pow(2) * weights)[rows].sum() / weights[rows].sum()


def student_losses(output: NetworkOutput, batch: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    log_probs = categorical_logits(output, batch).log_softmax(dim=1)
    # The move to learn and its identical copies count as one: log of their summed probability.
    equivalent = targets["target_masks"] > 0
    chosen_log_probs = log_probs.masked_fill(~equivalent, float("-inf")).logsumexp(dim=1)
    imitated, overridden = targets["imitated"], targets["overridden"]
    anchored = 1 - imitated

    anchor_probabilities = targets["probabilities"]
    present = (anchor_probabilities > 0) & torch.isfinite(log_probs)
    anchor_rows = torch.where(present, anchor_probabilities * (anchor_probabilities.clamp(min=1e-8).log() - log_probs), 0.0).sum(dim=1)

    move_value_loss, move_value_picks_best = _move_value_terms(output, targets["move_value_targets"])
    picks_label = equivalent.gather(1, log_probs.argmax(dim=1, keepdim=True)).squeeze(1).float()
    return {
        "imitation": _masked_mean(-chosen_log_probs, imitated),
        "anchor": _masked_mean(anchor_rows, anchored),
        "value": (output.value.float() - targets["value_targets"]).pow(2).mean(),
        "value_drift": (output.value.float() - targets["start_values"]).pow(2).mean(),
        "win": functional.binary_cross_entropy_with_logits(output.win_logit.float(), targets["won"]),
        "search_win": _masked_mean(functional.binary_cross_entropy_with_logits(
            output.win_logit.float(), targets["search_win_targets"].clamp(min=0), reduction="none"),
            (targets["search_win_targets"] >= 0).float()),
        "move_value": move_value_loss,
        "centered_move_value": _centered_move_value_loss(output, targets["move_value_targets"]),
        "search_value": _masked_mean((output.value.float() - targets["search_value_targets"].nan_to_num(0.0)).pow(2),
                                     torch.isfinite(targets["search_value_targets"]).float()),
        "move_value_picks_best": move_value_picks_best,
        "imitated_accuracy": _masked_mean(picks_label, imitated),
        "overridden_accuracy": _masked_mean(picks_label, overridden),
    }
