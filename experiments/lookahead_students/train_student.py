"""Trains a student network on recorded look-ahead labels (experiments/lookahead_students/notes.md, step 2).

Saves <output>/student_epoch_NN.pt after every epoch (frozen-model format: loads like models/beta-10.8.pt) and
<output>/metrics.jsonl. Test a student with scripts/evaluation/test_lookahead.py --network <student> --confident-probability 0 (never searches: plain play).

Usage:
    python experiments/lookahead_students/train_student.py --labels runs/teacher_01 --initialize-from models/gamma-5.3.pt --output runs/epsilon_02
"""

import argparse
import dataclasses
import signal
import sys
import json
from pathlib import Path

import torch

from autogitcg.model.combat_network_file import load_combat_network
from experiments.lookahead_students.label_dataset import LABEL_RULES, load_label_sets
from experiments.lookahead_students.student_training import StudentOptions, train_student
from autogitcg.observation.encode_actions import PREVIEW_START
from autogitcg.observation.encode_previews import PREVIEW_WIDTH
from autogitcg.observation.observation_encoder import ObservationEncoder


def main() -> None:
    defaults = StudentOptions()
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, nargs="+", required=True, help="label recording runs")
    parser.add_argument("--initialize-from", type=Path, required=True, help="the student's starting network (model file)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-share", type=float, default=0.1)
    parser.add_argument("--label-rule", choices=LABEL_RULES, default="best", help="the move to learn at searched decisions")
    parser.add_argument("--minimum-probability", type=float, default=0.0,
                        help="moves to learn that the searching network found less likely than this are anchored instead")
    parser.add_argument("--temperature", type=float, default=0.1, help='"improved" and "soft" rules: score difference per factor of e')
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--minibatch-size", type=int, default=defaults.minibatch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--preview-scale", type=float, default=1.0,
                        help="labels with action previews: preview inputs times this, so their zero-started weights learn faster")
    parser.add_argument("--anchor-weight", type=float, default=defaults.anchor_weight)
    parser.add_argument("--value-weight", type=float, default=defaults.value_weight)
    parser.add_argument("--value-anchor-weight", type=float, default=defaults.value_anchor_weight)
    parser.add_argument("--search-win-weight", type=float, default=defaults.search_win_weight,
                        help="train the win-chance output on search's win chance (use with --label-rule evaluator)")
    parser.add_argument("--move-value-loss-weight", type=float, default=defaults.move_value_weight,
                        help="train the move-value output on each candidate's search score (use with --label-rule evaluator)")
    parser.add_argument("--centered-move-value-weight", type=float, default=defaults.centered_move_value_weight,
                        help="train the move-value output on the differences between the scored candidates")
    parser.add_argument("--search-value-weight", type=float, default=defaults.search_value_weight,
                        help="train the value output toward the played move's search score (lower --value-anchor-weight with it)")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    arguments = parser.parse_args()

    options = StudentOptions(arguments.epochs, arguments.minibatch_size, arguments.learning_rate, arguments.anchor_weight,
                             arguments.value_weight, value_anchor_weight=arguments.value_anchor_weight,
                             search_win_weight=arguments.search_win_weight,
                             move_value_weight=arguments.move_value_loss_weight,
                             centered_move_value_weight=arguments.centered_move_value_weight,
                             search_value_weight=arguments.search_value_weight, seed=arguments.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(arguments.seed)
    training, validation = load_label_sets(arguments.labels, arguments.validation_share, arguments.seed, arguments.label_rule,
                                           arguments.minimum_probability, arguments.temperature)
    print(f"{len(training):,} training labels ({int(training.searched.sum()):,} searched, {int(training.imitated.sum()):,} imitated), "
          f"{len(validation):,} validation")
    network = load_combat_network(arguments.initialize_from, ObservationEncoder(), device)
    # Labels recorded with action previews (record_teacher_labels --previews) make a student that reads them; saved in
    # its config, so test and search scripts turn previews on for it.
    action_features = training.observations["action_features"]
    has_preview = PREVIEW_START + PREVIEW_WIDTH - 1  # the preview's last feature: 1 when the action has one
    if action_features.shape[-1] > has_preview and action_features[..., has_preview].max() > 0:
        network.config = dataclasses.replace(network.config, reads_previews=True, preview_scale=arguments.preview_scale)
        print(f"labels carry action previews: the student reads them (preview inputs x{arguments.preview_scale:g})")
    arguments.output.mkdir(parents=True, exist_ok=True)
    # A plain `kill` (SIGTERM) stops like Ctrl+C, so the cleanup runs (recording processes stopped).
    # Exit code 143 (SIGTERM) marks a deliberate stop, so a wrapper that restarts crashed runs leaves it stopped.
    signal.signal(signal.SIGTERM, lambda signal_number, frame: _stop_deliberately(signal_number))

    def on_epoch(epoch: int, metrics: dict) -> None:
        record = {"epoch": epoch, **metrics}
        with open(arguments.output / "metrics.jsonl", "a") as file:
            file.write(json.dumps(record) + "\n")
        shown = metrics["validation"]
        summary = (f"validation: imitation {shown['imitation']:.3f}, anchor {shown['anchor']:.3f}, value drift {shown['value_drift']:.4f}, "
                   f"picks the learned move {shown['imitated_accuracy']:.1%} (overrides {shown['overridden_accuracy']:.1%}), "
                   f"win chance vs search {shown['search_win']:.4f}, vs results {shown['win']:.4f}, "
                   f"value vs search {shown['search_value']:.4f}, centered move value {shown['centered_move_value']:.5f}, "
                   f"move value {shown['move_value']:.4f} (picks search's best {shown['move_value_picks_best']:.1%})")
        print(f"epoch {epoch}: {summary}", flush=True)
        if epoch > 0:
            torch.save({"network_config": dataclasses.asdict(network.config), "network": network.state_dict(),
                        "progress": {"update": epoch}, "frozen_from": f"{arguments.initialize_from} + {arguments.labels} epoch {epoch}"},
                       arguments.output / f"student_epoch_{epoch:02d}.pt")

    train_student(network, training, validation, options, device, on_epoch)


def _stop_deliberately(signal_number: int) -> None:
    print(f"stopped by signal {signal_number}", file=sys.stderr, flush=True)
    raise SystemExit(128 + signal_number)


if __name__ == "__main__":
    main()
