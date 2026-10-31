"""Train a combat network (Chudledum) with PPO self-play. docs/training.md explains the method and every setting.

A run lives in its folder (--run): metrics.jsonl, checkpoints, the network weights every 5 updates and every game
played. Running the same command again resumes from the latest checkpoint.

From scratch, with the default settings:
    python scripts/training/train.py --run runs/my_run --updates 1000

Theta 1, the PvP network: beta 10.8 fine-tuned on tournament decks, reading the opponent's replies:
    python scripts/training/train.py --run runs/theta_01 --updates 1000 --seed 7 \
        --initialize-from models/beta-10.8.pt --reference models/beta-10.8.pt \
        --replies --preview-scale 10 --reply-checks 4 \
        --deck-sources tournament=0.9,chudledee=0.07,stage=0.03 \
        --human-deck-opponents tournament=0.9,chudledee=0.1 --tournament-from 6-4 \
        --speed-penalty 0 --entropy-coefficient 0 --gae-lambda 0.8 --learning-rate 5e-5

The defaults fit a network learning from scratch. Past the beginner stage, a network stops improving with them (the
entropy bonus outweighs its faint advantage signal), so a fine-tune needs entropy 0, lambda 0.8 and a learning rate
of 5e-5, as above.
"""

import argparse
import os
import signal
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before torch loads; actors inherit it

from autogitcg.data.card_kinds import CARD_KINDS
from autogitcg.policies.reply_selection import ReplySelection
from autogitcg.training.ppo_config import PPOConfig
from autogitcg.training.trainer import TrainingOptions, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    run = parser.add_argument_group("run and machine")
    run.add_argument("--run", type=Path, required=True, help="run folder: metrics, checkpoints, weights, games")
    run.add_argument("--updates", type=int, default=1000, help="train until this update (also when resuming)")
    run.add_argument("--seed", type=int, default=0, help="use a new seed for each run")
    run.add_argument("--actors", type=int, default=TrainingOptions.actors, help="processes playing games")
    run.add_argument("--workers-per-actor", type=int, default=TrainingOptions.workers_per_actor, help="engine workers per actor")
    run.add_argument("--games-per-actor", type=int, default=TrainingOptions.concurrent_games_per_actor, help="games each actor plays at once")
    run.add_argument("--actor-device", default="", help='GPU for game play, e.g. "cuda:1" (default: the second GPU if there is one)')
    run.add_argument("--no-compile", action="store_true", help="don't torch.compile the learner (beta 1 crashed silently with it)")
    run.add_argument("--checkpoint-every", type=int, default=TrainingOptions.checkpoint_every)

    network = parser.add_argument_group("network and inputs")
    network.add_argument("--initialize-from", type=Path, help="start a fresh run from a trained network (a model file or checkpoint)")
    network.add_argument("--width", type=int, default=TrainingOptions.network_width, help="new network width (fresh runs only)")
    network.add_argument("--layers", type=int, default=TrainingOptions.network_layers, help="new network layers (fresh runs only)")
    network.add_argument("--previews", action="store_true", help="the network reads action previews (fresh runs; games about 2x slower)")
    network.add_argument("--replies", action="store_true",
                         help="it also reads the opponent's replies to each action (implies --previews; fresh runs; games much slower)")
    network.add_argument("--preview-scale", type=float, default=1.0,
                         help="preview and reply inputs times this, so their zero-started weights learn faster (fresh runs)")
    network.add_argument("--reply-checks", type=int, default=0,
                         help="learned reply selector: replies of this many actions per decision (0 = every action; docs/architecture.md)")
    network.add_argument("--full-reply-share", type=float, default=ReplySelection.full_share,
                         help="decisions where every action is checked anyway (the selector's training targets)")
    network.add_argument("--reply-interest-coefficient", type=float, default=1.0, help="weight of the selector's loss (with --reply-checks)")

    ppo_options = parser.add_argument_group("PPO (the defaults fit training from scratch; fine-tune with entropy 0, lambda 0.8, learning rate 5e-5)")
    ppo_options.add_argument("--steps-per-update", type=int, default=PPOConfig.steps_per_update)
    ppo_options.add_argument("--minibatch-size", type=int, default=PPOConfig.minibatch_size)
    ppo_options.add_argument("--learning-rate", type=float, default=PPOConfig.learning_rate)
    ppo_options.add_argument("--final-learning-rate", type=float, help="linear decay target (default: constant)")
    ppo_options.add_argument("--entropy-coefficient", type=float, default=PPOConfig.entropy_coefficient)
    ppo_options.add_argument("--gae-lambda", type=float, default=PPOConfig.gae_lambda, help="lower = less outcome noise, more trust in the value estimate")
    ppo_options.add_argument("--speed-penalty", type=float, default=TrainingOptions.speed_penalty,
                             help="per round taken off a win's reward; 0 = plain win/loss (use 0 for PvP)")

    games = parser.add_argument_group("decks and opponents (the defaults are the NPC-fight mix; docs/training.md has the PvP one)")
    games.add_argument("--deck-sources", help='deck-source mix, e.g. "tournament=0.9,chudledee=0.07,stage=0.03"')
    games.add_argument("--human-deck-opponents", help='opponent decks for tournament/Chudledee decks, e.g. "tournament=0.9,chudledee=0.1"')
    games.add_argument("--tournament-from", help='training tournament decks only from this game version on, e.g. "6-4" (default: all)')
    games.add_argument("--opponents", help='opponent mix, e.g. "reference=1" or "self=0.5,reference=0.5"')
    games.add_argument("--reference", default=str(TrainingOptions.reference),
                       help="frozen network for mirror-pair difficulty and the tournament benchmark; 'none' to leave out")
    games.add_argument("--accounts", type=int, default=64, help="simulated accounts whose collections give the collection decks")

    evaluation = parser.add_argument_group("benchmarks during training")
    evaluation.add_argument("--evaluate-every", type=int, default=TrainingOptions.evaluate_every, help="0 = never")
    evaluation.add_argument("--evaluation-matchups", type=int, default=TrainingOptions.evaluation_matchups)
    evaluation.add_argument("--tournament-evaluation-matchups", type=int, default=TrainingOptions.tournament_evaluation_matchups)

    dropped = parser.add_argument_group("dropped experiments (off by default; docs/training.md has what each run found)")
    dropped.add_argument("--teacher", type=Path, help="kickstarting (gamma 4): frozen network to imitate at first")
    dropped.add_argument("--teacher-weight", type=float, default=TrainingOptions.teacher_weight)
    dropped.add_argument("--teacher-fade-updates", type=int, default=TrainingOptions.teacher_fade_updates)
    dropped.add_argument("--distill-updates", type=int, default=TrainingOptions.distill_updates,
                         help="copying first (gamma 5): the first updates only imitate the teacher")
    dropped.add_argument("--distill-learning-rate", type=float, default=TrainingOptions.distill_learning_rate)
    dropped.add_argument("--exploration", type=float, default=TrainingOptions.exploration,
                         help="trying rare moves (gamma 6): share of a uniform choice mixed into the learner's choices")
    dropped.add_argument("--shaping-weight", type=float, default=TrainingOptions.shaping_weight,
                         help="HP reward (gamma 9): weight x change in (my health share - opponent's) after each move")
    dropped.add_argument("--card-play-bonus", type=float, default=TrainingOptions.card_play_bonus,
                         help="card rewards (delta 3-8): reward per action card played (a win is worth about 1)")
    dropped.add_argument("--card-kind-bonus", help='extra reward per card kind played, e.g. "events=0.01,resonances=0.02" (data/card_kinds.py)')
    dropped.add_argument("--future-coefficient", type=float, default=PPOConfig.future_coefficient,
                         help="board prediction (eta 2-3): weight of predicting the board a few own decisions after each move")
    dropped.add_argument("--move-value-coefficient", type=float, default=PPOConfig.move_value_coefficient,
                         help="move values (with the students): train the move-value output on final results")
    arguments = parser.parse_args()
    unknown_kinds = set(_named_weights(arguments.card_kind_bonus) or {}) - set(CARD_KINDS)
    if unknown_kinds:
        parser.error(f"unknown card kinds {sorted(unknown_kinds)}; known: {', '.join(CARD_KINDS)}")

    options = TrainingOptions(
        run_directory=arguments.run, updates=arguments.updates, actors=arguments.actors,
        workers_per_actor=arguments.workers_per_actor, concurrent_games_per_actor=arguments.games_per_actor,
        evaluate_every=arguments.evaluate_every, checkpoint_every=arguments.checkpoint_every, evaluation_matchups=arguments.evaluation_matchups,
        tournament_evaluation_matchups=arguments.tournament_evaluation_matchups, accounts=arguments.accounts, seed=arguments.seed,
        actor_device=arguments.actor_device, compile_learner=not arguments.no_compile,
        network_width=arguments.width, network_layers=arguments.layers, teacher=arguments.teacher,
        teacher_weight=arguments.teacher_weight, teacher_fade_updates=arguments.teacher_fade_updates,
        distill_updates=arguments.distill_updates, distill_learning_rate=arguments.distill_learning_rate,
        exploration=arguments.exploration, speed_penalty=arguments.speed_penalty, shaping_weight=arguments.shaping_weight,
        opponent_weights=_named_weights(arguments.opponents), source_weights=_named_weights(arguments.deck_sources),
        human_deck_opponents=_named_weights(arguments.human_deck_opponents), tournament_from=arguments.tournament_from, card_play_bonus=arguments.card_play_bonus,
        card_kind_bonus=_named_weights(arguments.card_kind_bonus),
        initialize_from=arguments.initialize_from, previews=arguments.previews, replies=arguments.replies, preview_scale=arguments.preview_scale,
        reply_selection=ReplySelection(arguments.reply_checks, arguments.full_reply_share) if arguments.reply_checks else None, reference=None if arguments.reference == "none" else Path(arguments.reference),
    )
    ppo = PPOConfig(steps_per_update=arguments.steps_per_update, minibatch_size=arguments.minibatch_size,
                    learning_rate=arguments.learning_rate, entropy_coefficient=arguments.entropy_coefficient,
                    gae_lambda=arguments.gae_lambda, move_value_coefficient=arguments.move_value_coefficient,
                    future_coefficient=arguments.future_coefficient,
                    reply_interest_coefficient=arguments.reply_interest_coefficient if arguments.reply_checks else 0.0,
                    reply_checks=arguments.reply_checks or ReplySelection.checks,
                    final_learning_rate=arguments.final_learning_rate or arguments.learning_rate)
    # A plain `kill` (SIGTERM) stops like Ctrl+C: the actor processes are shut down. Exit code 143
    # (the shell convention for SIGTERM) marks a deliberate stop, so a wrapper that restarts crashed runs leaves it stopped.
    signal.signal(signal.SIGTERM, lambda signal_number, frame: _stop_deliberately(signal_number))
    train(options, ppo)


def _named_weights(text: str | None) -> dict[str, float] | None:
    """"self=0.5,reference=0.5" -> {"self": 0.5, "reference": 0.5} (opponent and deck-source mixes)."""
    if not text:
        return None
    return {name: float(weight) for name, weight in (part.split("=") for part in text.split(","))}


def _stop_deliberately(signal_number: int) -> None:
    print(f"stopped by signal {signal_number}", file=sys.stderr, flush=True)
    raise SystemExit(128 + signal_number)


if __name__ == "__main__":
    main()
