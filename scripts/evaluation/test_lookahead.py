"""Does look-ahead help? A network with look-ahead search against the plain network (default: beta 10.8 both)
on the locked tournament test matchups, deck lists known to both players (docs/search.md).

Usage:
    python scripts/evaluation/test_lookahead.py --matchups 100 --output runs/lookahead_test/default.json
    python scripts/evaluation/test_lookahead.py --matchups 100 --margin 1000 ...   # control: never overrides, so plays like plain beta 10.8
    python scripts/evaluation/test_lookahead.py --greedy --confident-probability 0 ...   # plain beta 10.8 always taking its most likely move
    python scripts/evaluation/test_lookahead.py --blind --guess-builder models/chudledee-v3.4.pt ...   # blind PvP, guessed decks
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import torch

from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.model.combat_network_file import load_combat_network
from autogitcg.observation.enums import DECISION_KINDS
from autogitcg.observation.observation_encoder import ObservationEncoder
from autogitcg.policies.heuristic_choice_policy import HeuristicChoicePolicy
from autogitcg.policies.network_policy import NetworkPolicy
from autogitcg.policies.split_policy import SplitPolicy
from autogitcg.deck_guessing.tournament_deck_guesses import TournamentDeckGuesses
from autogitcg.policies.reply_selection import ReplySelection
from autogitcg.search.deck_guess import build_guesses
from autogitcg.search.lookahead_match import play_lookahead_matches
from autogitcg.search.lookahead_settings import LookaheadSettings
from autogitcg.training.benchmark_matchups import META_BENCHMARK_FILE, TOURNAMENT_BENCHMARK_FILE, load_benchmark_matchups



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=Path, default=Path("models/beta-10.8.pt"), help="the searching player's network")
    parser.add_argument("--opponent", type=Path, default=Path("models/beta-10.8.pt"))
    parser.add_argument("--network-kinds", nargs="+", choices=DECISION_KINDS,
                        help="--network answers only these decision kinds; --fallback-network answers the rest")
    parser.add_argument("--fallback-network", type=Path, default=Path("models/beta-10.8.pt"))
    parser.add_argument("--fallback-heuristic", action="store_true", help="the scripted heuristic answers the kinds --network doesn't")
    parser.add_argument("--matchups", type=int, default=100, help="test matchups (4 games each)")
    parser.add_argument("--skip-matchups", type=int, default=0, help="start after this many test matchups (for running shards in parallel)")
    parser.add_argument("--benchmark", choices=("tournament", "meta"), default="tournament",
                        help="tournament: the locked held-out test decks (their lists left out of deck guesses); meta: the most played "
                             "6.4+ training decks (scripts/data/lock_meta_test.py; guesses keep them, as a real opponent's public list)")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--candidates", type=int, default=LookaheadSettings.candidates)
    parser.add_argument("--redeals", type=int, default=LookaheadSettings.redeals)
    parser.add_argument("--horizon", type=int, default=LookaheadSettings.horizon)
    parser.add_argument("--confident-probability", type=float, default=LookaheadSettings.confident_probability)
    parser.add_argument("--reply-checks", type=int, default=0,
                        help="learned reply selector: replies of this many actions per decision (0 = every action)")
    parser.add_argument("--margin", type=float, default=LookaheadSettings.margin)
    parser.add_argument("--minimum-probability", type=float, default=LookaheadSettings.minimum_probability)
    parser.add_argument("--every-move-kind", action="store_true", help="also try the most likely move of each kind (setup cards too)")
    parser.add_argument("--horizon-rounds", type=int, default=0, help="score copies at this many rounds later (2 = end of next round); --horizon caps it")
    parser.add_argument("--score-by", choices=("win", "value"), default=LookaheadSettings.score_by)
    parser.add_argument("--own-view-only", action="store_true", help="score copies only at the searcher's own decisions")
    parser.add_argument("--greedy", action="store_true", help="the searching player takes its most likely move (opponent still samples)")
    parser.add_argument("--move-value-weight", type=float, default=0.0,
                        help="the searching player's moves: policy x exp(weight x move value) (network_policy.py)")
    parser.add_argument("--move-value-candidates", type=int, default=0, help="only the policy's this many top moves get the move-value bonus (0 = all)")
    parser.add_argument("--blind", action="store_true", help="deck lists unknown to both players (blind PvP)")
    parser.add_argument("--guess-builder", type=Path, help="blind: Chudledee checkpoint that guesses the opponent's deck")
    parser.add_argument("--guesses", type=int, default=4, help="guessed decks per opponent lineup")
    parser.add_argument("--guess-source", choices=("chudledee", "tournament"), default="tournament",
                        help="blind: guess the opponent's deck from similar tournament decks (opponents are modelled from public "
                             "data: about 66%% of hidden cards right, the matchups' own decks left out), or, for comparison only, "
                             "Chudledee (--guess-builder; about 10%%)")
    parser.add_argument("--device", default="", help='e.g. "cuda:1"; default: the first GPU, or the CPU')
    parser.add_argument("--seed", type=int, default=12345, help="game seeds; run several seeds in parallel for more games")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    settings = LookaheadSettings(arguments.candidates, arguments.redeals, arguments.horizon,
                                 arguments.confident_probability, arguments.margin, arguments.minimum_probability,
                                 arguments.every_move_kind, arguments.horizon_rounds, arguments.score_by,
                                 arguments.own_view_only)
    matchups = [dataclasses.replace(matchup, opponent_deck_known=(not arguments.blind, not arguments.blind))
                for matchup in load_benchmark_matchups(META_BENCHMARK_FILE if arguments.benchmark == "meta" else TOURNAMENT_BENCHMARK_FILE)[arguments.skip_matchups:arguments.skip_matchups + arguments.matchups]]
    device = torch.device(arguments.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    encoder = ObservationEncoder()
    network = load_combat_network(arguments.network, encoder, device)
    searcher, rollout = NetworkPolicy(network, device, greedy=arguments.greedy, move_value_weight=arguments.move_value_weight,
                                                     move_value_candidates=arguments.move_value_candidates), NetworkPolicy(network, device, greedy=True)
    if arguments.network_kinds:
        fallback = (HeuristicChoicePolicy(arguments.seed) if arguments.fallback_heuristic else
                    NetworkPolicy(load_combat_network(arguments.fallback_network, encoder, device), device, greedy=arguments.greedy))
        searcher = SplitPolicy(searcher, fallback, set(arguments.network_kinds))
    opponent_network = load_combat_network(arguments.opponent, encoder, device)
    opponent = NetworkPolicy(opponent_network, device)
    previews = network.config.reads_previews or opponent_network.config.reads_previews  # games carry action previews
    replies = network.config.reads_replies or opponent_network.config.reads_replies  # and the opponent's replies

    deck_guesses = None
    if arguments.blind and arguments.guess_source == "tournament":
        held_out = [deck for matchup in matchups for deck in matchup.decks] if arguments.benchmark == "tournament" else []
        deck_guesses = TournamentDeckGuesses(arguments.redeals, arguments.seed, excluded=held_out)
    elif arguments.guess_builder:
        lineups = [tuple(deck.characters) for matchup in matchups for deck in matchup.decks]
        deck_guesses = build_guesses(lineups, arguments.guess_builder, arguments.guesses, arguments.seed)
    total = 4 * len(matchups)
    finished = []

    progress_file = arguments.output.with_suffix(".progress.json")  # for watching a cloud run from the desktop
    progress_file.parent.mkdir(parents=True, exist_ok=True)

    def on_game_finished(row: dict) -> None:
        finished.append(row)
        played = [game for game in finished if "failure" not in game]
        wins = sum(game["searcher_won"] for game in played)
        progress_file.write_text(json.dumps({"finished": len(finished), "total": total, "wins": wins, "played": len(played)}))
        print(f"{len(finished)}/{total} games: look-ahead wins {wins / max(len(played), 1):.1%} of {len(played)}", flush=True)

    started = time.perf_counter()
    with EnginePool(arguments.workers, previews=previews, replies=replies) as pool:
        torch.manual_seed(arguments.seed)
        rows, counts = play_lookahead_matches(pool, encoder, searcher, rollout, opponent, matchups, arguments.seed, settings,
                                              on_game_finished, deck_guesses,
                                              ReplySelection(arguments.reply_checks) if arguments.reply_checks else None)
    played = [row for row in rows if "failure" not in row]
    summary = {
        "settings": dataclasses.asdict(settings), "reply_checks": arguments.reply_checks, "blind": arguments.blind, "greedy": arguments.greedy, "seed": arguments.seed,
        "guess_builder": str(arguments.guess_builder) if arguments.guess_builder else None, "guess_source": arguments.guess_source, "network": str(arguments.network), "opponent": str(arguments.opponent),
        "network_kinds": arguments.network_kinds, "fallback_network": ("heuristic" if arguments.fallback_heuristic else str(arguments.fallback_network)) if arguments.network_kinds else None,
        "games": len(rows), "failures": len(rows) - len(played),
        "win_rate": sum(row["searcher_won"] for row in played) / max(len(played), 1),
        "mean_reward": sum(row["searcher_reward"] for row in played) / max(len(played), 1),
        "seconds": time.perf_counter() - started, "counts": dict(counts),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
