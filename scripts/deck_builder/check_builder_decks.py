"""Are Chudledee's decks strong, or only easy for Chudledum to play? Compared with synergy or real tournament decks.

A (PvP): Chudledee's favourite decks against every opponent deck.
B (PvP): each opponent lineup with Chudledee's card choices against the same lineup with its original cards.
C (PvE, tournament opponents only): sampled stages played with the top tournament decks, with Chudledee's stage
   deck for that stage (built owning everything), and with Chudledee's favourite PvP deck.
Each part is played by every chosen player (the same player on both seats). An edge that holds for every player
is a property of the decks; one that appears only with Chudledum is fitted to how it plays.

Usage:
    python scripts/deck_builder/check_builder_decks.py --opponents tournament --repeats 4
    python scripts/deck_builder/check_builder_decks.py --opponents synergy --players alpha-2.40,heuristic
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from autogitcg.analysis.builder_deck_check.check_decks import builder_cards_for, builder_stage_decks, favourite_decks
from autogitcg.analysis.builder_deck_check.check_games import play_matchups
from autogitcg.analysis.builder_deck_check.check_players import NetworkPlayer, ScriptedPlayer
from autogitcg.analysis.confidence import format_rate, rate_with_interval
from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.deck_building.deck_validation import deck_problems
from autogitcg.observation.observation_encoder import ObservationEncoder
from autogitcg.progression.stage_catalog import build_stage_catalog
from autogitcg.training.deck_pools import load_synergy_decks
from autogitcg.training.matchup_sampler import Matchup
from autogitcg.data.tournament_decks import load_tournament_decks, most_played_per_lineup
from autogitcg.data.tournament_test_set import load_test_decks

PVE_TOURNAMENT_DECKS = 10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--builder-run", type=Path, default=Path("runs/chudledee_v1"))
    parser.add_argument("--combat", type=Path, default=Path("models/alpha-2.40.pt"))
    parser.add_argument("--earlier", type=Path, default=Path("models/alpha-2.20.pt"), help="the alpha-2.20 player's network")
    parser.add_argument("--opponents", choices=["synergy", "tournament", "tournament_test"], default="tournament",
                        help="tournament_test: only the locked test lineups (data/benchmarks/tournament_test_set_v1.json), unseen in training")
    parser.add_argument("--versions", nargs="+", default=["7-0", "7-1"], help="tournament versions")
    parser.add_argument("--opponent-lineups", type=int, default=40, help="tournament lineups used (most played first)")
    parser.add_argument("--stages", type=int, default=40, help="sampled stages for part C")
    parser.add_argument("--favourites", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=4, help="games per matchup and seat order")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--players", default="alpha-2.40,alpha-2.20,heuristic,random")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    output = arguments.output or Path(f"data/derived/builder_deck_check_{arguments.opponents}.json")

    builder = arguments.builder_run / "checkpoint_latest.pt"
    last_update = max(int(path.name.split(".")[0].removeprefix("update_")) for path in (arguments.builder_run / "decks").glob("update_*.jsonl.gz"))
    favourites = favourite_decks(arguments.builder_run, max(1, last_update - 29), last_update, 800, arguments.favourites)  # its last 30 updates
    if arguments.opponents == "synergy":
        opponents, _ = load_synergy_decks()
        opponent_names = [f"synergy {index}" for index in range(len(opponents))]
    else:
        decks = load_test_decks() if arguments.opponents == "tournament_test" else load_tournament_decks(arguments.versions)
        tournament = most_played_per_lineup(decks)[:arguments.opponent_lineups]
        opponents, opponent_names = [deck.deck for deck in tournament], [deck.lineup_name for deck in tournament]
    rebuilt = builder_cards_for([deck.characters for deck in opponents], builder)

    matchups = [(f"A favourite {index + 1}", _pvp(favourite, opponent)) for index, favourite in enumerate(favourites) for opponent in opponents]
    matchups += [("B Chudledee cards vs original cards", _pvp(built, original)) for built, original in zip(rebuilt, opponents)]
    stage_decks = []
    if arguments.opponents != "synergy":
        stages, _ = build_stage_catalog()
        stages = random.Random(7).sample(stages, arguments.stages)
        stage_decks = builder_stage_decks(stages, builder)
        for stage, stage_deck in zip(stages, stage_decks):
            matchups.append(("C Chudledee stage deck", _stage(stage_deck, stage)))
            matchups.append(("C Chudledee favourite 1", _stage(favourites[0], stage)))
            for deck, name in list(zip(opponents, opponent_names))[:PVE_TOURNAMENT_DECKS]:
                matchups.append((f"C tournament: {name}", _stage(deck, stage)))
                matchups.append(("C tournament decks (all)", _stage(deck, stage)))
    assert not any(deck_problems(deck) for deck in favourites + rebuilt + stage_decks)

    encoder = ObservationEncoder()
    players = {"alpha-2.40": lambda: NetworkPlayer(arguments.combat, encoder),
               "alpha-2.20": lambda: NetworkPlayer(arguments.earlier, encoder),
               "heuristic": lambda: ScriptedPlayer(heuristic=True, seed=1),
               "random": lambda: ScriptedPlayer(heuristic=False, seed=1)}
    results = {}
    with EnginePool(arguments.workers) as pool:
        for player_name in arguments.players.split(","):
            # the "all tournament decks" label duplicates games already listed per deck: play each once
            unique = [(label, matchup) for label, matchup in matchups if label != "C tournament decks (all)"]
            rows = play_matchups(pool, encoder, players[player_name](), unique, arguments.repeats, seed=2026)
            rows += [{**row, "label": "C tournament decks (all)"} for row in rows if row["label"].startswith("C tournament: ")]
            results[player_name] = rows
            _print_player(player_name, rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "favourites": [{"characters": deck.characters, "cards": deck.cards} for deck in favourites],
        "opponents": [{"name": name, "characters": deck.characters, "cards": deck.cards} for deck, name in zip(opponents, opponent_names)],
        "rebuilt": [{"characters": deck.characters, "cards": deck.cards} for deck in rebuilt],
        "stage_decks": [{"characters": deck.characters, "cards": deck.cards} for deck in stage_decks],
        "results": results}, indent=1))
    print(f"wrote {output}")


def _pvp(focus, opponent) -> Matchup:
    return Matchup("pvp", (focus, opponent), (), (False, False), ("focus", "opponent"))


def _stage(focus, stage) -> Matchup:
    return Matchup("stage", (focus, stage.enemy_deck), stage.enemy_equipment, (True, False), ("focus", f"stage:{stage.level_id}"))


def _print_player(player_name: str, rows: list[dict]) -> None:
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    print(f"\n{player_name}")
    for label, label_rows in sorted(by_label.items()):
        played = [row for row in label_rows if "failure" not in row]
        wins = np.array([row["focus_won"] for row in played], dtype=float)
        losses = np.mean([not row["focus_won"] and not row["draw"] for row in played])
        print(f"  {label}: win {format_rate(*rate_with_interval(wins)[:3])}, loss {losses:.3f}, "
              f"draws {sum(row['draw'] for row in played)}, failures {len(label_rows) - len(played)}, games {len(played)}, "
              f"mean rounds {np.mean([row['rounds'] for row in played]):.1f}", flush=True)


if __name__ == "__main__":
    main()
