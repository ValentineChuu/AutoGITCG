"""How each network plays, as a table printed to the terminal: one column per network, from the games of one or more
duels (scripts/evaluation/judge_duel.py writes them to <duel>/games/). Used to see why a network's results changed:
for example, whether a card-play reward really made it play more cards, or whether it holds its bursts.

Rows, per game: win rate (overall and per seat), rounds, card plays (split by kind: talent, weapon, artifact,
technique, food, resonance, other equipment, support, other event), tunings, switches, skills by type and rounds
ended. Then rates: how often it plays a card when it could, ends the round still holding a playable card, uses a
ready burst; and dice and cards left when it ends the round. Last, each network's most played cards. A network's
games are pooled by its name across the given duels (autogitcg/analysis/duel/play_style.py counts them).

    python scripts/evaluation/duel_play_style.py runs/duel/theta_vs_beta runs/duel/eta_vs_beta
"""

import argparse
from pathlib import Path

from autogitcg.analysis.duel.play_style import CARD_KINDS, read_styles
from autogitcg.analysis.definition_names import definition_names

PER_GAME = ["card plays", *(f"card plays: {kind}" for kind in (*CARD_KINDS, "equipment", "support", "event")), "tunings", "switches",
            "skills: normal", "skills: elemental", "skills: burst", "round ends"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("duels", type=Path, nargs="+", help="duel output folders (with games/*.jsonl.gz)")
    parser.add_argument("--top-cards", type=int, default=8)
    arguments = parser.parse_args()
    styles = read_styles(sorted(path for duel in arguments.duels for path in (duel / "games").glob("*.jsonl.gz")))
    names = list(styles)
    width = max(14, *(len(name) + 2 for name in names))
    print(f"{'':42}" + "".join(f"{name:>{width}}" for name in names))
    rows = [("games", lambda s: f"{s.games}"),
            ("win rate", lambda s: f"{100 * s.wins / s.games:.1f}%"),
            ("win rate going first (seat 0)", lambda s: f"{100 * s.wins_by_seat[0] / max(s.games_by_seat[0], 1):.1f}%"),
            ("win rate going second (seat 1)", lambda s: f"{100 * s.wins_by_seat[1] / max(s.games_by_seat[1], 1):.1f}%"),
            ("rounds per game", lambda s: f"{s.rounds / s.games:.2f}")]
    rows += [(f"{key} per game", lambda s, key=key: f"{s.counts[key] / s.games:.2f}") for key in PER_GAME]
    rows += [("plays a card when it can", lambda s: f"{100 * s.counts['played a card when it could'] / max(s.counts['decisions with a playable card'], 1):.1f}%"),
             ("ends the round with a playable card", lambda s: f"{100 * s.counts['round ends with a playable card'] / max(s.counts['round ends'], 1):.1f}%"),
             ("burst used when ready", lambda s: f"{100 * s.counts['burst used when ready'] / max(s.counts['decisions with a burst ready'], 1):.1f}%"),
             ("dice left at round end", lambda s: f"{s.counts['dice left at round end'] / max(s.counts['round ends'], 1):.2f}"),
             ("hand left at round end", lambda s: f"{s.counts['hand left at round end'] / max(s.counts['round ends'], 1):.2f}")]
    for label, value in rows:
        print(f"{label:42}" + "".join(f"{value(styles[name]):>{width}}" for name in names))
    card_names = {definition_id: name for definition_id, kind, name in definition_names() if kind == "card"}
    for name in names:
        top = styles[name].cards.most_common(arguments.top_cards)
        print(f"\n{name} most played per game: " + ", ".join(f"{card_names.get(card, card)} {count / styles[name].games:.2f}" for card, count in top))


if __name__ == "__main__":
    main()
