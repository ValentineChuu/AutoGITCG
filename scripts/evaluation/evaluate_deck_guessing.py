"""How well each deck guesser predicts a blind opponent's hidden cards (docs/search.md).

Real tournament decks from the locked test lineups play the opponent. For each, k of its cards are revealed (a
random sample: a simplification of the order cards get played), and each guesser predicts the other 30 - k:
  lineup     Chudledee from the lineup only (what look-ahead uses today, search/deck_guess.py)
  chudledee  Chudledee decks started from the revealed cards
  tournament the most similar tournament decks, the opponent's own list left out
  mix        chudledee and tournament, half each
Scores: share of the hidden cards the guess covers, and mean log-likelihood of a hidden card.

    CUDA_VISIBLE_DEVICES="" python scripts/evaluation/evaluate_deck_guessing.py --decks 100 --output docs/results/deck_guessing_v1.json
"""

import argparse
import json
import random
import statistics
import time
from collections import Counter
from pathlib import Path

import torch

from autogitcg.data.deck import Deck
from autogitcg.data.tournament_decks import downloaded_versions, load_tournament_decks
from autogitcg.deck_builder.lineup_feasibility import lineup_is_feasible
from autogitcg.deck_building.deck_rules import DECK_ACTION_CARDS
from autogitcg.deck_building.deck_validation import buildable_action_cards, buildable_characters
from autogitcg.deck_guessing.chudledee_guess import ChudledeeGuesser
from autogitcg.deck_guessing.guess_quality import mean_log_likelihood, overlap_share
from autogitcg.deck_guessing.remaining_cards import expected_remaining
from autogitcg.deck_guessing.tournament_guess import TournamentGuesser

TEST_SET = Path("data/benchmarks/tournament_test_set_v1.json")
GUESSERS = ("lineup", "chudledee", "tournament", "mix")


def _mixed(first: list[tuple[float, Counter]], second: list[tuple[float, Counter]]) -> list[tuple[float, Counter]]:
    """Half the weight to each list (each normalized first)."""
    def normalized(guesses):
        total = sum(weight for weight, _ in guesses)
        return [(0.5 * weight / total, deck) for weight, deck in guesses] if total else []
    return normalized(first) + normalized(second)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decks", type=int, default=100, help="test decks to guess")
    parser.add_argument("--revealed", type=int, nargs="+", default=[0, 3, 6, 10])
    parser.add_argument("--chudledee-guesses", type=int, default=4, help="Chudledee decks per guess")
    parser.add_argument("--builder", type=Path, default=Path("models/chudledee-v3.4.pt"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    rng = random.Random(arguments.seed)
    every_card = Counter({card: 2 for card in buildable_action_cards()})
    test_decks = [Deck(characters=entry["characters"], cards=entry["cards"]) for entry in json.loads(TEST_SET.read_text())]
    test_decks = [deck for deck in test_decks if len(deck.cards) == DECK_ACTION_CARDS and set(deck.characters) <= buildable_characters()
                  and lineup_is_feasible(list(deck.characters), every_card)]
    test_decks = rng.sample(test_decks, min(arguments.decks, len(test_decks)))
    tournament = TournamentGuesser([entry.deck for entry in load_tournament_decks(downloaded_versions())])
    chudledee = ChudledeeGuesser(arguments.builder, arguments.seed, torch.device("cpu"))
    card_kinds = len(buildable_action_cards())

    cases = [(deck, revealed_count, Counter(rng.sample(deck.cards, revealed_count))) for deck in test_decks for revealed_count in arguments.revealed]
    started = time.perf_counter()
    lineup_only = chudledee.guesses([(tuple(deck.characters), Counter()) for deck in test_decks], arguments.chudledee_guesses)
    lineup_by_deck = {id(deck): guesses for deck, guesses in zip(test_decks, lineup_only)}
    conditioned = chudledee.guesses([(tuple(deck.characters), revealed) for deck, _, revealed in cases], arguments.chudledee_guesses)
    print(f"built {len(test_decks) + len(cases)} x {arguments.chudledee_guesses} Chudledee guesses in {time.perf_counter() - started:.0f}s")

    scores = {(name, count): {"overlap": [], "log_likelihood": []} for name in GUESSERS for count in arguments.revealed}
    for (deck, revealed_count, revealed), chudledee_guesses in zip(cases, conditioned):
        truth = Counter(deck.cards) - revealed
        hidden = sum(truth.values())
        tournament_guesses = tournament.guesses(tuple(deck.characters), revealed, excluded=tuple(sorted(deck.cards)))
        by_name = {"lineup": lineup_by_deck[id(deck)], "chudledee": chudledee_guesses, "tournament": tournament_guesses,
                   "mix": _mixed(chudledee_guesses, tournament_guesses)}
        for name, guesses in by_name.items():
            expected = expected_remaining(guesses, revealed, hidden)
            scores[name, revealed_count]["overlap"].append(overlap_share(expected, truth))
            scores[name, revealed_count]["log_likelihood"].append(mean_log_likelihood(expected, truth, card_kinds))

    report = {"decks": len(test_decks), "chudledee_guesses": arguments.chudledee_guesses, "seed": arguments.seed, "results": {}}
    print(f"{'revealed':>8} " + " ".join(f"{name:>22}" for name in GUESSERS))
    for count in arguments.revealed:
        row = []
        for name in GUESSERS:
            overlap = statistics.mean(scores[name, count]["overlap"])
            likelihood = statistics.mean(scores[name, count]["log_likelihood"])
            report["results"][f"{name}@{count}"] = {"overlap": overlap, "log_likelihood": likelihood}
            row.append(f"{100 * overlap:5.1f}% covered {likelihood:6.2f}")
        print(f"{count:>8} " + " ".join(f"{cell:>22}" for cell in row))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
