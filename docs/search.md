# Search at play time: look-ahead, judges and deck guessing

The network alone picks a move in one forward pass. At play time it can also **search**: try its few most likely
moves in copies of the game and keep the one that turns out best. Search trains nothing; it makes any network play
stronger at the cost of time. Code: `autogitcg/search/`.

## Look-ahead

At an action decision (`search/lookahead_table.py`, `search/decision_search.py`):

1. If the network's top move is at least 90% likely, it is played without search.
2. Otherwise the **candidates** are the network's most likely moves (and, with `every_move_kind`, its most likely move
   of each kind not yet tried: a card, a skill, a switch, a tuning, ending the round, so that a plan it rates low still
   gets tested).
3. For each candidate the engine makes **copies** of the game at this decision, each with the opponent's hidden cards
   dealt differently, plays the candidate, and lets the network play both sides on (greedily) for a fixed number of
   decisions (the **horizon**).
4. Each copy is scored with the network's **value** at the end, always from the searching player's own view (the copy
   plays on to that player's next decision). The candidate with the best mean score replaces the network's own choice
   if it is better by a margin.

The searching player never sees the opponent's real hidden cards: copies deal them from a **guessed deck** (below).

Scoring by value matters: the network's separate win-chance output is not consistent between the two players (at one
position their two win chances summed to 1.33 on average, anywhere from 0.2 to 2.0), while the values of the two sides
sum to about zero, as they should.

## Judges

Named settings of the look-ahead (`search/judges.py`). "Moves x deals" is candidates times copies; all use a horizon
of 24 decisions. All but classic score by value from the searcher's own view.

| Judge | Moves | Deals | Beta 10.8 + judge vs plain beta 10.8 | Theta 1.13 + judge vs plain beta 10.8 |
|---|---|---|---|---|
| none | – | – | 50% | 69.4% |
| classic | 4 | 4 (win output, either view) | 59.8% | 75.6% |
| quick | 4 | 4 | 68.8% | 78.8% |
| wide | 8 + every move kind | 4 | 71.8% | 81.5% |
| cautious | 4 + every move kind | 8 | 73.5% | 83.5% |
| balanced | 8 + every move kind | 8 | – | 84.8% |
| deep | 8 + every move kind | 16 | 77.8% | **88.5%** |

Theta's games are on the meta test (current tournament decks, 400 games per row unless noted in
[results.md](results.md)); beta's are 400 blind games each on the locked tournament test, measured before the
tournament deck guess existed (the opponent's cards were guessed by Chudledee then). Quick costs about the same as classic and is
clearly better, so it is the default for interactive play. Deep is the strongest and the slowest: about 13 seconds
per searched decision on 8 desktop cores.

Theta's lead holds when both sides search with the same judge (66-72% against beta 10.8 with the same judge); search
helps most against a player that doesn't search.

## Guessing the opponent's deck

In a blind game the searching player knows the opponent's characters and the cards they have revealed, not their
list. People mostly play published community decks, so the guess comes from **real tournament decks**
(`deck_guessing/`): the published lists with the same lineup and the most overlap with what the opponent has
revealed, weighted by how much they were played (`--guess-source tournament`, the default).

Offline test on 100 locked test decks (`scripts/evaluation/evaluate_deck_guessing.py`,
`docs/results/deck_guessing_v1.json`), with the test decks' own lists left out of the database: share of the hidden
cards covered by the guess.

| Cards revealed | Guessed by the deck builder (Chudledee) for the lineup | Guessed from tournament decks |
|---|---|---|
| 0 | 12.0% | 59.6% |
| 3 | 10.3% | 63.3% |

The tournament guess is about six times better, so anything about the opponent (search copies, the replies input,
search labels) uses it. Chudledee builds the bot's own decks only. NPC opponents' decks are public in the game, so
NPC fights use the real list.

## What else was tried

- **Tree search** (copies branching at every decision, not only the first move) was no better than the look-ahead:
  62% against plain beta 10.8 with 64 walks per decision. See [experiments/tree_search](../experiments/tree_search).
- **Training a network to copy the search** (students of the look-ahead and of the deep judge) never beat the network
  the search started from. See [experiments/lookahead_students](../experiments/lookahead_students).
