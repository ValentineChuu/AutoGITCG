# Tree search for Chudledum (replacing the one-step look-ahead)

## Why

The look-ahead (`docs/search.md`) only varies the first move: every copy then follows the
network's habit, one greedy line per guessed deal, and is scored by the network's win chance. The judge test
(2026-10-19) showed the win chance is what breaks it: with copies played to the end,
"try every move kind" went from 55.0% to 64.0% against plain beta 10.8. And a setup (resonance, then support, then the
skill) is a chain of moves, which a first-move-only search can't find.

## Design: information-set Monte Carlo tree search with played-out leaves

The method strong hidden-card game bots use (bridge, Skat, Hearthstone/Magic research), with AlphaZero's way of
using the network to choose where to look:

- **One walk = one game copy** with the hidden cards dealt again (opponent's hand and deck from a Chudledee guess
  against a blind opponent, our own deck reshuffled, dice reseeded). Walks of one search share one tree, so the tree
  averages over deals instead of trusting one.
- **Tree nodes** are positions reached by a sequence of moves from the searched decision, whatever was dealt.
  Moves are keyed by what they are (kind, card or skill, targets: `search/move_options.py`), so the same move is
  the same edge in every copy, and each copy only picks among its own legal moves.
- **Choosing in the tree (PUCT):** a move's mean result so far plus `exploration x network probability x
  sqrt(visits here) / (1 + visits of the move)`. A move never tried is valued at the mean of the tried ones.
  Walks running at the same time count as losses until their results arrive (virtual loss), so they spread out.
- **Leaving the tree:** a walk adds one new node, then the network (greedy, both players) plays the copy out to
  the end of the game. The win or loss goes back up every edge it took, from each node's player's side. No win
  estimate is needed (option: `leaf_horizon_rounds` scores at a later round with the win chance instead).
- **The answer:** the root's most visited move.
- **Scope:** `branch_opponent=False` branches only over our own moves until the opponent acts (planning our turn);
  `True` branches over the opponent's replies too. Rerolls, card selections and switches after a defeat are always
  played by the network.

Code: `search/tree_search_settings.py`, `search/search_tree.py` (nodes, PUCT, backing up),
`search/tree_search.py` (one search and its walks), `search/tree_search_table.py` (real games and copies in one
engine pool), `search/tree_search_match.py`, `experiments/tree_search/play_tree_search.py`.

## Ground-truth set (`experiments/ground_truth/`)

Plain beta 10.8 games (blind, test matchups); in each game one unsure decision of one seat is held, and its compared moves
(beta 10.8's 6 most likely, plus its most likely move of every other kind) are each played out on the same guessed deals,
64 deals per batch, until the best move leads every other by 3 paired standard errors, every move has had identical
results so far, or 512 deals. The answer is "best if beta 10.8 plays the rest", the standard target of one step of search.
Each position stores the game's seed and responses (replayable) and a pickle of its decision and tracker, so any
network, judge or search setting can be scored on it: how often it picks the best move, and how much win chance
its pick loses against the best.

## Steps

Done (2026-10-20):

1. **Tree search, our turn only** (64 walks): 62.0% vs beta 10.8. **Both players:** 62.75%. No better than the look-ahead
   (62.3%; 63.7-65.2% in its best settings): the search changed beta 10.8's move in 41% of searched decisions, in no
   consistent direction (noise).
2. **Ground-truth set** (357 positions, `runs/ground_truth/`, `experiments/ground_truth/analyze_ground_truth.py`): beta 10.8's top move is
   the best in 20% of unsure decisions; held-out regret 9.7 win-chance points (random compared move 12.0; 20.0 on the
   98 separated positions); the best move's median probability under beta 10.8 is 1.8%; switching is underused (best in 41
   of 98 separated positions, beta 10.8's pick in 17). Separating moves took a median of 512 deals x 7 moves, so played-out
   search at affordable sizes can't see the differences: search needs a judge that ranks moves with low noise.

3. **Existing methods scored on the ground truth** (2026-10-21, `experiments/ground_truth/results/ground_truth_methods.json`): beta 10.8's win
   output is inconsistent between the players, so look-ahead must score by value from its own view. With that, 8 moves
   x 16 deals x 24 decisions deep lose 1.8 points per hard decision (beta 10.8 9.7, playtest look-ahead 6.9, fixed 5.1, tree
   search 7.8). A trained judge (old step 4) is not needed to get a good teacher: this deep judge is one.

4. **Deep judge as teacher** (2026-10-22/23): 151k labels in `runs/teacher_deep_01` (100k searched; the searcher won
   77% vs beta 10.8). Students from beta 10.8 on them: tries 1-2 (rules best/improved) 51.6% / 49.1% vs beta 10.8, ground-truth regret
   unchanged; try 3 (only clear wins) 21.6%: it learned to avoid beta 10.8's move everywhere. Cause: about 87% of best-move
   labels are near-ties within the search noise.
5. **More ground truth:** 2,148 positions in `runs/ground_truth_more` (2,505 in all; beta 10.8 regret 8.18).

Next (2026-10-23):

6. **Try 4, soft targets** (label rule `soft`, `soft_targets.py`): beta 10.8's own probabilities shifted toward each move by
   its advantage shrunk by its noise (temperatures 0.02 and 0.01). Bar: about 53% or more vs beta 10.8 over 2,000 games and
   lower ground-truth regret; if met, continue the loop (new labels with the student, next student).
7. **If try 4 fails: the deep judge at play time** (it already wins 77% vs beta 10.8): measure its time per decision and make
   it fast enough for PvP play.
