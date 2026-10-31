# Architecture

How a game is played, what the combat network sees, and what it outputs. Training is in
[training.md](training.md), search at play time in [search.md](search.md), the deck builder in
[deck-builder.md](deck-builder.md).

## The engine and the bridge

The game engine is its own repository, included at `engine/`: a C++ engine that plays exactly the games of
[Piovium](https://github.com/piovium/genius-invokation)'s TypeScript engine (`@gi-tcg/core` with the card data in
`@gi-tcg/data`, version 0.21.0, AGPL-3.0), plus custom definitions for NPC-only characters, 20-40 times faster.
Its JavaScript reference (`engine/reference/`: Piovium's engine with a small patch set, the same worker, the custom
characters) is what the C++ engine is checked against, game for game; set `AUTOGITCG_ENGINE=node` to play with it
instead.

Python never runs the rules itself. `autogitcg/bridge/EnginePool` starts worker processes
(`engine/reference/bridge/worker.mjs` or the C++ worker) and talks to them in newline-delimited JSON over stdio; each worker
runs many games at once. A game asks its players for decisions; each decision message carries:

- the request: the legal options (actions, dice to reroll, cards to swap, characters to choose), each action with
  its cost and the engine's own legal dice payment;
- the deciding player's own view of the state (`exposeState`): the opponent's hand, dice and deck order hidden;
- optional extra inputs the network can read: action **previews**, the opponent's **replies**, **follow-ups** and
  the **end of round** (below).

Python answers with the choice. Look-ahead search also asks the engine for copies of a game at a decision, with the
opponent's hidden cards re-dealt ([search.md](search.md)).

Games are reproducible: the engine is seeded, and decks are shuffled from the seed by the bridge (the engine's own
shuffle ignores the seed). Answers to decisions asked of both players at once (rerolls, opening hands) must be given
in the same order to replay a game.

## What the network sees

`autogitcg/observation/ObservationEncoder` turns a decision into fixed-shape arrays. Every input is something a
player could see on screen or remember from the game so far; nothing the engine hides is added back.

The board is a set of **tokens**, one per object:

| Group | Tokens | Live features (examples) |
|---|---|---|
| global | 1 | round, phase, whose turn, who ended the round first, dice per type, hand and deck sizes, Arcane Legend used, the end-of-round prediction |
| character | up to 4 per side | slot, active, HP, energy, aura, defeated |
| entity | statuses, combat statuses, summons, supports, equipment | owner, usage counter and its meaning, per-round usage |
| hand | my cards, the opponent's card backs | current cost after discounts, playable, tunable, markers |
| dice | my dice | type, matches an active character's element |
| knowledge | per card definition | still in my deck, seen played by the opponent, in the opponent's list (when known) |
| history | the last public events | card played, skill used, switch, tune, round end |
| action | one per legal option | kind, card or skill, cost, fast or not, targets, preview, replies, follow-ups |

Each token is the sum of three parts: its live features (projected), a **definition vector** for what it is (a learned
embedding of its card, character, skill or status, plus projected static facts from the catalog: cost, element, tags,
HP, triggers, what a summon or status does), and an embedding of its group. An action token also gets the vectors of
the board tokens it targets.

All tokens go through one transformer (4 layers, width 128, 4 attention heads, about 1.3 million weights for the
final network), so every token can attend to every other: this is where "this status protects that character" is
learned.

**Catalog facts** (`observation/catalog_features.py`) come from Piovium's card catalog and from reading the card
code (`scripts/data/export_entity_effects.py`: damage, element and healing of every summon and status). They let rare
and new cards start out similar to cards that behave alike.

### Inputs added later, and how networks survive them

Features added after a network was trained are appended at the end of their group, and a saved network loads with
zero weights for them (`model/network_config.grown_feature_widths`, `model/grown_catalog.py`), so it plays exactly as
before and can learn to use them. The later inputs:

- **Previews**: each legal action's immediate result (the engine's own preview: damage, HP, energy, entities).
- **Replies** (`engine/reference/bridge/opponent_replies.mjs`, `observation/encode_replies.py`): for a move, the opponent's
  whole turn after it, as the worst outcome for me over their possible lines. Their hidden cards are guessed from
  published tournament decks, never their real hand ([search.md](search.md)). After a fast action, also what they
  could do if the turn ended there.
- **Follow-ups** (`engine/reference/bridge/follow_ups.mjs`): for a move, the best outcome reachable with the rest of my turn.
- **End of round** (`engine/reference/bridge/end_of_round.mjs`): what the end phase would do if both players ended the round
  now.

Replies and follow-ups come from searching every distinct line of a player's turn (`engine/reference/bridge/turn_search.mjs`),
with line rules that cut useless moves (invalid or effectless moves, healing food on full HP, duplicate copies, one
tuning per die; draw and random cards end a line). They cost far more than a plain decision, so a learned **reply
selector** picks the moves worth checking: the network first scores the decision without replies and outputs a
per-move "reply interest"; the engine computes replies and follow-ups for the 4 most interesting moves; the network
scores again with them. Moves not checked carry a "not checked" flag. On 15% of decisions every move is checked,
which gives the selector its training targets: how much each move's score changed once its replies were known
(`policies/reply_selection.py`).

## What the network outputs

| Output | Used for |
|---|---|
| action scores | the move: one score per action token (also choosing the active character and picking a card) |
| per-die scores | reroll: an independent keep-or-reroll per die |
| per-card scores | opening-hand swap: an independent keep-or-swap per card |
| value | the expected score of the game from here (win `1 - 0.02 x rounds`, loss -1); the critic in training and the judge's score in search |
| win chance | a separate plain win probability; unreliable for search (the two players' estimates at one position don't add up to 1) |
| reply interest | which moves to check for replies |
| future board, move values | auxiliary training outputs (predicting the board a few decisions ahead; per-move values from search labels) |

Dice are paid with the engine's own legal payment for the chosen action (`autoSelectedDice`); the network does not
choose which dice to pay.

## The progression simulator

`autogitcg/progression/` simulates an account from the post-tutorial starter deck: the NPC stages it can play
(invitations, weekly guests, taverns, open world, built from the game's tables), their rewards, and the shop, so card
collections grow the way a real account's would. It is a training environment only: it produces realistic
collections for deck-builder training and lets the bots play the whole progression
(`scripts/progression/run_progression_playthrough.py`). Its rules are from the game data and the community wiki
(`autogitcg/progression/economy.py` lists the wiki pages).
