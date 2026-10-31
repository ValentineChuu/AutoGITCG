# Results

All games are blind (neither side sees the other's list) unless noted. "± x" is one standard error. The networks are
in the model releases ([models.md](models.md)).

## The PvP bot: theta 1.13

**Meta test** (`data/benchmarks/meta_benchmark_v1.json`): the most played list of each of the 40 most played
tournament lineups from version 6.4 on, paired into 100 matchups, each played in both deck orders and both seats.
These are the decks a PvP network trains on, so this measures play in the current meta.

| Theta 1.13 | vs beta 10.8 | vs alpha 2.40 | vs eta 7.3 |
|---|---|---|---|
| no search | **69.4%** (2,000 games) | 73.5% | 63.7% |

Against beta 10.8, alpha 2.40 itself wins 41.8% (beta 10.8 wins 58.2%) on the same test.

**With a judge** (look-ahead at play time, [search.md](search.md)), 400 games each unless noted:

| Theta 1.13's judge | vs plain beta 10.8 | vs beta 10.8 with the same judge | vs eta 7.3 with the same judge |
|---|---|---|---|
| none | 69.4% | – | 63.7% |
| classic | 75.6% (320 games) | | |
| quick | 78.8% (160) | 70.5% | 65.9% (320) |
| cautious | 83.5% | 71.5% | 63.7% |
| wide | 81.5% | 66.0% | 58.8% |
| balanced | 84.8% | 66.8% | 60.2% |
| deep | **88.5%** | 68.3% (240) | 70.0% (40) |

With the quick judge, theta 1.13 also beat the earlier playtest bot (beta 10.8 with the classic judge, guessing the
opponent's cards with Chudledee) 78.3% over 240 games on the same decks.

**Over training** (theta 1, plain, vs beta 10.8, 2,000 games per point):

| Update | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500 | 550 | 600 | **650** | 700 | 750 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| meta test | 55.7 | 60.1 | 62.7 | 63.8 | 65.5 | 65.7 | 65.8 | 66.6 | 65.1 | 67.5 | 66.2 | 68.3 | **69.4** | 68.5 | 69.3 |
| locked tournament test | 53.2 | 52.2 | 53.9 | 53.6 | 54.0 | 54.2 | 52.7 | 54.4 | 53.9 | 55.5 | 54.5 | 56.4 | **56.9** | 57.4 | 56.6 |

The **locked tournament test** (`data/benchmarks/tournament_test_set_v1.json`) holds lineups that never reached
training, 87% of them from before version 6.4. Theta trains on current decks only, so it gains less there, but it
still improves: playing current decks well didn't cost it the old ones.

## Against people

An earlier bot (beta 10.8 with the classic judge, decks built by Chudledee from a card pool) was played by a handful of
experienced players on a private playtest site: the people won 9 of the 10 games that were played to the end. Their
main complaints were the bot's decks and how few action cards it played. Theta 1.13 and tournament decks are the
answer to both; it has not yet been measured against people.

## NPC fights: beta 10.8

- **Every NPC stage** (558 setups of the in-game NPC opponents, 16 games each) with a deck Chudledee built from every
  obtainable card: **93.7%** won ([deck-builder.md](deck-builder.md)).

## The whole progression: alpha 2.40 with Chudledee v1

Simulated accounts from the starter deck, playing every stage they unlock, with Chudledee v1 building the decks and
alpha 2.40 playing them: 88% of matches won and the full character roster collected in a median of 694 matches,
against 43% and never with random legal decks.

## Supporting measurements

- **Deck guessing** (hidden cards of a real tournament deck covered by the guess): 59.6% with no card revealed, 63.3%
  after three, from published tournament decks; 10-12% when the deck builder guesses ([search.md](search.md)).
- **C++ engine**: every recorded game plays identically to the Node engine (thousands of games, every NPC stage, and
  3,404 of the 3,406 pieces of card code exercised), about 22 times faster on the same random games, and 35-45 times more training decisions per
  second (`engine/`).
