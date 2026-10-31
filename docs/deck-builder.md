# Chudledee, the deck builder

Chudledee builds a legal deck from a card collection: 3 characters, then 30 action cards, one pick at a time. It is a
separate network from Chudledum, the combat bot, and learns by having Chudledum play its decks. Code:
`autogitcg/deck_builder/` (the network and building), `autogitcg/deck_builder_training/` (training).

It is **benched**: the PvP bot (theta 1.13) plays real tournament decks, which suit it best, and Chudledee was not
retrained for it. It remains the way to build decks from a limited collection, as in the progression simulator.

## What it sees and picks

At each pick (`deck_builder/building_encoder.py`) it sees what it has picked so far, the opponent's deck when it is known
(an NPC stage, with any pre-equipped cards), and, while picking characters, the collection's action cards, so a lineup
can fit the cards the player owns. Only legal picks are offered (`deck_builder/building_state.py`): owned copies, the
copy limit, talents only for picked characters, tag requirements, and only lineups that can still reach 30 legal cards.
Every finished deck is legal.

Cards are identified with the combat network's vocabulary, and Chudledee starts with Chudledum's card, character and
skill embeddings copied in, so it begins knowing what the combat bot learned about each card.

## How it learns

1. **Warm start by imitation** (`scripts/deck_builder/pretrain_deck_builder.py`): it copies 60 hand-made synergy decks
   and the 748 legal NPC decks, each placed inside a larger simulated collection. A deck has no order, so each pick is
   scored against the whole set of cards still missing.
2. **Reinforcement learning** (`scripts/deck_builder/train_deck_builder.py`): each update builds 256 decks for sampled
   tasks, half NPC stages and half PvP. Each deck plays 4 games, 2 per seat, with a **frozen Chudledum** on both sides;
   the deck's reward is its mean game reward. PPO credits every pick with that reward minus the value estimate at the
   pick, so "this collection is weak" or "this stage is hard" doesn't count against the picks. PvP opponents mix
   synergy, NPC, tournament and random collection decks and Chudledee's own recent decks.

The yardstick (`deck_builder_training/builder_benchmark.py`) is 48 stage and 48 PvP tasks on held-out collections, with
fixed opponents and seeds, so Chudledee's decks and random legal decks play exactly the same games.

## Versions

The versions are small steps of the same training, so they keep simple numbers.

| Version | Trained against | What changed |
|---|---|---|
| v1 | alpha 2.40 | imitation warm start, then 300 updates of RL |
| v2 | beta 10.8 | from v1; real tournament decks among the PvP opponents |
| v3 | beta 10.8 | v2 plus 200 updates without the entropy bonus: stronger favourites, more 2-copy cards |

All three, with every 50-update checkpoint, are in the model releases ([models.md](models.md)).

## How well it works

- **Through the whole progression** (simulated accounts from the starter deck, bots on both sides; Chudledee v1 with
  alpha 2.40 playing): with Chudledee's decks the accounts won 88% of their matches and all 96 collected the full character roster (median 694 matches);
  with random legal decks they won 43% and none did in 2,500 matches.
- **Every NPC stage** with a deck built from every obtainable card (558 stage setups, 16 games each; Chudledee v3 with
  beta 10.8 playing): 93.7% won. The losses are mostly round-limit stalls against healer stages and fast losses to serious
  invitations with a pre-equipped talent.

## Limits

- **It barely adapts to the opponent.** With the full card pool, all 558 stage decks used one of three lineups (Yae Miko
  and Goldflame Qucusaur Tyrant with Nahida, Kokomi or Golden Wolflord) and only 96 distinct card lists.
- **It builds for its judge.** The reward is how well the frozen Chudledum plays a deck, so it favours decks that
  network handles well, not decks people would build. Imitating real tournament decks first
  (`scripts/deck_builder/pretrain_builder_on_human_decks.py`) made its decks look more human, but RL drifted back to the
  judge's favourites.
- **It plays no part in PvP.** The combat bot's opponents in PvP are guessed from published decks, never from
  Chudledee ([search.md](search.md)).
