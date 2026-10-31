# Training Chudledum

How the combat network is trained, the settings that produced the final networks, and what every training attempt
tried and learned. The network itself is described in [architecture.md](architecture.md); the deck builder's training
in [deck-builder.md](deck-builder.md); where it ran in [cloud-training.md](cloud-training.md).

## How it works

`scripts/training/train.py` runs **PPO** (proximal policy optimization) with self-play:

- **Actors** (`training/actor_process.py`, spawned processes) play thousands of games at once, each with its own engine
  workers, encoder and copy of the network. Every decision is stored with what was chosen and how likely it was. Actors
  keep playing during the learner's update; the staleness is measured (`policy_lag` in the metrics).
- **The learner** (`training/ppo_update.py`) updates the network on 32,768 steps per update: 3 epochs of minibatches of
  512, clipped policy loss, value loss, the auxiliary outputs, gradient clipping and an early stop on KL divergence.
- **The reward** (`training/game_result.py`): a win is worth `1 - 0.02 x rounds`, so a faster win is slightly better;
  every loss is -1, including reaching the round limit (a loss for both, as in the game). The speed term is a setting
  and is off for PvP (some PvP decks stall on purpose). Advantages use GAE.
- **Opponents** (`training/opponent_league.py`): the current network itself (both seats learn, 55%), past snapshots
  (30%), a frozen **reference** network in mirror pairs (8%), a scripted heuristic bot (5%) and random play (2%).
- **Decks** (`training/deck_pools.py`, `training/matchup_sampler.py`): NPC stages (the opponent's list is open, as in
  the game), real tournament decks, decks from the deck builder, hand-made synergy decks (`data/decks/`, suggested by a
  language model and checked by the engine) and decks from simulated collections. Seats are swapped at random, and the
  opponent's list is withheld at random, so one network learns both open-list and blind play. Every deck carries its
  origin, so results can be split by source.
- **Difficulty** (`training/adaptive_difficulty.py`): the reference plays mirror pairs (same seed, decks swapped), and
  the pair's mean score measures skill per matchup category. Training weight shifts toward the categories where the
  network is weakest.
- **Held-out tests** (`data/benchmarks/`): about 20% of tournament lineups are locked away and never reach training
  (`scripts/data/lock_tournament_test_set.py`; `tests/test_training.py` checks it). The **meta test**
  (`scripts/data/lock_meta_test.py`) is the opposite: blind matchups between the most played current decks, the ones a
  PvP network trains on, to measure play in the current meta.
- **Run folders** (`training/run_storage.py`): metrics, checkpoints, the weights every 5 updates and every game as
  append-only compressed JSON lines. A run resumes from its latest
  checkpoint with the same command.

### Fine-tuning a trained network

From scratch, the standard settings work (entropy bonus 0.01, GAE lambda 0.95). Fine-tuning a network that already
plays well does not improve with them: past the beginner stage, the per-update policy-gradient signal is mostly noise
from dice and draws and cancels out, while the entropy bonus pushes the same way every update and wins. The fix, found
in the beta diagnosis runs: **entropy 0, lambda 0.8, learning rate 5e-5**.

### The final PvP run (theta 1)

```sh
python scripts/training/train.py --run runs/theta_01 --updates 1000 --seed 7 \
    --initialize-from models/beta-10.8.pt --reference models/beta-10.8.pt \
    --replies --preview-scale 10 --reply-checks 4 \
    --deck-sources tournament=0.9,chudledee=0.07,stage=0.03 \
    --human-deck-opponents tournament=0.9,chudledee=0.1 --tournament-from 6-4 \
    --speed-penalty 0 --entropy-coefficient 0 --gae-lambda 0.8 --learning-rate 5e-5 \
    --evaluate-every 0 --checkpoint-every 5 --actors 10 --workers-per-actor 11
```

- It starts from beta 10.8 with every new input (previews, replies, follow-ups, end of round, entity facts) at zero
  weight, so update 0 plays exactly like beta 10.8.
- Decks: 90% real tournament decks from version 6.4 on (older lists mostly hold cards changed since), against the same
  mix.
- Every 50 updates a separate machine played 2,000 blind games against plain beta 10.8 on the meta test; the run stopped
  itself after two tests in a row without a new best. Best: update 650, **theta 1.13**, 69.4%.

The NPC bot is **beta 10.8**, trained with the NPC defaults (speed reward on, about half PvE decks).

## Every attempt, by family

Attempts are grouped into Greek-letter families by their main idea. A network is named
`<family> <run>.<update / 50>`: theta 1 at update 650 is theta 1.13. All of them can be downloaded
([models.md](models.md)). Win rates are blind games against the reference of the time.

### Alpha: PPO from scratch

| Run | Idea | Result |
|---|---|---|
| alpha 0 | does the pipeline learn at all | yes |
| alpha 1 | first real run | beat the scripted bot 75%; played few cards and tuned a lot |
| alpha 2 | small identity embeddings, the speed reward, a win-chance output | 95% vs the scripted bot: **alpha 2.40**, the first strong network |
| alpha 3 | alpha 2 trained longer | no gain |

### Beta: PPO from alpha 2.40

| Run | Idea | Result |
|---|---|---|
| beta 1 | tournament and deck-builder decks | stuck at alpha 2.40's level |
| beta 2 | a bigger network | below alpha 2.40 |
| beta 3-9 | diagnosis: why training stalls | the entropy bonus and lambda 0.95 block learning; the fix is entropy 0, lambda 0.8 |
| beta 10 | alpha 2.40 with the fix | about +65 Elo: **beta 10.8**, the NPC bot and the reference for everything after |

### Gamma: settings and size, from beta 10.8

| Run | Idea | Result |
|---|---|---|
| gamma 1-3 | entropy and learning-rate tweaks | flat at about 50% vs beta 10.8 |
| gamma 4 | a bigger network, kickstarted from beta 10.8 | 35-40% |
| gamma 5 | a bigger network that copied beta 10.8 first | copied it, then no gain |
| gamma 6 | review fixes and trying rare moves | a bug drifted it to 44% |
| gamma 7 | gamma 6 with the bug fixed | +1.5 points, then stalled |
| gamma 8 | no speed reward | stalled |
| gamma 9 | reward for HP dealt and kept | no change |

### Delta: decks and card play, from beta 10.8

People who played the bot noticed it plays few cards (about 7 a game against their 19). Delta tried to change that.

| Run | Idea | Result |
|---|---|---|
| delta 1 | mostly tournament decks | no better with human decks |
| delta 2 | human decks only | the same stall |
| delta 3-6 | rewards for playing cards (any card, then events and resonances) | up to 16 cards a game, no more wins |
| delta 7-8 | fading the card rewards out | the habit faded back to about 9 cards |
| delta 9 | no speed reward after the card rewards | card play still faded |

Against bots, how many cards the network plays didn't change its results. See also
[experiments/card_play_diagnostic](../experiments/card_play_diagnostic).

### Epsilon and zeta: students copying the search

The look-ahead plays clearly better than the network alone, so these runs trained networks to copy its choices
(epsilon: the look-ahead's labels; zeta: the deep judge's). No student beat the network the labels came from: most
labels are close calls between moves of nearly equal value, and the students learned noise. Details in
[experiments/lookahead_students](../experiments/lookahead_students).

### Eta: new inputs through PPO, from beta 10.8

| Run | Idea | Result |
|---|---|---|
| eta 1 | action previews as input | the new weights barely grew; 45-51% |
| eta 2-3 | predicting the board a few decisions ahead (auxiliary output) | predicted well; 47.5% |
| eta 4 | the opponent's replies to each move | 52.6% |
| eta 5 | card replies, danger after fast actions | 51.8% |
| eta 6 | the learned reply selector | 53.5% |
| eta 7 | eta 6 continued | 57.9% on fresh seeds (best: **eta 7.3**) |
| eta 8 | a PvP fine-tune of eta 7 | stopped early: a fine-tune can't undo the wrong settings |

Eta 4-8 trained with the NPC settings (speed reward on, half PvE decks) by mistake, which is why theta retrained from
beta 10.8 instead of continuing from them.

### Theta: the PvP retrain

| Run | Idea | Result |
|---|---|---|
| theta 1 | from beta 10.8 with the full input set (previews, follow-ups, full-turn replies, end of round, entity facts), PvP settings and current tournament decks | **theta 1.13**: 69.4% vs beta 10.8 on the meta test; the PvP bot |

## Lessons

- **Beating the reference bot is a narrow measure.** Card-play habits, play style and wins against people barely
  showed up in bot-vs-bot win rates.
- **Check every flag against what was decided before launching.** Five eta runs trained with the wrong reward and
  decks because the script defaults are the NPC setup.
- **Test on what the bot will face.** The locked tournament test is mostly old decks; a PvP network trained on current
  decks looked flat there (53 -> 52%) while the meta test showed its real progress (56 -> 60%).
- **New inputs grow slowly from zero.** Inputs added at zero weight barely grew at a fine-tuning learning rate (eta 1);
  theta 1 multiplied them by 10 (`--preview-scale 10`) so their weights could grow about ten times faster.
