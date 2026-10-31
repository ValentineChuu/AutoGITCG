# AutoGITCG training data: CSV bundle

This is data from training a reinforcement-learning bot for **Genius Invokation TCG (GITCG)**, the card game inside Genshin Impact.

## The game
- Each player has 3 characters and a 30-card deck. They roll elemental dice every round.
- Players spend dice on character skills: normal attack, elemental skill and elemental burst. A burst needs full energy.
- Players also spend dice on cards. Card types:
  - **talents:** equip a specific character;
  - **weapons and artifacts:** equipment;
  - **supports:** stay on the field;
  - **events and food:** one-off effects.
- **Elemental tuning** discards a card to convert one die to the active character's element.
- A player wins by defeating all opposing characters. Reaching round 15 ends the game as a **loss for both players**.

## The training
The bot (the "learner", a neural network) was trained with PPO, a standard reinforcement-learning algorithm. Reward: +1 for a win, −1 for a loss.

**Opponent mix:**
| Opponent | Share of games | Description |
|---|---|---|
| `self` | 55% | Both seats are the learner; self-play |
| `snapshot` | 20% | A past version of the learner |
| `heuristic` | 20% | A scripted bot. Priorities: escape a dying active character, play cards while keeping 3 dice, burst > elemental skill > normal attack, tune when nothing is affordable, reroll off-element dice |
| `random` | 5% | Random legal moves |

**Deck mix (`source`):**
| Source | Share of games | Description |
|---|---|---|
| `stage` | 50% | A random legal deck from a simulated player's collection vs a real PvE NPC stage deck (some NPCs have talents pre-equipped) |
| `npc_decks` | 25% | Two NPC 3+30 decks |
| `collections` | 25% | Two random collection decks |

Random collection decks are often incoherent; the deck builder that will fix this is not trained yet.

**Deck knowledge:** a seat is told the opponent's deck list (NPC mode) or not (blind mode), chosen at random per seat.

**Timeline:** 600 PPO updates, about 32k learner decisions each. The learning rate was constant 2e-4 until update 370, then decayed slowly.

## Files and columns
**`games.csv`:** one row per game (about 280k).
- `game_key` (`"update:index"`, joins every file), `update_number` (training update when the game finished), `policy_version`.
- `source`, `opponent`, `seat0_controller` / `seat1_controller` (`learner` / `heuristic` / `random` / `snapshot`).
- `seat0_knows_opponent_deck` / `seat1_knows_opponent_deck` (1 = NPC mode).
- `seed`, `winner` (0, 1, or empty = round-limit loss for both), `rounds`, `seat0_defeated` / `seat1_defeated` (characters lost), `move_count`.

**`game_characters.csv`:** `game_key, seat, slot, character_id`.

**`game_cards.csv`:** `game_key, seat, card_id, copies`.

**`game_equipment.csv`:** cards equipped before the game starts: `game_key, seat, character_slot, definition_id`.

**`definitions.csv`:** `id, kind (character/card/skill), name`. English names for every id.

**`seat_summary.csv`:** one row per game and seat. A summary of every move that seat made:
| Column | Meaning |
|---|---|
| `moves`, `action_decisions` | All decisions; main-phase decisions only |
| `skills_used` | All skill uses |
| `normal_attacks`, `elemental_skills`, `bursts` | Skill uses by type (card-granted skills count only in `skills_used`) |
| `cards_played`, `cards_tuned` | Cards played; cards discarded by elemental tuning |
| `switches` | Voluntary character switches |
| `forced_active_choices` | Picking a new active character after one is defeated, plus the initial pick |
| `dice_rerolled`, `opening_cards_swapped` | Mulligan and reroll totals |
| `mean_chosen_probability`, `mean_entropy` | How confident the network was. Network seats (learner/snapshot) only; empty for scripted seats |
| `first_value`, `last_value` | The network's own win estimate at its first and last decision, in [−1, 1]: −1 = sure loss, +1 = sure win, win chance = (value + 1) / 2 |
| `first_round`, `last_round` | First and last round with a move |

**`moves_sample.csv`:** every move of 5,000 random games, in order.
- `game_key, move_number, seat, round`.
- `decision_kind`: `action`, `chooseActive`, `rerollDice`, `switchHands` (opening mulligan) or `selectCard`.
- `action_kind`: `useSkill`, `playCard`, `switchActive`, `elementalTuning` or `declareEnd`.
- `definition_id`: the skill, card or character used. For rerolls and mulligans, the number of dice or cards changed instead.
- `is_fast`: fast actions don't end the turn.
- `chosen_probability, entropy, value, option_count`: network seats only. `option_count` is the number of legal options.

**`training_metrics.csv`:** one row per training update.
- Learning rate, policy entropy, update size (`approx_kl`, `clip_fraction`).
- Value-head quality (`value_loss`, `explained_variance`).
- `win_rate_vs_*`: win rates in training games.
- `benchmark_vs_*`: fixed 200-game benchmark every 10 updates.

## Notes
- **Seats:** "the learner" can sit in seat 0 or 1. In `self` games both seats are the learner.
- **Learner seat:** to find the learner seat in a game, use `seat0_controller` / `seat1_controller`.
- **Stage side:** in `stage` games, the side holding the NPC stage deck isn't marked. A random collection deck has 30 cards; many stage decks are 3 characters with an NPC card list.
- **Not recorded:** damage dealt and dice actually paid.
