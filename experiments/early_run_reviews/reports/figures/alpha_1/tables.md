# Generated tables for alpha 1

### Behaviour over training (means per learner seat)

n = learner seats (a self-play game contributes two). 95% intervals are under ±0.02 for every per-round mean at these sample sizes.

| measure | learner, updates 0-24 (n=11,202) | learner, updates 100-124 (n=17,651) | learner, updates 300-324 (n=18,909) | learner, updates 575-599 (n=20,561) | heuristic, same late games (n=2,685) |
|---|---|---|---|---|---|
| normal attacks / round | 0.74 | 0.81 | 0.70 | 0.66 | 0.68 |
| elemental skills / round | 0.32 | 0.69 | 0.83 | 0.84 | 0.58 |
| bursts / round | 0.09 | 0.26 | 0.30 | 0.30 | 0.20 |
| burst share of skills | 0.07 | 0.15 | 0.16 | 0.16 | 0.13 |
| cards played / round | 1.07 | 0.76 | 0.83 | 0.97 | 2.39 |
| talents played / game | 0.13 | 0.04 | 0.03 | 0.04 | 0.28 |
| tunings / round | 1.22 | 1.68 | 1.66 | 1.55 | 0.31 |
| voluntary switches / round | 1.27 | 0.68 | 0.46 | 0.50 | 0.24 |
| actions / round | 5.65 | 5.79 | 5.67 | 5.72 | 5.29 |
| share of rounds it ended first | 0.45 | 0.40 | 0.41 | 0.41 | 0.68 |
| nominal dice spent / round | 6.19 | 6.94 | 6.95 | 7.12 | 7.93 |
| dice rerolled / reroll | 4.76 | 4.40 | 4.39 | 4.57 | 4.49 |
| mulligan cards changed | 2.46 | 2.36 | 2.27 | 2.28 | 0.00 |

### Policy confidence over training

Per-seat means of per-decision values. Reroll and mulligan probabilities are of the whole keep/change pattern, and their entropies are summed over dice or cards.

| quantity | updates 0-24 | updates 100-124 | updates 300-324 | updates 575-599 |
|---|---|---|---|---|
| action: mean probability | 0.246 | 0.450 | 0.480 | 0.497 |
| action: mean entropy | 1.756 | 1.175 | 1.099 | 1.072 |
| chooseActive: mean probability | 0.532 | 0.675 | 0.772 | 0.807 |
| chooseActive: mean entropy | 0.737 | 0.532 | 0.376 | 0.318 |
| rerollDice: mean probability | 0.007 | 0.017 | 0.046 | 0.070 |
| rerollDice: mean entropy | 5.338 | 4.832 | 4.075 | 3.801 |
| switchHands: mean probability | 0.069 | 0.079 | 0.090 | 0.096 |
| switchHands: mean entropy | 3.313 | 3.166 | 3.043 | 2.978 |
| action: entropy / log(options) | 0.967 | 0.686 | 0.623 | 0.592 |

### Wins vs losses (updates 400-600)

Self-play: 54,047 decided games (paired: winner minus loser of the same game). Vs heuristic: 15,349 learner wins and 4,271 losses (round-limit games count as losses). ± is a 95% interval.

| measure | self-play winner | self-play loser | paired difference | learner wins vs heuristic | learner losses vs heuristic | difference |
|---|---|---|---|---|---|---|
| normal attacks / round | 0.65 | 0.68 | -0.02 ± 0.00 | 0.69 | 0.70 | -0.01 ± 0.01 |
| elemental skills / round | 0.92 | 0.73 | +0.19 ± 0.00 | 0.89 | 0.66 | +0.23 ± 0.01 |
| bursts / round | 0.36 | 0.21 | +0.14 ± 0.00 | 0.37 | 0.24 | +0.13 ± 0.00 |
| burst share of skills | 0.18 | 0.13 | +0.06 ± 0.00 | 0.19 | 0.14 | +0.04 ± 0.00 |
| cards played / round | 0.95 | 0.96 | -0.01 ± 0.01 | 0.91 | 0.80 | +0.11 ± 0.02 |
| talents played / game | 0.03 | 0.05 | -0.02 ± 0.00 | 0.03 | 0.05 | -0.02 ± 0.01 |
| tunings / round | 1.67 | 1.45 | +0.22 ± 0.01 | 1.67 | 1.32 | +0.35 ± 0.02 |
| voluntary switches / round | 0.47 | 0.62 | -0.15 ± 0.00 | 0.47 | 0.65 | -0.18 ± 0.02 |
| actions / round | 5.89 | 5.55 | +0.34 ± 0.01 | 5.88 | 5.28 | +0.59 ± 0.04 |
| share of rounds it ended first | 0.42 | 0.47 | -0.06 ± 0.00 | 0.19 | 0.31 | -0.11 ± 0.01 |
| nominal dice spent / round | 7.45 | 6.67 | +0.78 ± 0.01 | 7.43 | 6.43 | +1.00 ± 0.04 |
| dice rerolled / reroll | 4.53 | 4.49 | +0.03 ± 0.01 | 4.55 | 4.50 | +0.05 ± 0.02 |
| mulligan cards changed | 2.36 | 2.23 | +0.13 ± 0.01 | 2.37 | 2.01 | +0.36 ± 0.05 |
| bursts / game | 2.11 | 1.29 | +0.82 ± 0.01 | 2.21 | 1.78 | +0.43 ± 0.04 |
| cards played / game | 5.70 | 5.73 | -0.02 ± 0.04 | 5.61 | 6.02 | -0.41 ± 0.15 |
| nominal dice spent / game | 44.03 | 39.55 | +4.49 ± 0.07 | 44.87 | 47.01 | -2.13 ± 0.56 |
| own characters defeated | 1.26 | 2.97 | -1.71 ± 0.01 | 1.09 | 2.89 | -1.80 ± 0.02 |
| deck: action cards | 29.84 | 27.85 | +1.98 ± 0.07 | 29.79 | 25.20 | +4.59 ± 0.33 |
| deck: distinct cards | 22.14 | 21.37 | +0.77 ± 0.10 | 22.31 | 19.49 | +2.82 ± 0.33 |
| deck: talent copies | 1.72 | 1.59 | +0.13 ± 0.02 | 1.71 | 1.45 | +0.26 ± 0.05 |
| deck: largest same-element group | 1.36 | 1.41 | -0.05 ± 0.01 | 1.36 | 1.44 | -0.07 ± 0.02 |
| deck: mean card cost | 1.54 | 1.53 | +0.00 ± 0.00 | 1.54 | 1.53 | +0.01 ± 0.01 |

### Learner win rate by deck feature (updates 400-600)

Rate [95% interval] (n). In self-play the average is 0.5 by construction, so a group's rate measures its edge over the decks it meets. Groups overlap and are confounded with source (PvE decks are stage decks).

| learner's deck | vs heuristic | self-play |
|---|---|---|
| no action cards (PvE stage deck) | 0.134 [0.112, 0.160] (n=789) | 0.070 [0.062, 0.078] (n=4,176) |
| 30 action cards | 0.809 [0.804, 0.815] (n=18,831) | 0.516 [0.513, 0.519] (n=104,118) |
| talent in deck | 0.812 [0.805, 0.818] (n=14,381) | 0.519 [0.516, 0.522] (n=79,659) |
| no talent in deck | 0.802 [0.790, 0.813] (n=4,450) | 0.507 [0.501, 0.514] (n=24,459) |
| resonance card in deck | 0.807 [0.795, 0.818] (n=4,439) | 0.532 [0.526, 0.539] (n=24,369) |
| 3 characters, 3 different elements | 0.815 [0.808, 0.822] (n=11,940) | 0.526 [0.523, 0.530] (n=65,958) |
| 3 characters, 2 share an element | 0.776 [0.766, 0.786] (n=7,012) | 0.487 [0.482, 0.492] (n=38,712) |
| 3 characters, all one element | 0.475 [0.389, 0.563] (n=122) | 0.207 [0.179, 0.239] (n=694) |
| 1 or 2 characters (PvE) | 0.212 [0.173, 0.256] (n=378) | 0.145 [0.131, 0.162] (n=1,952) |
| 4 characters (PvE) | 0.196 [0.143, 0.263] (n=168) | 0.054 [0.042, 0.070] (n=978) |

### PvE stages by group (updates 400-600)

Groups are inferred from the stage's enemy deck (see deck_rows.py); many weekly-guest stages reuse invitation decks and are counted under 'invitation'.

| stage group | learner (collection deck) vs heuristic playing the stage deck | learner vs learner playing the stage deck (self-play) | learner vs snapshot playing the stage deck | learner playing the stage deck vs heuristic |
|---|---|---|---|---|
| serious_showdown | 0.671 [0.624, 0.714] (n=416) | 0.369 [0.350, 0.388] (n=2,445) | 0.362 [0.319, 0.407] (n=459) | 0.924 [0.894, 0.946] (n=408) |
| tavern | 0.756 [0.706, 0.800] (n=320) | 0.791 [0.773, 0.809] (n=1,966) | 0.825 [0.781, 0.861] (n=342) | 0.301 [0.256, 0.351] (n=355) |
| ascension | 0.885 [0.710, 0.960] (n=26) | 0.978 [0.923, 0.994] (n=91) | 1.000 [0.772, 1.000] (n=13) | 0.042 [0.007, 0.202] (n=24) |
| invitation | 0.703 [0.687, 0.718] (n=3,300) | 0.393 [0.386, 0.400] (n=18,834) | 0.445 [0.428, 0.462] (n=3,432) | 0.901 [0.891, 0.911] (n=3,430) |
| weekly_guest | 0.896 [0.865, 0.920] (n=471) | 0.855 [0.840, 0.868] (n=2,429) | 0.868 [0.834, 0.897] (n=441) | 0.233 [0.195, 0.276] (n=420) |
| open_world | 0.770 [0.717, 0.816] (n=274) | 0.724 [0.700, 0.748] (n=1,339) | 0.739 [0.680, 0.790] (n=245) | 0.438 [0.379, 0.498] (n=265) |

### How predictable self-play results are from the decks alone

Trained on 40,433 games, scored on 40,392 other games. Lower log loss/Brier is better; AUC 0.5 = no skill, 1 = perfect. The value head at round 0 sees both lineups (and the opponent's cards in NPC mode) before any play.

| predictor (held-out self-play games, updates 300-600) | log loss | Brier | accuracy | AUC | calibration error |
|---|---|---|---|---|---|
| constant (always the average) | 0.693 | 0.250 | 0.507 | 0.500 | 0.000 |
| deck-only model (characters + cards + seat) | 0.576 | 0.197 | 0.694 | 0.765 | 0.013 |
| value head at start of round 0 (n=40,392) | 0.612 | 0.210 | 0.673 | 0.747 | 0.075 |
| value head at start of round 1 (n=40,392) | 0.616 | 0.212 | 0.672 | 0.749 | 0.082 |
| value head at start of round 3 (n=40,312) | 0.556 | 0.186 | 0.722 | 0.807 | 0.072 |
| value head at start of round 5 (n=35,926) | 0.422 | 0.134 | 0.808 | 0.898 | 0.052 |

### Character strength in self-play (deck-only model)

Strongest and weakest 12 of 129 characters seen at least 200 times. +0.4 log-odds turns a 50% game into about 60%. NPC characters always come with their fixed NPC deck, so their effect includes that deck. Correlated features share credit; read as a ranking, not exact values.

| character | log-odds effect | decks containing it (training games) |
|---|---|---|
| Cyno | +0.63 | 1,135 |
| Qiqi | +0.55 | 684 |
| Fischl | +0.54 | 9,436 |
| Ganyu | +0.50 | 1,191 |
| Stonehide Lawachurl | +0.49 | 481 |
| Thoma | +0.47 | 870 |
| Ineffa | +0.45 | 866 |
| Frost Operative | +0.43 | 356 |
| Fatui Pyro Agent | +0.41 | 799 |
| Xinyan | +0.40 | 831 |
| Klee | +0.35 | 1,114 |
| Xiangling | +0.33 | 1,851 |
| ... |  |  |
| Illuga | -0.30 | 311 |
| Emilie | -0.30 | 362 |
| Lauma | -0.32 | 593 |
| Yoimiya | -0.33 | 876 |
| Kinich | -0.33 | 362 |
| Blazing Axe Mitachurl | -0.34 | 276 |
| Nefer | -0.35 | 349 |
| Xingqiu | -0.37 | 2,069 |
| Xianyun | -0.38 | 361 |
| Chasca | -0.39 | 344 |
| Maguu Kenki | -0.48 | 400 |
| Mavuika | -0.62 | 310 |

### Card strength in self-play (deck-only model, per copy)

Strongest and weakest 12 of 406 cards seen at least 200 times. +0.4 log-odds turns a 50% game into about 60%. NPC characters always come with their fixed NPC deck, so their effect includes that deck. Correlated features share credit; read as a ranking, not exact values.

| card | log-odds effect | decks containing it (training games) |
|---|---|---|
| The Wondrous Path of Truth | +0.30 | 202 |
| Undivided Heart | +0.29 | 534 |
| Vanarana | +0.28 | 1,217 |
| Adamah's Redemption | +0.28 | 200 |
| Rainbow Macarons | +0.26 | 2,176 |
| Abyssal Summons | +0.25 | 246 |
| Gift of the Goddess of Prophecy | +0.25 | 894 |
| Elemental Resonance: Woven Weeds | +0.24 | 603 |
| Elemental Resonance: Woven Stone | +0.24 | 1,183 |
| Floral Sidewinder | +0.23 | 469 |
| Nasha Town | +0.22 | 1,252 |
| Rockin' in a Flaming World | +0.22 | 335 |
| ... |  |  |
| Victory Flows from the Turning of Tides | -0.12 | 253 |
| Right of Final Interpretation | -0.12 | 388 |
| Skyward Pride | -0.13 | 406 |
| Night Realm's Gift: Repaid in Full | -0.13 | 224 |
| Bullet Trick | -0.13 | 240 |
| Beneficent | -0.14 | 219 |
| Dark-Shattering Flame | -0.15 | 242 |
| They Call Her Cloud Retainer | -0.16 | 222 |
| A Hero of Justice's Triumph | -0.17 | 325 |
| Companion's Counsel | -0.19 | 343 |
| "O Lips, Weave Me Songs and Psalms" | -0.19 | 233 |
| Vanguard's Coordinated Tactics | -0.22 | 257 |

### Learner vs heuristic by deck edge (updates 400-600)

The deck-only model says how often the learner's deck beats the heuristic's deck when both are played by the same network. The gap between the two curves is what better play is worth.

| deck-only win chance of the learner's deck | games | learner win rate |
|---|---|---|
| 0.01-0.20 | 1,958 | 0.338 [0.317, 0.359] |
| 0.20-0.30 | 1,890 | 0.640 [0.618, 0.662] |
| 0.30-0.38 | 1,950 | 0.728 [0.708, 0.747] |
| 0.38-0.44 | 1,969 | 0.802 [0.784, 0.819] |
| 0.44-0.50 | 2,062 | 0.823 [0.805, 0.838] |
| 0.50-0.56 | 1,887 | 0.866 [0.850, 0.881] |
| 0.56-0.63 | 1,996 | 0.883 [0.868, 0.896] |
| 0.63-0.70 | 1,971 | 0.901 [0.887, 0.913] |
| 0.70-0.80 | 1,887 | 0.919 [0.906, 0.930] |
| 0.80-0.99 | 2,050 | 0.916 [0.903, 0.927] |

### Hardest stage lineups for the learner (collection deck, updates 400-600)

Stage lineups with at least 150 games where the learner holds the collection deck (opponents: self-play, snapshots, heuristic; not random). Overall rate 0.517 over 36,843 games and 97 lineups that qualify.

| stage lineup | games | learner win rate | win rate when the heuristic plays the stage |
|---|---|---|---|
| Escoffier, Furina, Candace | 257 | 0.105 [0.073, 0.149] | 0.09 (n=35) |
| Sigewinne, Qiqi, Neuvillette | 265 | 0.121 [0.087, 0.165] | 0.19 (n=36) |
| Ineffa, Cyno, Mona | 314 | 0.121 [0.089, 0.162] | 0.22 (n=36) |
| Ganyu, Diona, Keqing | 283 | 0.155 [0.118, 0.202] | 0.21 (n=33) |
| Charlotte, Barbara, Qiqi | 279 | 0.161 [0.123, 0.209] | 0.21 (n=33) |
| Wriothesley, Thoma, Shenhe | 340 | 0.162 [0.126, 0.205] | 0.38 (n=47) |
| Qiqi, Diona, Yanfei | 304 | 0.197 [0.157, 0.246] | 0.42 (n=50) |
| Aino, Ineffa, Sangonomiya Kokomi | 256 | 0.215 [0.169, 0.269] | 0.42 (n=36) |
| Furina, Lyney, Neuvillette | 153 | 0.242 [0.181, 0.315] | 0.30 (n=20) |
| Jahoda, Aino, Ineffa | 282 | 0.252 [0.205, 0.306] | 0.57 (n=30) |
| Ororon, Sangonomiya Kokomi, Ifa | 292 | 0.257 [0.210, 0.310] | 0.78 (n=40) |
| Fischl, Xiangling, Diona | 303 | 0.257 [0.211, 0.309] | 0.70 (n=44) |
| Xinyan, Yun Jin, Hu Tao | 256 | 0.270 [0.219, 0.327] | 0.62 (n=34) |
| Ifa, Mona, Lisa | 284 | 0.271 [0.223, 0.326] | 0.85 (n=26) |
| Dahlia, Diona, Mona | 291 | 0.278 [0.230, 0.332] | 0.43 (n=35) |

### When games are decided (learner seats, held-out games, updates 400-600)

Chances are isotonic-calibrated (one map for all opponents; it under-rates the learner vs heuristic and random). 'Decided round' = first round start from which the estimate stays >= 0.8 (wins) or <= 0.2 (losses) to the end; 'clear >= 3 rounds before the end' counts the deciding round itself.

| matchup | learner seats | win rate | rounds | decided round | clear >= 3 rounds before the end | never clear before the end | comeback rate (was <= 0.2) | collapse rate (was >= 0.8) | swing per round | times crossing 50% |
|---|---|---|---|---|---|---|---|---|---|---|
| collections / heuristic | 2,493 | 0.856 | 5.92 | 4.82 | 0.370 | 0.287 | 0.317 (n=252) | 0.054 (n=1,830) | 0.089 | 0.53 |
| collections / random | 594 | 0.998 | 5.75 | 3.63 | 0.588 | 0.030 | 0.955 (n=22) | 0.000 (n=576) | 0.090 | 0.38 |
| collections / self | 13,734 | 0.500 | 5.56 | 4.97 | 0.244 | 0.363 | 0.079 (n=4,704) | 0.142 (n=5,797) | 0.095 | 0.57 |
| collections / snapshot | 2,426 | 0.516 | 5.65 | 4.95 | 0.269 | 0.354 | 0.081 (n=806) | 0.129 (n=1,069) | 0.095 | 0.59 |
| npc_decks / heuristic | 2,487 | 0.761 | 6.77 | 5.67 | 0.339 | 0.326 | 0.233 (n=442) | 0.079 (n=1,578) | 0.081 | 0.58 |
| npc_decks / random | 582 | 0.997 | 6.34 | 4.29 | 0.552 | 0.077 | 0.960 (n=25) | 0.000 (n=536) | 0.084 | 0.45 |
| npc_decks / self | 13,352 | 0.498 | 6.35 | 5.59 | 0.268 | 0.374 | 0.089 (n=4,631) | 0.140 (n=5,357) | 0.084 | 0.62 |
| npc_decks / snapshot | 2,453 | 0.534 | 6.37 | 5.56 | 0.267 | 0.361 | 0.086 (n=779) | 0.131 (n=1,103) | 0.085 | 0.59 |
| stage / heuristic (collection side) | 2,389 | 0.732 | 6.38 | 5.23 | 0.362 | 0.324 | 0.241 (n=498) | 0.091 (n=1,497) | 0.079 | 0.58 |
| stage / heuristic (stage side) | 2,490 | 0.780 | 6.19 | 4.30 | 0.488 | 0.224 | 0.159 (n=491) | 0.036 (n=1,684) | 0.073 | 0.40 |
| stage / random (collection side) | 605 | 0.993 | 6.22 | 4.43 | 0.483 | 0.091 | 0.952 (n=42) | 0.004 (n=550) | 0.084 | 0.48 |
| stage / random (stage side) | 670 | 0.939 | 6.37 | 3.81 | 0.648 | 0.084 | 0.570 (n=93) | 0.002 (n=577) | 0.074 | 0.30 |
| stage / self (collection side) | 13,480 | 0.482 | 5.92 | 4.72 | 0.359 | 0.310 | 0.072 (n=5,289) | 0.116 (n=5,475) | 0.078 | 0.49 |
| stage / self (stage side) | 13,480 | 0.516 | 5.92 | 4.65 | 0.374 | 0.307 | 0.062 (n=4,771) | 0.121 (n=6,078) | 0.078 | 0.47 |
| stage / snapshot (collection side) | 2,441 | 0.528 | 5.92 | 4.75 | 0.364 | 0.313 | 0.084 (n=876) | 0.096 (n=1,071) | 0.078 | 0.50 |
| stage / snapshot (stage side) | 2,558 | 0.538 | 5.92 | 4.58 | 0.393 | 0.295 | 0.068 (n=894) | 0.105 (n=1,196) | 0.076 | 0.46 |

### Value head calibration (held-out round starts, updates 400-600)

One point per learner seat per round start. The calibrator is one isotonic fit over all opponents (fitted on other games); a per-opponent fit would remove the remaining bias.

| points | round starts | mean predicted | actual win rate | Brier raw | Brier calibrated | calibration error raw | calibration error calibrated | AUC |
|---|---|---|---|---|---|---|---|---|
| all | 456,784 | 0.560 | 0.552 | 0.166 | 0.162 | 0.058 | 0.002 | 0.839 |
| vs self | 320,532 | 0.536 | 0.498 | 0.169 | 0.165 | 0.063 | 0.034 | 0.839 |
| vs snapshot | 58,910 | 0.546 | 0.529 | 0.166 | 0.162 | 0.057 | 0.012 | 0.842 |
| vs heuristic | 62,220 | 0.652 | 0.750 | 0.160 | 0.155 | 0.119 | 0.120 | 0.813 |
| vs random | 15,122 | 0.744 | 0.969 | 0.131 | 0.127 | 0.225 | 0.253 | 0.947 |
| source stage | 227,702 | 0.553 | 0.549 | 0.155 | 0.151 | 0.054 | 0.005 | 0.860 |
| source npc_decks | 120,824 | 0.558 | 0.547 | 0.180 | 0.175 | 0.064 | 0.005 | 0.813 |
| source collections | 108,258 | 0.576 | 0.563 | 0.175 | 0.170 | 0.060 | 0.005 | 0.821 |
| stage, learner holds the collection deck | 113,163 | 0.539 | 0.521 | 0.162 | 0.157 | 0.059 | 0.020 | 0.852 |
| NPC mode (knows opponent deck) | 262,034 | 0.555 | 0.549 | 0.164 | 0.160 | 0.056 | 0.003 | 0.843 |
| blind mode | 194,750 | 0.566 | 0.556 | 0.169 | 0.164 | 0.060 | 0.003 | 0.833 |
| rounds 1-2 | 152,466 | 0.564 | 0.555 | 0.206 | 0.200 | 0.070 | 0.011 | 0.750 |
| rounds 3-4 | 151,556 | 0.564 | 0.555 | 0.165 | 0.162 | 0.052 | 0.010 | 0.839 |
| rounds 5-6 | 111,443 | 0.559 | 0.554 | 0.127 | 0.124 | 0.047 | 0.008 | 0.905 |
| rounds 7-15 | 41,319 | 0.528 | 0.522 | 0.130 | 0.125 | 0.063 | 0.013 | 0.903 |

### Isotonic map: raw value-head win chance -> observed win rate

| raw | 0.02 | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 | 0.50 | 0.70 | 0.80 | 0.85 | 0.90 | 0.95 | 0.98 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| observed | 0.059 | 0.122 | 0.197 | 0.256 | 0.307 | 0.386 | 0.505 | 0.634 | 0.702 | 0.752 | 0.805 | 0.871 | 0.933 |

### What a 'concede below p' rule would have done (held-out games, updates 400-600)

Checked at each round start from round 1 with the isotonic-calibrated chance. 'Rounds saved' counts the concession round itself.

| games | concede when calibrated chance < p | games | share conceded | win rate of conceded games | share of all wins thrown away | mean rounds saved per concession |
|---|---|---|---|---|---|---|
| vs heuristic | 0.02 | 9,859 | 0.056 | 0.022 | 0.002 | 2.89 |
| vs heuristic | 0.05 | 9,859 | 0.083 | 0.058 | 0.006 | 3.31 |
| vs heuristic | 0.10 | 9,859 | 0.117 | 0.114 | 0.017 | 3.72 |
| vs heuristic | 0.15 | 9,859 | 0.145 | 0.177 | 0.033 | 4.09 |
| vs heuristic | 0.20 | 9,859 | 0.171 | 0.226 | 0.049 | 4.37 |
| stage, learner holds the collection deck (not vs random) | 0.02 | 18,310 | 0.117 | 0.006 | 0.001 | 1.57 |
| stage, learner holds the collection deck (not vs random) | 0.05 | 18,310 | 0.192 | 0.015 | 0.005 | 1.88 |
| stage, learner holds the collection deck (not vs random) | 0.10 | 18,310 | 0.274 | 0.039 | 0.020 | 2.43 |
| stage, learner holds the collection deck (not vs random) | 0.15 | 18,310 | 0.326 | 0.065 | 0.041 | 2.86 |
| stage, learner holds the collection deck (not vs random) | 0.20 | 18,310 | 0.364 | 0.086 | 0.060 | 3.22 |
| self-play | 0.02 | 54,046 | 0.115 | 0.008 | 0.002 | 1.99 |
| self-play | 0.05 | 54,046 | 0.187 | 0.016 | 0.006 | 2.18 |
| self-play | 0.10 | 54,046 | 0.270 | 0.035 | 0.019 | 2.51 |
| self-play | 0.15 | 54,046 | 0.321 | 0.055 | 0.036 | 2.79 |
| self-play | 0.20 | 54,046 | 0.359 | 0.075 | 0.054 | 3.05 |

### Seat effect (updates 400-600)

Seat 0 acts first in round 1. Round-limit games count as a loss for both seats.

| source | self-play: seat 0 win rate | learner vs heuristic, learner in seat 0 | learner vs heuristic, learner in seat 1 |
|---|---|---|---|
| all | 0.503 [0.499, 0.507] (n=54,147) | 0.779 [0.771, 0.787] (n=9,776) | 0.786 [0.777, 0.794] (n=9,844) |
| stage | 0.501 [0.495, 0.507] (n=27,104) | 0.745 [0.733, 0.757] (n=4,833) | 0.756 [0.744, 0.768] (n=4,876) |
| npc_decks | 0.498 [0.490, 0.506] (n=13,439) | 0.764 [0.747, 0.780] (n=2,431) | 0.771 [0.755, 0.787] (n=2,519) |
| collections | 0.512 [0.503, 0.520] (n=13,604) | 0.858 [0.844, 0.871] (n=2,512) | 0.860 [0.845, 0.873] (n=2,449) |

### NPC mode (knows the opponent's deck) vs blind mode (updates 400-600)

Compared within one source and side, because knowledge odds differ by side in stage games (collection side 80%, stage side 50%) and pooling would mix in the side's own win rate. In symmetric self-play the clean test is games where exactly one seat knows: a win rate above 0.5 is the value of the information.

| learner seats | win rate, knows the deck | win rate, blind | difference (95%) |
|---|---|---|---|
| self-play npc_decks, only this seat knows | 0.499 [0.487, 0.511] (n=6,756) | (other seat blind) | - |
| self-play collections, only this seat knows | 0.503 [0.491, 0.515] (n=6,796) | (other seat blind) | - |
| self-play stage, collection side | 0.477 [0.471, 0.484] (n=21,624) | 0.489 [0.476, 0.502] (n=5,480) | -0.012 ± 0.015 |
| self-play stage, stage side | 0.521 [0.513, 0.529] (n=13,639) | 0.517 [0.508, 0.525] (n=13,465) | +0.004 ± 0.012 |
| vs heuristic, npc_decks | 0.757 [0.740, 0.774] (n=2,478) | 0.778 [0.761, 0.794] (n=2,472) | -0.020 ± 0.024 |
| vs heuristic, collections | 0.853 [0.838, 0.866] (n=2,478) | 0.865 [0.851, 0.878] (n=2,483) | -0.012 ± 0.019 |
| vs heuristic, stage collection side | 0.730 [0.716, 0.744] (n=3,816) | 0.718 [0.690, 0.746] (n=991) | +0.011 ± 0.031 |
| vs heuristic, stage side | 0.774 [0.756, 0.790] (n=2,381) | 0.773 [0.756, 0.789] (n=2,521) | +0.001 ± 0.023 |

### Heuristic games (updates 400-600): the two sides in heuristic wins and losses

| measure | heuristic, its wins (n=4,200) | heuristic, its losses (n=15,420) | learner, in games the heuristic won | learner, in games it won |
|---|---|---|---|---|
| bursts / game | 2.08 | 1.17 | 1.73 | 2.21 |
| bursts / round | 0.29 | 0.18 | 0.23 | 0.37 |
| elemental skills / round | 0.61 | 0.58 | 0.66 | 0.89 |
| normal attacks / round | 0.69 | 0.67 | 0.70 | 0.69 |
| cards played / round | 2.43 | 2.38 | 0.80 | 0.91 |
| talents played / game | 0.36 | 0.28 | 0.05 | 0.03 |
| tunings / round | 0.32 | 0.31 | 1.32 | 1.67 |
| voluntary switches / round | 0.21 | 0.24 | 0.64 | 0.47 |
| share of rounds it ended first | 0.60 | 0.70 | 0.31 | 0.19 |
| nominal dice spent / round | 8.31 | 7.83 | 6.42 | 7.43 |
| own characters defeated | 0.94 | 2.97 | 2.93 | 1.09 |

### Do heuristic wins come from the deck? (updates 400-600)

19,620 games. Deck-only win chance from the model fitted on self-play (same network on both sides), so it measures the matchup, not the play.

| condition | share of all heuristic games | share of heuristic wins | heuristic win rate when true | heuristic win rate otherwise |
|---|---|---|---|---|
| learner's deck-only win chance < 0.5 | 0.496 | 0.762 | 0.329 [0.320, 0.338] | 0.101 [0.095, 0.107] |
| learner's deck-only win chance < 0.3 | 0.194 | 0.461 | 0.508 [0.492, 0.524] | 0.143 [0.138, 0.149] |
| learner's deck-only win chance < 0.2 | 0.098 | 0.301 | 0.657 [0.636, 0.678] | 0.166 [0.161, 0.171] |

### Learner's own (calibrated) win estimate in heuristic games

| round start | mean estimate, learner won | mean estimate, learner lost | share of losses where it still thought it was ahead |
|---|---|---|---|
| 1 | 0.606 | 0.406 | 0.366 (n=4,200) |
| 2 | 0.653 | 0.424 | 0.407 (n=4,200) |
| 3 | 0.697 | 0.426 | 0.409 (n=4,196) |
| 4 | 0.747 | 0.412 | 0.395 (n=4,150) |

### Cards played vs tuned away, per game (vs heuristic, updates 400-600, 19,620 games)

Elemental tuning discards a card to change one die to the active character's element.

| card category | learner played | learner tuned | learner share played | heuristic played | heuristic tuned | heuristic share played |
|---|---|---|---|---|---|---|
| talent | 0.03 | 0.81 | 0.04 | 0.30 | 0.48 | 0.38 |
| equipment | 0.92 | 3.42 | 0.21 | 4.16 | 0.57 | 0.88 |
| support | 1.94 | 1.75 | 0.53 | 4.01 | 0.14 | 0.97 |
| event | 2.81 | 3.87 | 0.42 | 6.64 | 0.79 | 0.89 |

### Cards the learner tunes away most / plays most (vs heuristic, updates 400-600)

Cards with at least 150 uses on each side. The heuristic only tunes when no skill is affordable, so its share is a 'play it when possible' baseline.

| card | category | learner plays + tunes | learner share played | heuristic share played |
|---|---|---|---|---|
| Grand Expectation | talent | 176 | 0.00 | 0.00 |
| Chaotic Entropy | talent | 804 | 0.00 | 0.00 |
| Master of Weaponry | event | 2,931 | 0.00 | 0.09 |
| A Summation of Interest | talent | 163 | 0.01 | 0.30 |
| Undivided Heart | talent | 278 | 0.01 | 0.17 |
| Stellar Predator | talent | 792 | 0.01 | 0.38 |
| Companion's Counsel | talent | 150 | 0.01 | 0.26 |
| Guardian's Oath | event | 2,761 | 0.01 | 0.65 |
| Countless Sights to See | talent | 207 | 0.01 | 0.36 |
| Floral Sidewinder | talent | 197 | 0.02 | 0.32 |
| Flowing Flame | talent | 793 | 0.02 | 0.38 |
| Mystical Abandon | talent | 151 | 0.02 | 0.20 |
| Saurian Dining Buddies | event | 315 | 0.03 | 0.21 |
| Beacon of the Reed Sea | equipment | 215 | 0.03 | 0.61 |
| Light of Foliar Incision | equipment | 219 | 0.04 | 0.52 |
| ... |  |  |  |  |
| Liben | support | 1,302 | 0.82 | 1.00 |
| Timmie | support | 2,721 | 0.82 | 1.00 |
| Elemental Resonance: Woven Winds | event | 161 | 0.83 | 1.00 |
| Elemental Resonance: Enduring Rock | event | 751 | 0.83 | 0.96 |
| Rainbow Macarons | event | 1,096 | 0.83 | 0.97 |
| Genius's Upgrade Technique | event | 487 | 0.83 | 1.00 |
| Elemental Resonance: Woven Waters | event | 1,179 | 0.83 | 1.00 |
| Nasha Town | support | 586 | 0.84 | 1.00 |
| Opera Epiclese | support | 451 | 0.84 | 1.00 |
| Elemental Resonance: Woven Ice | event | 1,039 | 0.87 | 1.00 |
| The Bestest Travel Companion! | event | 9,488 | 0.88 | 0.97 |
| Knights of Favonius Library | support | 1,527 | 0.90 | 1.00 |
| Tideseal Stone | support | 320 | 0.93 | 1.00 |
| Vanarana | support | 517 | 0.93 | 0.99 |
| Favonius Cathedral | support | 4,338 | 0.97 | 0.96 |
