# Card play versus elemental tuning diagnostic

*Written 2026-10-15. This uses `runs/alpha_02/checkpoint_02000.pt`, 300 newly played benchmark-deck games, 3,130 eligible decisions, and 23,040 counterfactual games. No training was run.*

## Plain-language summary

The main answer is **C: it depends on the card and situation**, with several broad types closer to B than A. There is no evidence that simply playing all the equipment and supports that the bot tunes away would improve it.

- **Weapon — C.** Playing won 55.6% versus 50.9% after tuning, but the situation-cluster interval is compatible with either action being better: +4.6 percentage points (-2.7 to +12.1). With 0–2 dice, playing won 12.8 points more often (+2.8 to +24.4); the network already recognizes this slice, assigning play 64.9% and tune 0.4%.
- **Artifact — B overall, with a specific A exception.** Both branches won 48.3%; the difference was 0.0 points (-3.9 to +3.7). Tuning artifacts broadly is defensible. Gambler's Earrings is the clearest exception: play gained 4.0 points (+0.5 to +8.5) across five situations while the network assigned play 2.6% and tune 35.4%.
- **Support — B.** Playing and tuning were effectively tied: 57.4% versus 56.8%, difference +0.6 points (-3.8 to +5.3). No dice band had a resolved advantage.
- **Talent — B.** Playing was 48.9% versus 51.6% after tuning, difference -2.7 points (-7.1 to +0.9). This does not support a remaining general talent-underplay problem.
- **Event — B.** Wins were identical within noise (+0.1 points, -4.3 to +4.0), but playing took 0.095 more remaining rounds (+0.016 to +0.190). Tuning was faster under the run's speed objective.
- **Food — C, leaning B.** Playing gained an unresolved 2.2 win-rate points (-2.1 to +6.6) but took 0.182 more remaining rounds (+0.066 to +0.324). Sweet Madame was a real play-favoring exception: +10.6 points (+1.3 to +21.2) across eight situations.

So the original five-cards-played statistic does identify unusual behavior, but it does **not** by itself prove a broad policy failure. The bot often spends its hand as a dice resource without losing win rate, and sometimes finishes faster by doing so. The useful training target is the narrower set of states where its card-versus-tune ranking is wrong, not card play in general.

## What was tested

The collector played 300 games drawn from the frozen benchmark matchups with the final `alpha 2` network controlling both seats. It retained at most two eligible decisions per game and card type, using reservoir sampling within each game. An eligible decision had to offer both a legal play of a card and legal elemental tuning of that same card. This produced 3,130 decisions across 300 games:

| Card type | Eligible decisions | Distinct source games |
|---|---:|---:|
| Weapon | 492 | 252 |
| Artifact | 582 | 293 |
| Support | 596 | 299 |
| Talent | 328 | 188 |
| Event | 600 | 300 |
| Food | 532 | 270 |

For the causal test, 48 decisions of each type were selected, stratified as evenly as the data allowed over 0–2, 3–5 and 6+ available dice. Talent and weapon decisions with 0–2 dice were rare, so their unused quota was filled from the other bands.

Each selected game was deterministically replayed from its original shuffled decks and seed, using the exact recorded responses up to the decision. A SHA-256 fingerprint over the complete exposed state, request and counters had to match the saved decision. The card was then forced to be played or tuned. If a card had multiple legal play targets, the target was sampled from the network's probabilities conditional on playing that card. The stochastic final network controlled both seats afterward.

There were 40 rollouts per branch per decision: 288 decisions × 2 branches × 40 = **23,040 complete games**. All 576 decision/branch cells contain exactly 40 games. There were no engine failures, missing options, prefix mismatches or fingerprint mismatches.

The initial shuffle and future engine randomness came from the same seed and replayed prefix in both branches. Source games were collected one at a time, and every stochastic response uses a seed tied to its game and decision, so asynchronous batching cannot change an action. Post-branch network sampling varies between rollouts but is exactly reproducible.

## What the network thinks

The probabilities below are means over all collected eligible decisions, not just the smaller branch sample. “Other” is probability assigned to every action other than playing or tuning this particular card. Confidence intervals are 95% game-cluster bootstrap intervals. The value head estimates the shared pre-action state's expected speed-adjusted reward on a -1 to +1 scale; it is **not action-conditional**, so there is not a separate value output for play and tune. The win head is shown only to make the state interpretation easier and is known from the main review to be uncalibrated.

| Type | Decisions | Play probability | Tune probability | Other | Value estimate | Win-head probability |
|---|---:|---:|---:|---:|---:|---:|
| Weapon | 492 | 5.5% [4.3, 8.4] | 10.9% [9.1, 13.4] | 83.6% | +0.050 [-0.019, +0.112] | 53.9% [49.1, 58.0] |
| Artifact | 582 | 2.2% [1.6, 2.8] | 25.2% [22.9, 28.5] | 72.6% | +0.004 [-0.052, +0.058] | 52.3% [48.4, 56.1] |
| Support | 596 | 8.9% [7.1, 11.0] | 12.6% [10.4, 14.7] | 78.5% | +0.049 [-0.004, +0.106] | 55.6% [51.9, 59.5] |
| Talent | 328 | 35.3% [34.0, 44.9] | 10.7% [7.5, 12.7] | 54.0% | -0.001 [-0.077, +0.074] | 53.1% [47.9, 58.2] |
| Event | 600 | 10.3% [8.6, 12.1] | 10.4% [8.6, 12.3] | 79.3% | +0.096 [+0.040, +0.151] | 56.8% [53.0, 60.7] |
| Food | 532 | 10.4% [8.6, 12.9] | 8.1% [6.7, 10.3] | 81.5% | +0.004 [-0.061, +0.063] | 52.6% [48.0, 56.6] |

The largest pattern is dice-dependent. When only 0–2 dice remain, a card that is still playable is often cheap or free, and tuning a card has less immediate use. At 3–5 dice, the network sharply suppresses most card play and often prefers tuning.

| Type | Dice | Decisions | Play probability | Tune probability | Value estimate |
|---|---:|---:|---:|---:|---:|
| Weapon | 0–2 | 9 | 64.9% [38.1, 88.9] | 0.4% [0.0, 1.0] | +0.061 [-0.353, +0.488] |
| Weapon | 3–5 | 174 | 1.3% [0.6, 2.6] | 15.4% [11.3, 20.8] | +0.131 [+0.034, +0.217] |
| Weapon | 6+ | 309 | 6.1% [4.3, 9.1] | 8.7% [6.2, 10.2] | +0.004 [-0.049, +0.093] |
| Artifact | 0–2 | 24 | 11.6% [4.8, 21.7] | 12.5% [5.5, 22.2] | +0.135 [-0.088, +0.336] |
| Artifact | 3–5 | 180 | 1.1% [0.6, 1.7] | 27.6% [23.6, 35.0] | +0.060 [-0.035, +0.139] |
| Artifact | 6+ | 378 | 2.1% [1.3, 3.1] | 24.9% [21.3, 27.6] | -0.031 [-0.086, +0.046] |
| Support | 0–2 | 35 | 61.9% [47.7, 75.4] | 1.5% [0.6, 2.8] | +0.043 [-0.149, +0.230] |
| Support | 3–5 | 215 | 3.8% [2.2, 5.3] | 17.6% [14.2, 23.4] | +0.056 [-0.029, +0.131] |
| Support | 6+ | 346 | 6.6% [4.4, 8.5] | 10.5% [8.1, 12.6] | +0.045 [-0.010, +0.126] |
| Talent | 0–2 | 1 | 40.0% [40.0, 40.0] | 0.2% [0.2, 0.2] | -0.514 [-0.514, -0.514] |
| Talent | 3–5 | 90 | 35.0% [26.5, 46.3] | 12.7% [7.6, 20.4] | +0.133 [+0.025, +0.252] |
| Talent | 6+ | 237 | 35.4% [31.6, 43.6] | 10.0% [6.3, 11.7] | -0.049 [-0.116, +0.055] |
| Event | 0–2 | 42 | 32.3% [20.0, 46.4] | 1.6% [0.4, 2.6] | +0.177 [-0.006, +0.321] |
| Event | 3–5 | 189 | 5.4% [3.1, 8.0] | 13.0% [9.8, 17.9] | +0.114 [+0.036, +0.211] |
| Event | 6+ | 369 | 10.3% [7.7, 12.1] | 10.0% [7.5, 12.0] | +0.077 [+0.007, +0.139] |
| Food | 0–2 | 38 | 38.0% [26.1, 52.3] | 1.6% [0.7, 2.9] | +0.174 [-0.029, +0.374] |
| Food | 3–5 | 175 | 2.9% [1.6, 4.3] | 10.7% [7.4, 15.3] | -0.000 [-0.099, +0.081] |
| Food | 6+ | 319 | 11.2% [7.5, 12.4] | 7.4% [5.6, 9.9] | -0.014 [-0.087, +0.062] |

## What actually happens after play versus tune

All intervals in this section use a 10,000-sample bootstrap over the 48 source situations, so 40 repeats of one state are not treated as 40 independent situations. “Round difference” is play minus tune in rounds remaining from the decision; negative is faster after playing.

| Type | Play win rate | Tune win rate | Play − tune win rate | Play − tune remaining rounds |
|---|---:|---:|---:|---:|
| Weapon | 55.6% [43.8, 67.0] | 50.9% [39.1, 62.9] | +4.6 pp [-2.7, +12.1] | -0.003 [-0.156, +0.138] |
| Artifact | 48.3% [36.7, 60.2] | 48.3% [36.5, 60.2] | +0.0 pp [-3.9, +3.7] | +0.043 [-0.079, +0.197] |
| Support | 57.4% [46.2, 68.4] | 56.8% [45.6, 68.1] | +0.6 pp [-3.8, +5.3] | +0.061 [-0.045, +0.163] |
| Talent | 48.9% [37.0, 60.9] | 51.6% [40.3, 62.9] | -2.7 pp [-7.1, +0.9] | +0.015 [-0.060, +0.093] |
| Event | 65.7% [54.9, 76.1] | 65.6% [54.9, 76.1] | +0.1 pp [-4.3, +4.0] | **+0.095 [+0.016, +0.190]** |
| Food | 63.2% [52.3, 73.5] | 61.0% [50.1, 72.0] | +2.2 pp [-2.1, +6.6] | **+0.182 [+0.066, +0.324]** |

No broad type has a win-rate interval wholly above zero. Event and food play are measurably slower without a resolved win-rate gain.

### Outcome differences by available dice

| Type | Dice | Situations | Play − tune win rate (95% cluster interval) |
|---|---:|---:|---:|
| Weapon | 0–2 / 3–5 / 6+ | 9 / 20 / 19 | **+12.8 [+2.8, +24.4]** / +4.4 [-9.4, +18.0] / +1.1 [-8.3, +12.4] pp |
| Artifact | 0–2 / 3–5 / 6+ | 16 / 16 / 16 | -0.8 [-3.4, +1.6] / +0.2 [-10.8, +9.8] / +0.6 [-3.9, +5.3] pp |
| Support | 0–2 / 3–5 / 6+ | 16 / 16 / 16 | -2.2 [-5.9, +0.6] / +3.9 [-7.2, +16.1] / -0.0 [-5.8, +6.2] pp |
| Talent | 0–2 / 3–5 / 6+ | 1 / 17 / 30 | +2.5 [+2.5, +2.5] / -5.7 [-15.4, +0.1] / -1.2 [-5.4, +3.1] pp |
| Event | 0–2 / 3–5 / 6+ | 16 / 16 / 16 | -4.4 [-11.6, +1.7] / +3.9 [-0.9, +9.7] / +0.8 [-8.0, +8.1] pp |
| Food | 0–2 / 3–5 / 6+ | 16 / 16 / 16 | +5.8 [-1.4, +13.6] / +0.6 [-7.3, +8.7] / +0.2 [-6.1, +6.9] pp |

The low-dice weapon result is a real play-favoring slice, but not a learned-policy failure: the policy already prefers play by 64.9% to 0.4%. The single low-dice talent state is too small to interpret.

### Individual cards

Card-level results are exploratory because the stratified category sample was not designed to give every card many independent states. Repeated cards with a cluster interval excluding zero or touching zero at one endpoint were:

| Card | Type | Situations | Network play / tune probability | Play − tune win rate |
|---|---|---:|---:|---:|
| Lithic Spear | Weapon | 4 | 2.7% / 2.6% | +20.6 pp [0.0, +60.0] |
| Raven Bow | Weapon | 5 | 17.9% / 15.7% | **+19.5 pp [+0.5, +43.0]** |
| Sweet Madame | Food | 8 | 17.6% / 2.9% | **+10.6 pp [+1.3, +21.2]** |
| Favonius Cathedral | Support | 7 | 32.6% / 0.2% | **+5.4 pp [+0.4, +10.7]** |
| Gambler's Earrings | Artifact | 5 | **2.6% / 35.4%** | **+4.0 pp [+0.5, +8.5]** |
| Skyward Atlas | Weapon | 3 | 4.0% / 24.2% | -23.3 pp [-70.0, 0.0] |

Gambler's Earrings is the clearest policy error: the network strongly prefers tuning, but play won more often across five situations. Raven Bow also has a resolved play advantage, although the policy's play/tune probabilities are already close. Sweet Madame and Favonius Cathedral favor play and are already recognized by the network. Skyward Atlas points the other way and agrees with the policy's preference for tuning. These are card-specific signals, not evidence for a broad A conclusion.

## Answer: A, B or C

| Type | Answer | Evidence-based interpretation |
|---|---|---|
| Weapon | **C** | Low-dice play helps and is already recognized; overall advantage remains unresolved and cards differ. |
| Artifact | **B** overall, **A** for Gambler's Earrings | Aggregate tie makes tuning defensible, but Gambler's is systematically misranked in this sample. |
| Support | **B** | Aggregate and every dice band are unresolved; no general play benefit. |
| Talent | **B** | No remaining broad underplay signal; tuning has the stronger point estimate. |
| Event | **B** | Same win rate after play or tune, but playing is significantly slower. |
| Food | **C**, leaning **B** | Overall win effect is unresolved and playing is slower, but Sweet Madame clearly favors play. |

This classification is deliberately conservative. “No significant difference” does not prove exact equality, but the intervals reject a large universal benefit from playing these cards. Earlier development pilots selected different source states and likewise found no broad type with a clear play advantage; their moving category point estimates reinforced the use of deterministic sampling and state-cluster intervals rather than raw rollout counts.

## Training suggestions

These are suggestions, not results. None requires or recommends a direct reward for playing cards.

1. **Suggestion — oversample ambiguous play/tune states.** Put naturally reached decisions where both actions are legal into a replay stratum, especially equipment states and cards such as Gambler's Earrings where the counterfactual ordering conflicts with the policy. Continue training on normal returns.
2. **Suggestion — add an action-conditioned auxiliary return target.** The current value head can only describe the state. A small auxiliary head could predict the multi-step return after each legal action from ordinary trajectories or occasional counterfactual branches. It should not alter the environment reward.
3. **Suggestion — use targeted counterfactual data sparingly.** Branch a small fraction of collected states during data generation, then train the auxiliary target or prioritize policy updates where the observed play/tune ordering conflicts with the policy. Keep a held-out branch set so the same counterfactuals are not used for both training and proof.
4. **Suggestion — test card-conditional exploration.** A controlled run could modestly flatten the policy only among legal play/tune alternatives in underexplored states, without rewarding card use. Compare final win rate on the locked branch set and normal benchmarks.
5. **Suggestion — inspect Gambler's Earrings first and preserve the low-dice weapon behavior.** They give one concrete error and one concrete success, which is a better debugging target than a blanket “play more cards” change.

## Limits

- The result applies to benchmark-deck situations reached by the stochastic `alpha 2` network, not every legal deck or live HoYoverse NPC policy.
- Only states where the same card was both legally playable and tunable are included. It does not explain cards that were unplayable, nor whether a different earlier action caused the bad hand.
- A forced first action can place the network slightly off its usual state distribution. That is intentional for the causal question, but it measures recovery under the current policy as well as the immediate card effect.
- Forty stochastic continuations reduce rollout noise, but the real independent sample is the 48 source situations per type. The cluster intervals reflect that.
- Play variants were sampled according to the network conditional on playing. The test therefore asks whether the network's own way of playing the card beats tuning, not whether an oracle-selected target would.
- GPU access was unavailable in the sandbox, so inference ran on CPU. This changes runtime, not model outputs or game rules.

## Reproduction and generated data

```sh
.venv/bin/python experiments/card_play_diagnostic/collect_card_play_situations.py --games 300 --workers 10 --concurrent-games 1
.venv/bin/python experiments/card_play_diagnostic/run_card_play_branches.py --per-type 48 --rollouts 40 --workers 10 --concurrent-games 96
.venv/bin/python experiments/card_play_diagnostic/summarize_card_play_diagnostic.py
```

The generated, gitignored analysis data are:

- `data/derived/card_play_situations.json`
- `data/derived/card_play_branches.json`
- `data/derived/card_play_summary.json`
