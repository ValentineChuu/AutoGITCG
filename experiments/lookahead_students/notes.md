# Look-ahead as teacher (Chudledum v3, next attempt)

Written 2026-10-17 after `gamma 5` was stopped. Built the same day (`experiments/lookahead_students/`):
`experiments/lookahead_students/record_teacher_labels.py` (step 1, several recording processes writing label shards; the game loop is
`search/lookahead_table.py`) and `experiments/lookahead_students/train_student.py` (step 2). Step 3 uses `scripts/evaluation/test_lookahead.py
--network <student> --confident-probability 0` (never searches: plain play). The starting student is gamma 5 at update 180
(gamma 5.3.30, between milestones, so not in the release; `scripts/training/freeze_combat_network.py --weights` makes it).

## Why

- PPO has stalled at beta 10.8's level twice. The small network (`beta 10` onward, runs J/K/L) and the larger one distilled from beta 10.8 (`gamma 5`) both sit at 50-53% against beta 10.8. Capacity wasn't the limit; the training signal is.
- Look-ahead with beta 10.8 (24 decisions ahead, 4 candidates) beats plain beta 10.8 74.8% of the time. Its choices are a much stronger signal than PPO's advantage estimates.
- So: train the network to play, without look-ahead, the moves look-ahead picks. Then look-ahead on top of the improved network is a stronger teacher again, and the loop repeats (expert iteration, the AlphaZero idea). Learned throughout; nothing is hand-designed.

## One iteration

1. **Generate labels.** Play games with look-ahead on the searching network, against plain networks (current and past) on the training matchups (`MatchupSampler`, so the locked test set stays out). At every searched decision, store the observation and the move look-ahead chose. Also store each game's result for the value head.
2. **Train the student** with supervised learning only:
   - cross-entropy toward the chosen move at searched decisions;
   - a pull toward its own starting distribution at the other decisions (it already plays those confidently), so it doesn't drift where there is no label;
   - value toward the game result.
3. **Test** the student without look-ahead on the locked tournament matchups against beta 10.8 (the usual 400 games). Success: clearly above 52%. Then make it the searcher for the next iteration.

## Choices

- **Student:** the larger network from `gamma 5` (update 180 weights; plays like beta 10.8) or beta 10.8 itself. Start with the larger one: it has room to absorb more than beta 10.8 knows.
- **Searcher for iteration 1:** beta 10.8 (small), because search runs the network for every copy decision and the small one is about 3x cheaper. Later iterations search with the student.
- **Label budget:** about 10 searched decisions per game. 400 games took 55 minutes on 9 desktop cores (CPU only), so about 4,000 labels an hour locally. That's too slow for the desktop; generation belongs on a cloud machine with many cores and a GPU. Measure labels per hour there before choosing the size of iteration 1 (aim: 50,000 or more labels).
- **Which decisions count as searched:** the current rule (top move under 90% likely). Fewer than 1 in 3 searched choices change the move, so most labels confirm the network; that's fine and anchors it.

## Risks

- The student may fail to reproduce what look-ahead finds, because part of its edge comes from seeing re-dealt futures. The test in step 3 shows this after one iteration, before much is spent.
- Labels from one searcher can be narrow. Vary decks and opponents, and keep the pull toward the starting policy.
- Blind PvP: generate labels only from games with known deck lists until the blind guess is tested (`blind_guess` run).
