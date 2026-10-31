# Experiments

Ideas that were tried and dropped, and one-off investigations. None of this code is used by the bot or its training;
each folder holds its own code, scripts, tests and a README with the question, what was done and the outcome. The
training attempts themselves (every network family) are in [docs/training.md](../docs/training.md).

| Folder | Question | Outcome |
|---|---|---|
| [tree_search](tree_search) | Does a tree search that branches at every decision beat the look-ahead? | No: 62% vs plain beta 10.8, the look-ahead 64-65% for less |
| [ground_truth](ground_truth) | What is the best move, really, and which method finds it? | Beta 10.8's top move is best only 20% of the time; the deep judge loses least; set the judges |
| [lookahead_students](lookahead_students) | Can a network learn to play like the search? | No student beat its source network: most labels are close calls |
| [privileged_critic](privileged_critic) | Does a critic that sees the opponent's hand predict better? | Barely: the uncertainty is in future draws and rolls |
| [card_play_diagnostic](card_play_diagnostic) | Is tuning cards away a mistake? | Mostly not: it depends on the card and situation |
| [dice_payment](dice_payment) | Does automatic dice payment cost skills? | Never, in 4,604 payments |
| [held_bursts](held_bursts) | Why does the network hold ready bursts? | It picks other moves while the burst is available; replies made it worse |
| [early_run_reviews](early_run_reviews) | What did the first runs learn? | Reviews of alpha 1 and alpha 2 that shaped the later runs |

Run the experiments' tests with the rest: `pytest tests experiments`.
