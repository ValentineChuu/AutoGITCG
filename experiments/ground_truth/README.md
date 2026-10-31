# The ground-truth set: what is the best move, really?

**Question:** how good are the network's moves, and which way of choosing a move (the network alone, the look-ahead,
tree search, other settings) picks the best one?

**What was done:** at sampled decisions where the network was unsure, every compared move was played out in
hundreds to thousands of paired game copies (the same deals for every move), giving the true best move and how much
each other move loses, in win-chance points (`build_ground_truth.py`). 357 positions, then 2,148 more. Every method
was then scored on the same positions (`score_methods_on_ground_truth.py`, `compare_methods_on_ground_truth.py`,
`analyze_ground_truth.py`; results in `results/`).

**Outcome:**
- Beta 10.8's own top move was the best one in only about 20% of these positions, and lost 9.7 win-chance points per
  hard decision on held-out positions (a random compared move: 12.0). The best moves often had very low probability
  under the network (median 1.8%), so there is a lot of room above it.
- The look-ahead scored by the network's *value* from the searching player's own view lost far less than one scored
  by the win output, and the deep judge (8 moves, 16 deals) lost only 1.8 points: the best player available. This set
  the judges ([docs/search.md](../../docs/search.md)) and made the deep judge the teacher for the students
  ([../lookahead_students](../lookahead_students)).
