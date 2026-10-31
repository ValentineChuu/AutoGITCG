# Tree search at play time

**Question:** the look-ahead only varies the first move and then lets the network play the copies out. Would a tree
search that branches at every decision play better?

**What was done:** a Monte-Carlo-style tree search over game copies (`tree_search*.py`, `search_tree.py`,
`move_options.py`), both searching only my own turn and searching both players, tested in blind games against plain
beta 10.8 on the locked tournament test matchups (`play_tree_search.py`).

**Outcome:** no better than the look-ahead. With 64 walks per decision it won 62.0-62.8% against plain beta 10.8; the
look-ahead reached 64-65% on the same games at a fraction of the cost. Games are too random for a few dozen played-out
walks to tell close moves apart. This led to the ground-truth set ([../ground_truth](../ground_truth)), which showed the
look-ahead scored by value from one's own view was the better direction ([docs/search.md](../../docs/search.md)).

`notes.md` is the original plan and log of the experiment.
