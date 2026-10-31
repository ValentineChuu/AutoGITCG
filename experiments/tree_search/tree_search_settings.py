"""Settings for tree search at action decisions (design: experiments/tree_search/notes.md)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TreeSearchSettings:
    iterations: int = 64  # game copies per searched decision, each one walk down the tree plus a played-out ending
    parallel: int = 8  # copies of one search running at once (the tree is shared, so more = less informed walks)
    exploration: float = 1.25  # weight of the network's preference over a move's results so far (PUCT constant)
    branch_opponent: bool = False  # False: the tree only branches over our own moves until the opponent acts (our
    # turn); True: the opponent's moves branch too (both players, until the copy leaves the tree)
    confident_probability: float = 0.9  # no search when the network's top move is at least this likely
    leaf_horizon_rounds: int = 0  # 0: a copy that left the tree is played to the end of the game and scored by its
    # result; > 0: it is scored by the network's win chance once it reaches the round (searched round + this)
