"""The search for one action decision: its tree, and the walks (game copies) that grow it (experiments/tree_search/notes.md).

Each walk is one copy of the game with the hidden cards dealt again. It starts with a root move chosen in the tree,
keeps choosing in the tree while its decisions are tree nodes, and once it leaves the tree (a new node, or a player
the tree doesn't branch for) it is played out by the network and scored. The answer is the root's most visited move.
"""

from collections import Counter
from dataclasses import dataclass, field

from autogitcg.bridge.messages import Decision
from autogitcg.observation.match_recorder import MatchRecorder
from experiments.tree_search.move_options import MoveOption
from experiments.tree_search.search_tree import Node, abandon, back_up
from experiments.tree_search.tree_search_settings import TreeSearchSettings


def priors_of(options: dict[str, MoveOption]) -> dict[str, float]:
    return {key: option.probability for key, option in options.items()}


@dataclass
class Walk:
    search: "TreeSearch"
    recorder: MatchRecorder  # the real game's trackers, then this copy's decisions
    path: list[tuple[Node, str]]  # the tree edges taken so far
    in_tree: bool = True
    tree_moves: int = 1  # moves chosen in the tree (the root move included)

    def tree_move(self, decision: Decision, options: dict[str, MoveOption], settings: TreeSearchSettings) -> str | None:
        """At one of the copy's action decisions: the move chosen in the tree, or None once the walk has left it
        (the copy is then played by the network)."""
        if not self.in_tree:
            return None
        node, key = self.path[-1]
        edge = node.edges[key]
        branches = decision.who == self.search.decision.who or settings.branch_opponent
        if not branches or (edge.child is not None and edge.child.who != decision.who):
            self.in_tree = False
            return None
        if edge.child is None:
            edge.child = Node(decision.who)  # one new node per walk, valued by playing the copy out
            self.in_tree = False
            return None
        chosen = edge.child.select(priors_of(options), settings.exploration)
        self.path.append((edge.child, chosen))
        self.tree_moves += 1
        return chosen


@dataclass
class TreeSearch:
    decision: Decision
    recorder: MatchRecorder  # the real game's
    options: dict[str, MoveOption]  # the root's moves
    settings: TreeSearchSettings
    root: Node = None
    launched: int = 0
    open_walks: int = 0
    counts: Counter = field(default_factory=Counter)  # finished / failed walks, tree moves

    def __post_init__(self):
        self.root = Node(self.decision.who)

    def wants_walk(self) -> bool:
        return self.launched < self.settings.iterations and self.open_walks < self.settings.parallel

    def start_walk(self, opponent_cards: list[int] | None = None, opponent_hand_ids: list[int] = ()) -> Walk:
        """A new walk: its root move is chosen now. opponent_cards: as in MatchRecorder.copy_for_redeal."""
        key = self.root.select(priors_of(self.options), self.settings.exploration)
        recorder = self.recorder.copy_for_redeal(self.decision, opponent_cards, opponent_hand_ids)
        recorder.on_response(self.decision, self.options[key].response)
        self.launched += 1
        self.open_walks += 1
        return Walk(self, recorder, [(self.root, key)])

    def finish_walk(self, walk: Walk, searcher_result: float | None) -> None:
        """searcher_result: the copy's score from the searching player's side (+1 win, -1 loss), None if it failed."""
        self.open_walks -= 1
        if searcher_result is None:
            abandon(walk.path)
            self.counts["walks failed"] += 1
            return
        back_up(walk.path, {self.decision.who: searcher_result, 1 - self.decision.who: -searcher_result})
        self.counts["walks finished"] += 1
        self.counts["tree moves"] += walk.tree_moves

    @property
    def finished(self) -> bool:
        return self.launched >= self.settings.iterations and self.open_walks == 0

    def chosen(self) -> MoveOption:
        return self.options[self.root.most_visited(priors_of(self.options))]

    def root_visits(self) -> dict[str, tuple[int, float]]:
        """Per root move tried: (visits, mean result for the searching player)."""
        return {key: (edge.visits, edge.mean) for key, edge in self.root.edges.items() if edge.visits}
