"""The tree of one search (information-set Monte Carlo tree search, experiments/tree_search/notes.md).

A node is the point reached by a sequence of moves from the searched decision, whatever hidden cards a copy was dealt:
copies with different deals share it, and each copy only chooses among the moves legal in its own game. Edges are
moves (move_options.move_key). Results are kept from the side of the player choosing at the node.

Copies of one search run at the same time, so an edge being walked counts as a loss until its copy's result arrives
(virtual loss): the next walks spread out instead of all following the same path.
"""

import math
from dataclasses import dataclass, field

VIRTUAL_LOSS = -1.0


@dataclass
class Edge:
    visits: int = 0
    total: float = 0.0  # sum of results from the choosing player's side (pending walks count as losses)
    child: "Node | None" = None

    @property
    def mean(self) -> float:
        return self.total / self.visits


@dataclass
class Node:
    who: int  # the player choosing here
    edges: dict[str, Edge] = field(default_factory=dict)

    def select(self, priors: dict[str, float], exploration: float) -> str:
        """PUCT among the moves legal in this copy (priors: key -> network probability), counting the walk as pending.
        A move never tried is valued at the mean of the tried ones (or 0)."""
        tried = [self.edges[key] for key in priors if key in self.edges and self.edges[key].visits]
        visits = sum(edge.visits for edge in tried)
        untried_value = sum(edge.total for edge in tried) / visits if visits else 0.0
        scale = exploration * math.sqrt(max(visits, 1))

        def score(key: str) -> float:
            edge = self.edges.get(key)
            edge_visits = edge.visits if edge else 0
            value = edge.mean if edge_visits else untried_value
            return value + scale * priors[key] / (1 + edge_visits)

        key = max(priors, key=score)
        edge = self.edges.setdefault(key, Edge())
        edge.visits += 1
        edge.total += VIRTUAL_LOSS
        return key

    def most_visited(self, priors: dict[str, float]) -> str:
        """The move to play: most visits, ties broken by the network's probability."""
        return max(priors, key=lambda key: (self.edges[key].visits if key in self.edges else 0, priors[key]))


def back_up(path: list[tuple[Node, str]], result_for: dict[int, float]) -> None:
    """A finished walk: each edge's pending loss is replaced by the result for its node's player."""
    for node, key in path:
        node.edges[key].total += result_for[node.who] - VIRTUAL_LOSS


def abandon(path: list[tuple[Node, str]]) -> None:
    """A failed walk: its pending visits are taken back."""
    for node, key in path:
        edge = node.edges[key]
        edge.visits -= 1
        edge.total -= VIRTUAL_LOSS
