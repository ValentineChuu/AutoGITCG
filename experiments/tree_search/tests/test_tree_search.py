"""The search tree's bookkeeping (search/search_tree.py) and move keys (search/move_options.py)."""

from experiments.tree_search.move_options import move_key
from experiments.tree_search.search_tree import Node, abandon, back_up


def test_untried_likely_moves_are_tried_before_repeating_a_loss():
    node = Node(who=0)
    priors = {"attack": 0.5, "card": 0.4, "end": 0.1}
    first = node.select(priors, exploration=1.25)
    assert first == "attack"
    back_up([(node, first)], {0: -1.0, 1: 1.0})
    assert node.select(priors, exploration=1.25) == "card"


def test_pending_walks_spread_out_and_results_replace_them():
    node = Node(who=1)
    priors = {"a": 0.5, "b": 0.5}
    first, second = node.select(priors, 1.0), node.select(priors, 1.0)
    assert {first, second} == {"a", "b"}
    back_up([(node, first)], {0: -1.0, 1: 1.0})
    abandon([(node, second)])
    assert node.edges[first].visits == 1 and node.edges[first].mean == 1.0
    assert node.edges[second].visits == 0 and node.edges[second].total == 0.0
    assert node.most_visited(priors) == first


def test_results_are_counted_from_each_nodes_player():
    ours, theirs = Node(who=0), Node(who=1)
    path = [(ours, ours.select({"x": 1.0}, 1.0)), (theirs, theirs.select({"y": 1.0}, 1.0))]
    back_up(path, {0: 1.0, 1: -1.0})
    assert ours.edges["x"].mean == 1.0 and theirs.edges["y"].mean == -1.0


def test_move_key_ignores_fields_that_differ_between_copies():
    played = {"action": {"$case": "playCard", "value": {"cardId": 7, "cardDefinitionId": 332004, "targetIds": [], "willBeEffectless": False}}}
    effectless = {"action": {"$case": "playCard", "value": {"cardId": 7, "cardDefinitionId": 332004, "targetIds": [], "willBeEffectless": True}}}
    tuned = {"action": {"$case": "elementalTuning", "value": {"removedCardId": 7}}}
    assert move_key(played) == move_key(effectless) != move_key(tuned)


def test_paired_comparison_separates_a_move_that_wins_on_the_same_deals():
    from experiments.ground_truth.move_comparison import MoveComparison
    comparison = MoveComparison(2)
    for deal in range(40):
        luck = 1.0 if deal % 2 else -1.0  # the deal decides most games: both moves win or lose together
        comparison.add(0, deal, luck)
        comparison.add(1, deal, 1.0 if deal % 4 == 0 else luck)  # move 1 also wins a quarter of the bad deals
    assert comparison.best() == 1
    assert comparison.separated(3.0)
    assert comparison.summary()[0]["difference_to_best"] == -0.5


def test_a_position_lost_whatever_is_played_counts_as_identical():
    from experiments.ground_truth.move_comparison import MoveComparison
    comparison = MoveComparison(3)
    for deal in range(16):
        for move in range(3):
            comparison.add(move, deal, -1.0)
    assert comparison.all_identical() and not comparison.separated(3.0)
    comparison.add(2, 16, 1.0)
    comparison.add(0, 16, -1.0)
    comparison.add(1, 16, -1.0)
    assert not comparison.all_identical()
