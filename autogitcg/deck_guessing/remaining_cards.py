"""From guessed whole decks to the expected counts of the cards still hidden (hand plus deck)."""

from collections import Counter


def expected_remaining(guesses: list[tuple[float, Counter]], revealed: Counter, hidden_count: int) -> dict[int, float]:
    """guesses: (weight, deck card counts). Each guess minus the revealed cards, averaged by weight and scaled to
    `hidden_count` cards (a guess that doesn't hold a revealed card gets no extra credit for it)."""
    total_weight = sum(weight for weight, _ in guesses)
    if not guesses or total_weight <= 0 or hidden_count <= 0:
        return {}
    expected: Counter = Counter()
    for weight, deck in guesses:
        for card, count in (deck - revealed).items():
            expected[card] += weight * count / total_weight
    size = sum(expected.values())
    return {card: count * hidden_count / size for card, count in expected.items()} if size else {}
