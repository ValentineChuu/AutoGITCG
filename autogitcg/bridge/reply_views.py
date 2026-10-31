"""Per-seat reply views for EnginePool.create_game (engine/reference/bridge/opponent_replies.mjs): which seats get the
opponent-replies input and what each may know about its opponent's cards. Open deck lists (NPC fights) give card
replies from the opponent's real remaining cards; a blind seat gets card replies only from a guess (public tournament
decks, deck_guessing/reply_guess_candidates.py), never from their real cards; a seat with None gets no replies."""

from collections.abc import Callable

from autogitcg.data.deck import Deck

OPEN = {"opponentDeck": "open"}
HIDDEN = {"opponentDeck": "hidden"}  # skills and switches only


def policy_reads_replies(policy) -> bool:
    """A network policy whose network reads replies, or a split policy (policies/split_policy.py) with one."""
    if hasattr(policy, "network"):
        return policy.network.config.reads_replies
    return any(policy_reads_replies(part) for part in (getattr(policy, "main", None), getattr(policy, "fallback", None)) if part)


def reply_views(reads_replies: tuple[bool, bool], opponent_deck_known: tuple[bool, bool], decks: tuple[Deck, Deck],
                guess_candidates: Callable[[Deck], list] | None = None) -> list[dict | None]:
    """guess_candidates(opponent deck) -> [[lineup score, card ids], ...] for blind seats (empty: no guess)."""
    views = []
    for who in (0, 1):
        if not reads_replies[who]:
            views.append(None)
        elif opponent_deck_known[who]:
            views.append(OPEN)
        else:
            candidates = guess_candidates(decks[1 - who]) if guess_candidates else []
            views.append({"opponentDeck": "guessed", "candidates": candidates} if candidates else HIDDEN)
    return views
