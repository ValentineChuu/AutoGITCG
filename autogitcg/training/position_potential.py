"""A position score for reward shaping: my team's remaining health share minus the opponent's (defeated characters
count 0), from the public board. Scaled by the shaping weight in Trajectory.finish.

Shaping with the change in this score after every move (and the score counted as 0 once the game ends) is
potential-based: over a whole game the extra rewards cancel, so the best strategy stays exactly the same, but each
move gets immediate feedback on damage dealt and taken instead of only the final result (Ng, Harada and Russell 1999).
"""


def health_share(player: dict) -> float:
    characters = player.get("character", [])
    max_total = sum(character.get("maxHealth", 0) for character in characters)
    alive_health = sum(0 if character.get("defeated", False) else character.get("health", 0) for character in characters)
    return alive_health / max_total if max_total else 0.0


def position_potential(state: dict, who: int) -> float:
    return health_share(state["player"][who]) - health_share(state["player"][1 - who])
