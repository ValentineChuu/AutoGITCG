"""What each action would do right now, from the engine's action preview (the same preview the game shows a person
hovering a move). Appended to every action token, so the network sees a move's immediate result instead of having
to work it out: HP, defeat, energy and aura changes per character, reactions, who ends up active, and entities
created, removed, drawn or used up per side.

Actions without a preview (games created without previews, choose-active and select-card tokens) get zeros,
including the "has preview" flag, so networks trained before previews existed see exactly their old input.
Positions follow the board encoding: the deciding player's characters first, then the opponent's.
"""

from autogitcg.observation.encode_board import CHARACTER_COUNT, CHARACTERS_PER_SIDE
from autogitcg.observation.enums import REACTION_NAMES

REACTION_CODES = sorted(REACTION_NAMES)
BOARD_AREAS = {1, 2, 3, 4}  # EntityArea: character, combat status, summon, support
HAND_AREA = 5
PER_CHARACTER = ("health", "defeated", "energy", "aura")
PREVIEW_WIDTH = (len(PER_CHARACTER) * CHARACTER_COUNT + len(REACTION_CODES) + CHARACTER_COUNT
                 + 2 * 4 + 2 + 1)


def _characters_by_id(state: dict, who: int) -> dict[int, tuple[int, dict]]:
    """character id -> (board position, its current state)."""
    found = {}
    for side_offset, player_index in enumerate((who, 1 - who)):
        for slot, character in enumerate(state["player"][player_index].get("character", [])[:CHARACTERS_PER_SIDE]):
            found[character["id"]] = (side_offset * CHARACTERS_PER_SIDE + slot, character)
    return found


def _side_by_entity_id(state: dict, who: int) -> dict[int, int]:
    """Board entity id -> 0 (the deciding player's) or 1 (the opponent's)."""
    sides = {}
    for side_offset, player_index in enumerate((who, 1 - who)):
        player = state["player"][player_index]
        for character in player.get("character", []):
            for entity in character.get("entity", []):
                sides[entity["id"]] = side_offset
        for area in ("combatStatus", "summon", "support"):
            for entity in player.get(area, []):
                sides[entity["id"]] = side_offset
    return sides


def preview_features(preview: list | None, state: dict, who: int) -> list[float]:
    if preview is None:
        return [0.0] * PREVIEW_WIDTH
    characters = _characters_by_id(state, who)
    entity_sides = _side_by_entity_id(state, who)
    health, energy, aura = {}, {}, {}  # position -> value after the action
    defeated, reactions, new_active = set(), set(), {}
    created, removed, drawn, used = [0, 0], [0, 0], [0, 0], [0, 0]
    for entry in preview:
        mutation = entry.get("mutation", entry)
        kind, value = mutation.get("$case"), mutation.get("value", {})
        if kind in ("damage", "applyAura"):
            if value.get("reactionType", 0) in REACTION_NAMES:
                reactions.add(value["reactionType"])
            if value.get("targetId") in characters:
                position = characters[value["targetId"]][0]
                aura[position] = value.get("newAura", 0)
                if value.get("causeDefeated"):
                    defeated.add(position)
        elif kind == "modifyEntityVar":
            entity_id, name, number = value.get("entityId"), value.get("variableName"), value.get("variableValue", 0)
            if entity_id in characters:
                position = characters[entity_id][0]
                if name == "health":
                    health[position] = number
                elif name == "energy":
                    energy[position] = number
                elif name == "aura":
                    aura[position] = number
                elif name == "alive" and number == 0:
                    defeated.add(position)
            elif entity_id in entity_sides and number < value.get("oldValue", number):
                used[entity_sides[entity_id]] += 1
        elif kind == "switchActive":
            if value.get("characterId") in characters:
                position = characters[value["characterId"]][0]
                new_active[position // CHARACTERS_PER_SIDE] = position
        elif kind in ("createEntity", "removeEntity", "moveEntity"):
            side = (value.get("toWho", value.get("who", who)) != who) if kind == "moveEntity" else (value.get("who", who) != who)
            area = value.get("toWhere") if kind == "moveEntity" else value.get("where")
            if area == HAND_AREA and kind != "removeEntity":
                drawn[side] += 1
            elif area in BOARD_AREAS:
                (removed if kind == "removeEntity" else created)[side] += 1

    current = {position: character for position, character in characters.values()}
    features = []
    for position in range(CHARACTER_COUNT):
        before = current.get(position, {})
        features += [(health.get(position, before.get("health", 0)) - before.get("health", 0)) / 10,
                     float(position in defeated),
                     (energy.get(position, before.get("energy", 0)) - before.get("energy", 0)) / 3,
                     float(aura.get(position, before.get("aura", 0)) != before.get("aura", 0))]
    features += [float(code in reactions) for code in REACTION_CODES]
    features += [float(new_active.get(position // CHARACTERS_PER_SIDE) == position) for position in range(CHARACTER_COUNT)]
    for counts in (created, removed, drawn, used):
        features += [min(count, 6) / 3 for count in counts]
    health_change = {position: health.get(position, character.get("health", 0)) - character.get("health", 0)
                     for position, character in current.items()}
    damage_dealt = -sum(min(0, change) for position, change in health_change.items() if position >= CHARACTERS_PER_SIDE)
    healed = sum(max(0, change) for position, change in health_change.items() if position < CHARACTERS_PER_SIDE)
    features += [damage_dealt / 10, healed / 10, 1.0]
    return features
