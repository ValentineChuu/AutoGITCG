"""A decision's board as short text lines from the deciding player's view (opponent hand cards stay hidden)."""

from autogitcg.analysis.game_text.english_names import card_name, character_name
from autogitcg.observation.enums import AURA_ELEMENT_CODES, ELEMENTS, ENTITY_TYPE_EQUIPMENT

PHASE_NAMES = {0: "init hands", 1: "init actives", 2: "roll", 3: "action", 4: "end", 5: "game end"}


def _aura_text(aura: int) -> str:
    elements = [ELEMENTS[AURA_ELEMENT_CODES[code]] for code in (aura & 0xF, aura >> 4) if code in AURA_ELEMENT_CODES]
    return f" [{'+'.join(elements)}]" if elements else ""


def _entity_text(entity: dict, counters: dict[int, dict[str, int]]) -> str:
    values = counters.get(entity["id"], {})
    shown = [f"{value}" for name, value in values.items() if name in ("usage", "shield", "count", "stack")]
    return card_name(entity["definitionId"]) + (f" ({', '.join(shown)})" if shown else "")


def character_lines(player: dict, counters: dict[int, dict[str, int]]) -> list[str]:
    lines = []
    for character in player.get("character", []):
        marker = ">" if character["id"] == player.get("activeCharacterId") else " "
        if character.get("defeated"):
            lines.append(f"{marker} {character_name(character['definitionId'])}: defeated")
            continue
        line = (f"{marker} {character_name(character['definitionId'])}  {character.get('health', 0)}/{character.get('maxHealth', 0)} HP"
                f"  {character.get('energy', 0)}/{character.get('maxEnergy', 0)} energy{_aura_text(character.get('aura', 0))}")
        attached = [_entity_text(entity, counters) for entity in character.get("entity", [])]
        equipment = [text for entity, text in zip(character.get("entity", []), attached) if entity.get("type") == ENTITY_TYPE_EQUIPMENT]
        statuses = [text for entity, text in zip(character.get("entity", []), attached) if entity.get("type") != ENTITY_TYPE_EQUIPMENT]
        lines.append(line)
        if equipment or statuses:
            lines.append("      " + "; ".join(equipment + statuses))
    return lines


def area_lines(player: dict, counters: dict[int, dict[str, int]]) -> list[str]:
    lines = []
    for label, key in [("Combat statuses", "combatStatus"), ("Summons", "summon"), ("Supports", "support")]:
        entities = player.get(key, [])
        if entities:
            lines.append(f"  {label}: " + "; ".join(_entity_text(entity, counters) for entity in entities))
    return lines


def own_hand_names(player: dict) -> list[str]:
    return [card_name(card["definitionId"]) for card in player.get("handCard", [])]


def heading(state: dict) -> str:
    return f"Round {state.get('roundNumber', 0)}, {PHASE_NAMES.get(state.get('phase', 0), 'phase')} phase"


def situation_line(state: dict, who: int, counters: dict[int, dict[str, int]]) -> str:
    """One line for a caption: round, own active character, own summons, dice and hand size."""
    me = state["player"][who]
    parts = []
    active = [character for character in me.get("character", []) if character["id"] == me.get("activeCharacterId")]
    if active:
        parts.append(f"{character_name(active[0]['definitionId'])} active ({active[0].get('health', 0)} HP)")
    summons = [_entity_text(entity, counters) for entity in me.get("summon", [])]
    if summons:
        parts.append(f"{', '.join(summons)} on the field")
    parts.append(f"{len(me.get('dice', []))} dice, {len(me.get('handCard', []))} cards in hand")
    return f"{heading(state)}: " + "; ".join(parts)
