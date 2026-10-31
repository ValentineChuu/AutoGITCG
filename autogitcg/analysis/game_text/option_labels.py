"""Readable labels for a decision's choices, in the same order as the network's option probabilities.

action: valid actions in request order (see encode_actions); chooseActive / selectCard: candidates in order;
rerollDice: one per die; switchHands: one per own hand card.
"""

from functools import cache

from autogitcg.analysis.game_text.english_names import card_name, character_name, skill_name
from autogitcg.data.piovium_catalog import load_catalog

ELEMENT_BY_DICE = {1: "cryo", 2: "hydro", 3: "pyro", 4: "electro", 5: "anemo", 6: "geo", 7: "dendro", 8: "omni"}

COST_NAMES = {0: "any", 8: "same", 9: "energy", 10: "legend"}
SKILL_TYPE_NAMES = {"normal": "Normal Attack", "elemental": "Skill", "burst": "Burst", "technique": "Technique"}


@cache
def _skill_types() -> dict[int, str]:
    catalog = load_catalog()
    return {skill["id"]: skill["skillType"] for owner in catalog["characters"] + catalog["entities"] for skill in owner["skills"]}


def cost_text(requirements: list[dict]) -> str:
    parts = []
    for requirement in requirements:
        if requirement.get("count", 0) <= 0:
            continue
        name = COST_NAMES.get(requirement["type"]) or ELEMENT_BY_DICE[requirement["type"]].capitalize()
        parts.append(f"{requirement['count']} {name}")
    return " + ".join(parts) if parts else "free"


def _character_definitions(state: dict) -> dict[int, int]:
    return {character["id"]: character["definitionId"] for player in state["player"] for character in player.get("character", [])}


def _target_text(target_ids: list[int], state: dict) -> str:
    definitions = _character_definitions(state)
    names = [character_name(definitions[target]) for target in target_ids if target in definitions]
    return f" on {', '.join(names)}" if names else ""


def _action_label(action: dict, state: dict, who: int) -> str:
    kind, value = action["action"]["$case"], action["action"]["value"]
    hand = {card["id"]: card["definitionId"] for card in state["player"][who].get("handCard", [])}
    match kind:
        case "useSkill":
            skill_id = value["skillDefinitionId"]
            skill_type = SKILL_TYPE_NAMES.get(_skill_types().get(skill_id), "Skill")
            return f"{skill_type}: {skill_name(skill_id)}"
        case "playCard":
            return f"Play {card_name(value['cardDefinitionId'])}{_target_text(value.get('targetIds', []), state)}"
        case "switchActive":
            return f"Switch to {character_name(value['characterDefinitionId'])}"
        case "elementalTuning":
            return f"Tune away {card_name(hand.get(value['removedCardId'], 0))}"
        case "declareEnd":
            return "End round"
    return kind


def option_labels(kind: str, request: dict, state: dict, who: int) -> list[str]:
    if kind == "action":
        return [f"{_action_label(action, state, who)}  ({cost_text(action.get('requiredCost', []))})"
                for action in request["action"] if action.get("validity", 0) == 0]
    if kind == "chooseActive":
        definitions = _character_definitions(state)
        return [f"Choose {character_name(definitions.get(candidate, 0))}" for candidate in request["candidateIds"]]
    if kind == "selectCard":
        return [f"Pick {card_name(definition_id)}" for definition_id in request["candidateDefinitionIds"]]
    if kind == "rerollDice":
        return [f"Reroll die {position + 1} ({ELEMENT_BY_DICE[die]})" for position, die in enumerate(state["player"][who]["dice"])]
    if kind == "switchHands":
        return [f"Swap {card_name(card['definitionId'])}" for card in state["player"][who].get("handCard", [])]
    return []


def is_categorical(kind: str) -> bool:
    """One choice among options (probabilities sum to 1); the others are independent yes/no per die or card."""
    return kind in ("action", "chooseActive", "selectCard")
