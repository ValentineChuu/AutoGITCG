"""A simple scripted player: the stable baseline the learned policy is measured against.

Rules, in priority order: switch away from an active character about to fall; play cards while dice remain
for a skill afterwards; Burst > Elemental Skill > Normal Attack; tune a card when no skill is affordable;
otherwise end the round. Rerolls keep Omni dice and dice matching an alive character's element; keeps the
opening hand; picks the healthiest character when choosing an active one.
"""

import random

from autogitcg.bridge.messages import Decision
from autogitcg.data.piovium_catalog import load_catalog
from autogitcg.observation.enums import ELEMENTS, OMNI_DICE, VALID

SKILL_PRIORITY = {"burst": 30, "elemental": 20, "normal": 10, "technique": 5}
CARD_PRIORITY = 25
SWITCH_PRIORITY = 40
TUNING_PRIORITY = 1
LOW_HEALTH = 3
DICE_KEPT_FOR_A_SKILL = 3


class HeuristicPolicy:
    def __init__(self):
        catalog = load_catalog()
        self.skill_type = {skill["id"]: skill["skillType"] for character in catalog["characters"] for skill in character["skills"]}
        self.element_dice = {
            character["id"]: next((ELEMENTS.index(tag) + 1 for tag in character["tags"] if tag in ELEMENTS), None)
            for character in catalog["characters"]
        }

    def respond(self, decision: Decision, rng: random.Random) -> dict:
        me = decision.state["player"][decision.who]
        match decision.kind:
            case "action":
                return self._action(decision.request["action"], me, rng)
            case "chooseActive":
                health = {character["id"]: character.get("health", 0) for character in me.get("character", [])}
                return {"activeCharacterId": max(decision.request["candidateIds"], key=lambda candidate: health.get(candidate, 0))}
            case "selectCard":
                return {"selectedDefinitionId": rng.choice(decision.request["candidateDefinitionIds"])}
            case "rerollDice":
                wanted = {self.element_dice.get(character["definitionId"]) for character in me.get("character", [])
                          if not character.get("defeated", False)}
                return {"diceToReroll": [die for die in me.get("dice", []) if die != OMNI_DICE and die not in wanted]}
            case "switchHands":
                return {"removedHandIds": []}
        raise ValueError(f"unknown decision kind: {decision.kind}")

    def _action(self, actions: list[dict], me: dict, rng: random.Random) -> dict:
        characters = {character["id"]: character for character in me.get("character", [])}
        active_health = characters.get(me.get("activeCharacterId"), {}).get("health", 0)
        dice_count = len(me.get("dice", []))
        valid = [(index, action) for index, action in enumerate(actions) if action.get("validity", VALID) == VALID]
        skill_available = any(action["action"]["$case"] == "useSkill" for _, action in valid)

        def priority(action: dict) -> float:
            kind, value = action["action"]["$case"], action["action"]["value"]
            match kind:
                case "useSkill":
                    return SKILL_PRIORITY.get(self.skill_type.get(value["skillDefinitionId"]), 0)
                case "playCard":
                    if value.get("willBeEffectless", False):
                        return -1
                    dice_left = dice_count - len(action.get("autoSelectedDice", []))
                    return CARD_PRIORITY if dice_left >= DICE_KEPT_FOR_A_SKILL or not skill_available else -1
                case "switchActive":
                    target_health = characters.get(value["characterId"], {}).get("health", 0)
                    return SWITCH_PRIORITY if active_health <= LOW_HEALTH and target_health > active_health + 2 else -2
                case "elementalTuning":
                    return TUNING_PRIORITY if not skill_available else -1
            return 0  # declareEnd

        best = max(priority(action) for _, action in valid)
        index, action = rng.choice([(index, action) for index, action in valid if priority(action) == best])
        return {"chosenActionIndex": index, "usedDice": action["autoSelectedDice"]}
