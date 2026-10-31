"""Whether an action option is an Elemental Burst: a skill whose cost includes energy."""

ENERGY_COST = 9  # DiceType.ENERGY in a skill's required cost


def is_burst(option: dict) -> bool:
    return option["action"]["$case"] == "useSkill" and any(cost["type"] == ENERGY_COST for cost in option.get("requiredCost", []))
