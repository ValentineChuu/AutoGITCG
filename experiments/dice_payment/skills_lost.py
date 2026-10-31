"""For one answered card play or tuning: the active character's skills affordable after the payment the bot made, and
after the best other legal payment. A skill only the other payment keeps is "lost to the payment". The card's own effect
(dice it adds, switching) is ignored, so this is a first estimate."""

from experiments.dice_payment.affordability import can_pay, payments, remaining
from autogitcg.observation.enums import OMNI_DICE


def affordable_skills(request_actions: list[dict], dice: list[int], energy: int) -> set[int]:
    return {action["action"]["value"]["skillDefinitionId"] for action in request_actions
            if action["action"]["$case"] == "useSkill" and can_pay(action["requiredCost"], dice, energy)}


def skills_lost(request_actions: list[dict], chosen: dict, used_dice: list[int], dice: list[int], energy: int) -> set[int]:
    """Skills an other payment of the chosen card play / tuning keeps affordable on top of everything this one kept (the
    largest such gain), or nothing when no payment beats this one."""
    case = chosen["action"]["$case"]
    if case == "playCard":
        after = lambda paid: remaining(dice, paid)
        options = payments(chosen["requiredCost"], dice)
    elif case == "elementalTuning":
        target = chosen["action"]["value"]["targetDice"]
        after = lambda paid: remaining(dice, paid) + [target]
        options = [(die,) for die in set(dice) if die not in (target, OMNI_DICE)]
    else:
        return set()
    kept = affordable_skills(request_actions, after(used_dice), energy)
    gains = [affordable_skills(request_actions, after(option), energy) - kept for option in options]
    better = [gain for gain, option in zip(gains, options) if gain and kept <= affordable_skills(request_actions, after(option), energy)]
    return max(better, key=len, default=set())
