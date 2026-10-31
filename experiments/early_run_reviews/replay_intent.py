"""Conservative reaction-intent measures from benchmark replays with full states."""

import gzip
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from autogitcg.analysis.confidence import wilson_interval
from autogitcg.analysis.definition_kinds import character_element, skill_facts
from autogitcg.observation.enums import AURA_ELEMENT_CODES, ELEMENTS, VALID

WINDOWS = {"early": (20, 400), "middle": (820, 1200), "late": (1620, 2000)}
REACTIVE_WITH = {
    "cryo": {"pyro", "electro", "hydro"},
    "hydro": {"pyro", "electro", "cryo", "dendro", "anemo", "geo"},
    "pyro": {"cryo", "hydro", "electro", "dendro", "anemo", "geo"},
    "electro": {"pyro", "cryo", "hydro", "dendro", "anemo", "geo"},
    "anemo": {"cryo", "hydro", "pyro", "electro"},
    "geo": {"cryo", "hydro", "pyro", "electro"},
    "dendro": {"pyro", "hydro", "electro"},
}


def _elements_in_aura(aura: int) -> set[str]:
    codes = {aura & 15, (aura >> 4) & 15} - {0}
    return {ELEMENTS[AURA_ELEMENT_CODES[code]] for code in codes if code in AURA_ELEMENT_CODES}


def _reacts(element: str, aura: int) -> bool:
    return bool(_elements_in_aura(aura) & REACTIVE_WITH.get(element, set()))


def _active_character(player: dict) -> dict | None:
    return next((character for character in player.get("character", [])
                 if character["id"] == player.get("activeCharacterId")), None)


def _action(decision: dict) -> tuple[dict, int] | tuple[None, None]:
    if decision["kind"] != "action":
        return None, None
    index = decision["response"]["chosenActionIndex"]
    return decision["request"]["action"][index], index


def _rate(successes: int, count: int) -> dict:
    if count == 0:
        return {"successes": successes, "count": 0, "rate": None, "low": None, "high": None}
    low, high = wilson_interval(successes, count)
    return {"successes": successes, "count": count, "rate": successes / count,
            "low": low, "high": high}


def analyse_replays(run: Path) -> dict:
    elements, skills = character_element(), skill_facts()
    counts = defaultdict(lambda: defaultdict(float))
    for path in sorted((run / "replays").glob("update_*.jsonl.gz")):
        update = int(path.name.split("_")[1].split(".")[0])
        window = next((name for name, (low, high) in WINDOWS.items() if low <= update <= high), None)
        if window is None:
            continue
        with gzip.open(path, "rt", encoding="utf-8") as source:
            records = [json.loads(line) for line in source]
        for record in records:
            learner_seat = record["learner_seat"]
            pending_switch = {0: None, 1: None}
            for decision in record["decisions"]:
                action, chosen_index = _action(decision)
                if action is None:
                    continue
                who = decision["who"]
                controller = "learner" if who == learner_seat else record["opponent"]
                key = (window, controller)
                case = action["action"]["$case"]
                state = decision["state"]
                active = _active_character(state["player"][who])
                opponent_active = _active_character(state["player"][1 - who])
                if active is None or opponent_active is None:
                    continue
                aura = opponent_active.get("aura", 0)

                if pending_switch[who] is not None:
                    counts[key]["reaction_ready_switch_followups"] += 1
                    if case == "useSkill" and active["definitionId"] == pending_switch[who]:
                        counts[key]["reaction_ready_switch_then_skill"] += 1
                    pending_switch[who] = None

                if case == "switchActive":
                    target = action["action"]["value"].get("characterDefinitionId", 0)
                    if _reacts(elements.get(target, ""), aura):
                        counts[key]["reaction_ready_switches"] += 1
                        pending_switch[who] = target
                    counts[key]["switches"] += 1
                    continue
                if case != "useSkill":
                    continue
                skill_id = action["action"]["value"]["skillDefinitionId"]
                skill_type = skills.get(skill_id, ("other", 0))[0]
                if skill_type not in ("elemental", "burst"):
                    continue
                chosen_reactive = _reacts(elements.get(active["definitionId"], ""), aura)
                counts[key]["elemental_or_burst_skills"] += 1
                counts[key]["chosen_reactive_skills"] += int(chosen_reactive)
                alive = [character for character in state["player"][who].get("character", [])
                         if not character.get("defeated", False)]
                counts[key]["random_alive_character_expectation"] += (
                    sum(_reacts(elements.get(character["definitionId"], ""), aura) for character in alive)
                    / max(len(alive), 1))

                valid_skills = []
                for option in decision["request"]["action"]:
                    value = option["action"]
                    if option.get("validity") == VALID and value["$case"] == "useSkill":
                        option_type = skills.get(value["value"]["skillDefinitionId"], ("other", 0))[0]
                        if option_type in ("elemental", "burst"):
                            valid_skills.append(_reacts(elements.get(active["definitionId"], ""), aura))
                if valid_skills and any(valid_skills) and not all(valid_skills):
                    counts[key]["mixed_skill_choices"] += 1
                    counts[key]["mixed_chose_reactive"] += int(chosen_reactive)
                    counts[key]["mixed_random_expectation"] += sum(valid_skills) / len(valid_skills)

    result = {}
    for (window, controller), values in sorted(counts.items()):
        result.setdefault(window, {})[controller] = {
            "reactive_elemental_or_burst_skill": _rate(int(values["chosen_reactive_skills"]),
                                                         int(values["elemental_or_burst_skills"])),
            "random_alive_character_expected_rate": values["random_alive_character_expectation"] /
                                                     max(values["elemental_or_burst_skills"], 1),
            "mixed_valid_skills_chose_reactive": _rate(int(values["mixed_chose_reactive"]),
                                                          int(values["mixed_skill_choices"])),
            "mixed_random_choice_expected_rate": values["mixed_random_expectation"] / max(values["mixed_skill_choices"], 1),
            "reaction_ready_switch": _rate(int(values["reaction_ready_switches"]), int(values["switches"])),
            "reaction_ready_switch_then_skill": _rate(int(values["reaction_ready_switch_then_skill"]),
                                                         int(values["reaction_ready_switch_followups"])),
        }
    return result


def write_replay_intent(run: Path, folder: Path, output: Path) -> dict:
    result = analyse_replays(run)
    folder.mkdir(parents=True, exist_ok=True)
    figure, (skill_axis, switch_axis) = plt.subplots(1, 2, figsize=(10.5, 4))
    for controller in ("learner", "heuristic"):
        skill_axis.plot(list(WINDOWS), [result[window][controller]["reactive_elemental_or_burst_skill"]["rate"]
                                        for window in WINDOWS], marker="o", label=controller)
        switch_axis.plot(list(WINDOWS), [result[window][controller]["reaction_ready_switch"]["rate"]
                                         for window in WINDOWS], marker="o", label=controller)
    skill_axis.plot(list(WINDOWS), [result[window]["learner"]["random_alive_character_expected_rate"]
                                    for window in WINDOWS], linestyle="--", color="gray",
                    label="learner random-alive-character baseline")
    skill_axis.set(title="Elemental/Burst skills into an existing aura", ylabel="share with reaction-compatible element")
    switch_axis.set(title="Switches into a reaction-ready element", ylabel="share of voluntary switches")
    for axis in (skill_axis, switch_axis):
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(folder / "reaction_intent_from_replays.png", dpi=160)
    plt.close(figure)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result
