"""Numeric codes used by the engine's exposed state (from @gi-tcg/typings)."""

# DiceType: 1-7 elements, 8 omni.
DICE_TYPE_COUNT = 9
OMNI_DICE = 8
ELEMENT_DICE = range(1, 8)

# DiceRequirementType keys used in costs: 0 void (any), 1-7 elements, 8 aligned, 9 energy, 10 legend.
COST_TYPE_COUNT = 11

# Element order shared by auras, dice and character tags.
ELEMENTS = ["cryo", "hydro", "pyro", "electro", "anemo", "geo", "dendro"]

# PhaseType
PHASE_COUNT = 6

# EntityType in the exposed state.
ENTITY_TYPE_STATUS = 2
ENTITY_TYPE_COMBAT_STATUS = 3
ENTITY_TYPE_SUMMON = 4
ENTITY_TYPE_SUPPORT = 5
ENTITY_TYPE_EQUIPMENT = 6

# EquipmentType: 0 other (talent), 1 weapon, 2 artifact, 3 technique.
EQUIPMENT_TYPE_COUNT = 4

# Character tag bits.
CHARACTER_TAG_BITS = [1, 2, 4, 8, 16]  # shield, barrier, skill disabled, Nightsoul's Blessing, Bond of Life
# Card/entity tag bits.
ENTITY_TAG_BITS = [1, 2]  # abyss, conductive

# Aura: low nibble is the first element, high nibble a second one (only Cryo+Dendro coexist).
AURA_ELEMENT_CODES = {1: 0, 2: 1, 3: 2, 4: 3, 7: 6}

# Decision kinds ($case of the RPC request).
DECISION_KINDS = ["action", "chooseActive", "rerollDice", "switchHands", "selectCard"]

# Action kinds ($case of an Action).
ACTION_KINDS = ["useSkill", "playCard", "switchActive", "elementalTuning", "declareEnd"]

VALID = 0  # ActionValidity.VALID

# ReactionType codes reported by the engine bridge (@gi-tcg/typings).
REACTION_NAMES = {
    101: "melt", 102: "vaporize", 103: "overloaded", 104: "superconduct", 105: "electro_charged", 106: "frozen",
    107: "swirl_cryo", 108: "swirl_hydro", 109: "swirl_pyro", 110: "swirl_electro",
    111: "crystallize_cryo", 112: "crystallize_hydro", 113: "crystallize_pyro", 114: "crystallize_electro",
    115: "burning", 116: "bloom", 117: "quicken",
    3001: "lunar_electro_charged", 3002: "lunar_bloom", 3003: "lunar_crystallize_hydro",
}
