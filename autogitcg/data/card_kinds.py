"""One kind per action card, the most specific that applies: tagged kinds first (a resonance is also an event, a
weapon also equipment), then the plain card type."""

from functools import cache

from autogitcg.data.dimbreath_tables import load_table

TAGGED_KINDS = (("arcane legends", "GCG_TAG_LEGEND"), ("transfigurations", "GCG_TAG_CARD_BLESSING"),
                ("resonances", "GCG_TAG_RESONANCE"), ("talents", "GCG_TAG_TALENT"), ("weapons", "GCG_TAG_WEAPON"),
                ("artifacts", "GCG_TAG_ARTIFACT"), ("food", "GCG_TAG_FOOD"))
TYPE_KINDS = {"GCG_CARD_ASSIST": "supports", "GCG_CARD_EVENT": "events", "GCG_CARD_MODIFY": "equipment"}
CARD_KINDS = tuple(kind for kind, _ in TAGGED_KINDS) + tuple(TYPE_KINDS.values())


@cache
def _kinds() -> dict[int, str]:
    kinds = {}
    for row in load_table("GCGCardExcelConfigData"):
        tags = row.get("tagList", [])
        tagged = next((kind for kind, tag in TAGGED_KINDS if tag in tags), None)
        kinds[row["id"]] = tagged or TYPE_KINDS.get(row.get("cardType"), "other")
    return kinds


def card_kind(definition_id: int) -> str:
    return _kinds().get(definition_id, "other")
