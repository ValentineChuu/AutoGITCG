"""Every NPC stage setup against a deck Chudledee builds for it from every obtainable card: which fights Chudledum still
can't win reliably even with the best deck (gaps in its play, not in the collection).

Stages with the same kind, deck and NPC equipment are checked once. Results go to <output>.jsonl, one line per stage
setup; rerunning skips setups already there.
    python scripts/evaluation/check_stages_with_full_pool.py --games 16 --output runs/stage_check/full_pool_v2
"""

import argparse
import collections
import json
from pathlib import Path

import torch

from autogitcg.analysis.builder_deck_check.check_players import NetworkPlayer
from autogitcg.analysis.stage_check.stage_games import play_stage_games
from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.deck_builder.builder_policy import BuilderPolicy
from autogitcg.deck_builder.builder_setup import BuilderSetup, load_builder_checkpoint
from autogitcg.deck_builder.building_task import STAGE_MODE, BuildingTask
from autogitcg.deck_building.deck_validation import buildable_action_cards, buildable_characters
from autogitcg.observation.observation_encoder import ObservationEncoder
from autogitcg.progression.stage_catalog import build_stage_catalog
from autogitcg.training.collection_pool import Collection

BUILD_BATCH = 32


def _setup_key(stage) -> str:
    equipment = sorted((item.character_index, item.definition_id) for item in stage.enemy_equipment)
    return json.dumps([stage.kind, stage.enemy_deck.characters, sorted(stage.enemy_deck.cards), equipment])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--combat", type=Path, default=Path("models/beta-10.8.pt"))
    parser.add_argument("--builder", type=Path, default=Path("models/chudledee-v3.4.pt"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--limit", type=int, help="check at most this many stage setups (for a quick test)")
    arguments = parser.parse_args()
    output = arguments.output.with_suffix(".jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)

    setups = {}  # setup key -> stages sharing it
    for stage in build_stage_catalog()[0]:
        setups.setdefault(_setup_key(stage), []).append(stage)
    done = {json.loads(line)["setup"] for line in output.open()} if output.exists() else set()
    todo = [key for key in setups if key not in done][:arguments.limit]
    print(f"{len(setups)} stage setups, {len(done)} done, {len(todo)} to check", flush=True)

    setup = BuilderSetup()
    device = torch.device("cpu")
    builder = BuilderPolicy(setup.network_from(load_builder_checkpoint(arguments.builder)).to(device), setup.encoder, device, greedy=True)
    everything = Collection(tuple(sorted(buildable_characters())), collections.Counter({card: 2 for card in buildable_action_cards()}), 0)
    encoder = ObservationEncoder()
    player = NetworkPlayer(arguments.combat, encoder)
    played = 0
    with EnginePool(arguments.workers) as pool, output.open("a") as file:
        for first in range(0, len(todo), BUILD_BATCH):
            keys = todo[first:first + BUILD_BATCH]
            stages = [setups[key][0] for key in keys]
            decks = [built.deck for built in builder.build([BuildingTask(everything, STAGE_MODE, stage, "everything") for stage in stages])]
            results = collections.defaultdict(list)

            def on_result(stage, won, rounds):
                nonlocal played
                results[stage.level_id].append((won, rounds))
                played += 1

            play_stage_games([(stage, deck) for stage, deck in zip(stages, decks) for _ in range(arguments.games)],
                             player, encoder, pool, arguments.seed + first, on_result)
            for key, stage, deck in zip(keys, stages, decks):
                games = results[stage.level_id]
                file.write(json.dumps({"setup": key, "kind": stage.kind, "level_ids": [s.level_id for s in setups[key]],
                                       "name": stage.name, "deck": {"characters": deck.characters, "cards": deck.cards},
                                       "games": len(games), "wins": sum(won for won, _ in games),
                                       "rounds": [rounds for _, rounds in games]}) + "\n")
            file.flush()


if __name__ == "__main__":
    main()
