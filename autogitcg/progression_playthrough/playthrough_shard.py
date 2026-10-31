"""One playthrough process: a share of the accounts (every n-th one), its own engine pool and networks.

Reports to the parent through a queue: ("match", won), ("account_over", matches played), ("failure", message),
("error", traceback) and finally ("shard_done", None).
"""

import gzip
import json
import random
import traceback
from dataclasses import dataclass
from pathlib import Path

from autogitcg.progression.shop import Shop
from autogitcg.progression.simulator import ProgressionSimulator
from autogitcg.progression.stage_catalog import build_stage_catalog


@dataclass(frozen=True)
class ShardSettings:
    arm: str
    run: Path
    first_account: int
    account_step: int
    account_count: int
    max_matches: int
    builder: Path
    combat: Path
    workers: int
    seed: int


def run_shard(settings: ShardSettings, messages) -> None:
    try:
        _run(settings, messages)
    except Exception:
        messages.put(("error", traceback.format_exc()))
    messages.put(("shard_done", None))


def _run(settings: ShardSettings, messages) -> None:
    import torch
    torch.set_num_threads(1)
    stages, _ = build_stage_catalog()
    shop = Shop()
    account_numbers = [settings.first_account + settings.account_step * index for index in range(settings.account_count)]
    simulators = [ProgressionSimulator(stages, shop, random.Random(settings.seed * 1000 + number)) for number in account_numbers]
    with gzip.open(settings.run / f"matches_{settings.first_account:03d}.jsonl.gz", "wt") as matches_file:
        def on_match(record: dict) -> None:
            local = record["account"]
            record["account"] = account_numbers[local]
            matches_file.write(json.dumps(record) + "\n")
            matches_file.flush()  # readable while the run goes on
            messages.put(("match", record["won"]))

        def on_account_over(local: int, matches_played: int) -> None:
            messages.put(("account_over", matches_played))

        def on_failure(message: str) -> None:
            messages.put(("failure", message))

        if settings.arm == "estimated":
            from autogitcg.progression_playthrough.estimated_playthrough import run_estimated
            run_estimated(simulators, settings.max_matches, settings.seed, on_match, on_account_over)
            return
        from autogitcg.analysis.builder_deck_check.check_players import NetworkPlayer
        from autogitcg.bridge.engine_pool import EnginePool
        from autogitcg.observation.observation_encoder import ObservationEncoder
        from autogitcg.progression_playthrough.deck_sources import ChudledeeDecks, RandomDecks
        from autogitcg.progression_playthrough.engine_playthrough import run_engine_playthrough
        encoder = ObservationEncoder()
        if settings.arm == "lookahead":
            _run_lookahead(settings, simulators, encoder, on_match, on_failure, on_account_over)
            return
        player = NetworkPlayer(settings.combat, encoder)
        if settings.arm == "random_decks":
            decks = RandomDecks(settings.seed * 1000 + settings.first_account)
        else:
            decks = ChudledeeDecks(settings.builder, skip_unusable_weapons=settings.arm == "chudledee_usable_weapons")
        with EnginePool(settings.workers) as pool:
            run_engine_playthrough(simulators, decks, player, pool, encoder, settings.max_matches,
                                   settings.seed * 1000 + settings.first_account, on_match, on_failure, on_account_over)


def _run_lookahead(settings: ShardSettings, simulators, encoder, on_match, on_failure, on_account_over) -> None:
    """The account's side searches (4 candidates, horizon 24: the classic judge's settings); the NPC side
    plays plain Chudledum from the same network."""
    import torch
    from autogitcg.bridge.engine_pool import EnginePool
    from autogitcg.model.combat_network_file import load_combat_network
    from autogitcg.policies.network_policy import NetworkPolicy
    from autogitcg.progression_playthrough.deck_sources import ChudledeeDecks
    from autogitcg.progression_playthrough.lookahead_playthrough import run_lookahead_playthrough
    from autogitcg.search.lookahead_settings import LookaheadSettings
    from autogitcg.search.lookahead_table import LookaheadTable
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = load_combat_network(settings.combat, encoder, device)
    seed = settings.seed * 1000 + settings.first_account
    with EnginePool(settings.workers) as pool:
        table = LookaheadTable(pool, encoder, NetworkPolicy(network, device), NetworkPolicy(network, device, greedy=True),
                               NetworkPolicy(network, device), LookaheadSettings(candidates=4, horizon=24), seed)
        run_lookahead_playthrough(simulators, ChudledeeDecks(settings.builder), table, settings.max_matches, seed,
                                  on_match, on_failure, on_account_over)
