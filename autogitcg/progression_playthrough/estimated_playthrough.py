"""The stand-in arm: every match decided by fast_forward.estimated_result (no engine), the assumption behind
the training collections."""

import random
from collections.abc import Callable

from autogitcg.progression.fast_forward import estimated_result
from autogitcg.progression.simulator import ProgressionSimulator
from autogitcg.progression_playthrough.match_record import match_record, with_state_after
from autogitcg.progression_playthrough.stopping import playthrough_over


def run_estimated(simulators: list[ProgressionSimulator], max_matches: int, seed: int, on_match: Callable[[dict], None],
                  on_account_over: Callable[[int, int], None]) -> None:
    rng = random.Random(seed)
    for index, simulator in enumerate(simulators):
        while not playthrough_over(simulator.account, max_matches) and (planned := simulator.next_match()) is not None:
            result = estimated_result(simulator.account, planned.stage, rng)
            record = match_record(index, simulator.account, planned, result, None)
            simulator.record(planned, result)
            on_match(with_state_after(record, simulator.account))
        on_account_over(index, simulator.account.matches_played)
