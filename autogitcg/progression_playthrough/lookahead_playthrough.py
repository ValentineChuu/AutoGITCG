"""Look-ahead arm: like engine_playthrough, but the account's side is played by Chudledum with look-ahead
(search/lookahead_table.py) and the NPC side by plain Chudledum. All accounts run side by side, one match each at a
time; a failed game is retried (it is not reported to the simulator)."""

import random
from collections.abc import Callable

from autogitcg.bridge.messages import GameFinished
from autogitcg.progression.match_result import match_result_from
from autogitcg.progression.simulator import ProgressionSimulator
from autogitcg.progression_playthrough.engine_playthrough import MAX_FAILURES_PER_ACCOUNT
from autogitcg.progression_playthrough.match_record import match_record, with_state_after
from autogitcg.progression_playthrough.stopping import playthrough_over
from autogitcg.search.lookahead_table import LookaheadTable
from autogitcg.training.matchup_sampler import Matchup, swapped


def run_lookahead_playthrough(simulators: list[ProgressionSimulator], deck_source, table: LookaheadTable, max_matches: int,
                              seed: int, on_match: Callable[[dict], None], on_failure: Callable[[str], None],
                              on_account_over: Callable[[int, int], None]) -> None:
    """on_account_over(account index, matches played) is called once per account when its playthrough ends."""
    rng = random.Random(seed)
    failures = [0] * len(simulators)
    playing = {}  # game id -> (account index, planned match, deck, account seat)
    idle = set(range(len(simulators)))

    def on_game_end(row: dict, event, recorder) -> None:
        index, planned, deck, account_seat = playing.pop(row["game_id"])
        idle.add(index)
        if isinstance(event, GameFinished):
            simulator = simulators[index]
            result = match_result_from(event, account_seat, recorder.trackers[account_seat])
            record = match_record(index, simulator.account, planned, result, deck)
            simulator.record(planned, result)
            on_match(with_state_after(record, simulator.account))
        else:
            failures[index] += 1
            on_failure(f"{planned.stage.kind} {planned.stage.level_id}: {event.message.splitlines()[0][:150]}")

    table.on_game_end = on_game_end
    while idle or playing:
        requests = []
        for index in sorted(idle):
            simulator = simulators[index]
            planned = None
            if not playthrough_over(simulator.account, max_matches) and failures[index] < MAX_FAILURES_PER_ACCOUNT:
                planned = simulator.next_match()
            if planned is not None:
                requests.append((index, planned))
            else:
                on_account_over(index, simulator.account.matches_played)
        idle.clear()
        decks = deck_source.decks([(simulators[index].account, planned.stage) for index, planned in requests])
        for (index, planned), deck in zip(requests, decks):
            stage = planned.stage
            matchup = Matchup("stage", (deck, stage.enemy_deck), stage.enemy_equipment, (True, False), ("account", f"stage:{stage.level_id}"))
            account_seat = rng.randrange(2)
            if account_seat:
                matchup = swapped(matchup)
            row = {}
            table.start_game(matchup, rng.randrange(2**31), searcher_seat=account_seat, row=row)
            playing[row["game_id"]] = (index, planned, deck, account_seat)
        if not playing:
            break
        table.step()
