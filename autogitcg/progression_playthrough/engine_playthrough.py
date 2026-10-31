"""Engine arms: every planned match is played for real. A deck source picks the account's deck; one player
(e.g. Chudledum) plays both the account's side, knowing the stage deck, and the NPC side.

All accounts run side by side, one match each at a time, so decks are built and decisions answered in batches.
Seats are random per match. A failed game is retried (it is not reported to the simulator).
"""

import random
from collections.abc import Callable

from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.bridge.messages import Decision, GameFailed, GameFinished
from autogitcg.observation.match_recorder import MatchRecorder
from autogitcg.observation.observation_encoder import ObservationEncoder
from autogitcg.progression.match_result import match_result_from
from autogitcg.progression.simulator import ProgressionSimulator
from autogitcg.progression_playthrough.match_record import match_record, with_state_after
from autogitcg.progression_playthrough.stopping import playthrough_over
from autogitcg.training.matchup_sampler import Matchup, swapped

MAX_FAILURES_PER_ACCOUNT = 20


def run_engine_playthrough(simulators: list[ProgressionSimulator], deck_source, player, pool: EnginePool,
                           encoder: ObservationEncoder, max_matches: int, seed: int,
                           on_match: Callable[[dict], None], on_failure: Callable[[str], None],
                           on_account_over: Callable[[int, int], None]) -> None:
    """on_account_over(account index, matches played) is called once per account when its playthrough ends."""
    rng = random.Random(seed)
    failures = [0] * len(simulators)
    active = {}  # game id -> (account index, planned match, deck, account seat, recorder)
    idle = set(range(len(simulators)))
    while idle or active:
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
            game_id = pool.create_game(*matchup.decks, seed=rng.randrange(2**31), equipment=matchup.equipment)
            active[game_id] = (index, planned, deck, account_seat, MatchRecorder(matchup.decks, matchup.opponent_deck_known))
        if not active:
            break
        pending = []
        for event in pool.poll(timeout=1):
            index, planned, deck, account_seat, recorder = active[event.game_id]
            if isinstance(event, Decision):
                tracker = recorder.on_decision(event)
                pending.append((event, encoder.encode(event, tracker) if player.needs_encoding else None))
                continue
            del active[event.game_id]
            idle.add(index)
            if isinstance(event, GameFinished):
                simulator = simulators[index]
                result = match_result_from(event, account_seat, recorder.trackers[account_seat])
                record = match_record(index, simulator.account, planned, result, deck)
                simulator.record(planned, result)
                on_match(with_state_after(record, simulator.account))
            elif isinstance(event, GameFailed):
                failures[index] += 1
                on_failure(f"{planned.stage.kind} {planned.stage.level_id}: {event.message.splitlines()[0][:150]}")
        for (decision, _), response in zip(pending, player.respond(pending)):
            active[decision.game_id][4].on_response(decision, response)
            pool.respond(decision, response)
