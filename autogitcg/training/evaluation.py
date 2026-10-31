"""Fixed benchmark: the learner against a scripted opponent or a frozen network on a fixed list of matchups and seeds.

Every matchup is played four times: with the decks in both orders (so the learner plays each deck once,
which cancels out deck strength) and with the learner in both seats. Same matchups and seeds every time,
so results are comparable across checkpoints; every game is returned as a row for paired analysis, and
failed games come back with a full report. The first `replay_matchups` matchups are also recorded in full
(every state, the options, and the learner's probabilities).
"""

import random

from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.bridge.messages import Decision, GameFailed, GameFinished
from autogitcg.observation.enums import REACTION_NAMES
from autogitcg.observation.match_recorder import MatchRecorder
from autogitcg.observation.observation_encoder import ObservationEncoder
from autogitcg.policies.heuristic_policy import HeuristicPolicy
from autogitcg.policies.network_policy import NetworkChoice, NetworkPolicy
from autogitcg.policies.random_policy import random_response
from autogitcg.training.game_result import final_reward
from autogitcg.training.matchup_sampler import Matchup, swapped
from autogitcg.training.opponent_league import HEURISTIC


def evaluate(pool: EnginePool, encoder: ObservationEncoder, learner: NetworkPolicy, opponent: str,
             matchups: list[Matchup], seed: int, replay_matchups: int = 0,
             opponent_network: NetworkPolicy | None = None) -> tuple[dict, list[dict], list[dict], dict[str, str]]:
    """Returns (summary metrics, replays, one row per game, failure reports by summary line).
    opponent_network: plays the opponent seat instead of a scripted bot (opponent then just names it)."""
    rng = random.Random(seed)
    heuristic = HeuristicPolicy()
    games: dict[int, tuple[dict, MatchRecorder]] = {}  # game id -> (row, recorder)
    replays: dict[int, dict] = {}
    failure_reports: dict[str, str] = {}
    for number, original in enumerate(matchups):
        for deck_order, matchup in enumerate((original, swapped(original))):
            for learner_seat in (0, 1):
                game_id = pool.create_game(*matchup.decks, seed=seed + number, equipment=matchup.equipment)
                row = {"matchup": number, "deck_order": deck_order, "learner_seat": learner_seat, "seed": seed + number,
                       "learner_deck": matchup.deck_origins[learner_seat], "opponent_deck": matchup.deck_origins[1 - learner_seat]}
                games[game_id] = (row, MatchRecorder(matchup.decks, matchup.opponent_deck_known))
                if number < replay_matchups:
                    replays[game_id] = {**row, "opponent": opponent, "decisions": [],
                                        "decks": [{"characters": deck.characters, "cards": deck.cards} for deck in matchup.decks]}

    while pool.has_active_games():
        pending, opponent_pending = [], []
        for event in pool.poll():
            row, recorder = games[event.game_id]
            if isinstance(event, Decision):
                tracker = recorder.on_decision(event)
                if event.who == row["learner_seat"]:
                    pending.append((event, encoder.encode(event, tracker)))
                    continue
                if opponent_network is not None:
                    opponent_pending.append((event, encoder.encode(event, tracker)))
                    continue
                response = heuristic.respond(event, rng) if opponent == HEURISTIC else random_response(event, rng)
                _record(replays, event, response, None, None)
                recorder.on_response(event, response)
                pool.respond(event, response)
            elif isinstance(event, GameFinished):
                seat = row["learner_seat"]
                row.update(winner=event.winner, rounds=event.rounds, learner_won=event.winner == seat,
                           learner_reward=final_reward(event, seat), learner_damage=event.damage_dealt[seat],
                           learner_reactions={REACTION_NAMES.get(code, str(code)): count for code, count in event.reactions[seat].items()})
                if event.game_id in replays:
                    replays[event.game_id].update(winner=event.winner, rounds=event.rounds)
            elif isinstance(event, GameFailed):
                summary = event.message.splitlines()[0][:120]
                row.update(failure=summary)
                failure_reports.setdefault(summary, f"benchmark vs {opponent}: {row}\n{event.message[:4000]}")
        for (decision, encoded), choice in zip(pending, learner.choose(pending)):
            _record(replays, decision, choice.response, choice, encoded.request_indices)
            games[decision.game_id][1].on_response(decision, choice.response)
            pool.respond(decision, choice.response)
        if opponent_pending:
            for (decision, _), choice in zip(opponent_pending, opponent_network.choose(opponent_pending)):
                _record(replays, decision, choice.response, None, None)
                games[decision.game_id][1].on_response(decision, choice.response)
                pool.respond(decision, choice.response)

    rows = [row for row, _ in games.values()]
    return _summary(rows), list(replays.values()), rows, failure_reports


def _summary(rows: list[dict]) -> dict:
    finished = [row for row in rows if "failure" not in row]
    wins = [row for row in finished if row["learner_won"]]
    played = max(len(finished), 1)
    return {
        "games": len(rows),
        "failures": len(rows) - len(finished),
        "win_rate": len(wins) / played,
        "mean_reward": sum(row["learner_reward"] for row in finished) / played,
        "mean_rounds_of_wins": sum(row["rounds"] for row in wins) / max(len(wins), 1),
        "round_limit_rate": sum(row["winner"] is None for row in finished) / played,
        "learner_reactions_per_game": sum(sum(row["learner_reactions"].values()) for row in finished) / played,
    }


def _record(replays: dict[int, dict], decision: Decision, response: dict, choice: NetworkChoice | None,
            option_request_indices: list | None) -> None:
    """option_request_indices: what each probability refers to (engine action index, character id or card id)."""
    if decision.game_id not in replays:
        return
    entry = {"who": decision.who, "kind": decision.kind, "state": decision.state, "request": decision.request,
             "counters": {str(entity_id): values for entity_id, values in decision.counters.items()}, "response": response}
    if choice is not None:
        entry.update(probabilities=[round(float(probability), 4) for probability in choice.probabilities],
                     option_request_indices=option_request_indices, value=round(choice.value, 4),
                     win_probability=round(choice.win_probability, 4), entropy=round(choice.entropy, 4))
    replays[decision.game_id]["decisions"].append(entry)
