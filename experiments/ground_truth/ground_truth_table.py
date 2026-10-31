"""Builds ground-truth positions: plain network games, one sampled unsure decision per game, every compared move
played out on the same guessed deals until the best is clear (experiments/tree_search/notes.md).

Both seats play the network (sampling, like normal play). In each game one seat is the sampled seat; at its unsure
action decisions a position is taken with `sample_probability` (at most one per game). The game then waits while the
position's copies run: deal after deal (each deal = every compared move, its copies sharing the deal's seed and guessed
cards), at most `deals_in_flight` deals of a position open at once, each copy played to the end by the network
(greedy, both players). New deals start as soon as old ones finish (no batch waits for its slowest game, which left
the engine idle). After every completed deal the position is checked; once the best move is separated, every move
had identical results, or `max_deals` deals are done, its remaining copies are stopped, the position is written
and its game stopped. Each position keeps the game's
seed and every response so far, so the exact decision can be replayed.

A taken position is also replayed on every other engine worker (position_mirrors.py) and its deals are spread over
all of them, since a copy runs on the worker of the game it is made from.

Blind games deal the opponent's hidden cards from Chudledee guesses (search/deck_guess.py), as the playtest does.
"""

import random
from collections import Counter
from dataclasses import dataclass, field

from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.bridge.messages import Decision, GameFinished
from experiments.ground_truth.candidate_moves import candidate_moves
from experiments.ground_truth.ground_truth_settings import GroundTruthSettings
from experiments.ground_truth.move_comparison import MoveComparison
from experiments.ground_truth.position_mirrors import MirrorReplay
from autogitcg.observation.match_recorder import MatchRecorder
from autogitcg.observation.observation_encoder import EncodedDecision, ObservationEncoder
from autogitcg.policies.network_policy import NetworkChoice, NetworkPolicy
from autogitcg.search.deck_guess import Lineup, hidden_cards
from experiments.tree_search.move_options import MoveOption
from autogitcg.training.game_result import won
from autogitcg.training.matchup_sampler import Matchup


@dataclass
class Position:
    game_id: int
    decision: Decision
    recorder: MatchRecorder  # the sampled seat's view at the decision
    encoded: EncodedDecision
    choice: NetworkChoice  # the network's own (sampled) choice and probabilities
    moves: list[MoveOption]
    comparison: MoveComparison
    deals_started: int = 0
    deals_completed: int = 0
    open_copies_by_deal: Counter = field(default_factory=Counter)
    settled: bool = False  # no more deals: separated, identical results, or max_deals
    anchors: list[Decision] = field(default_factory=list)  # the decision in the game and in each mirror that reached it
    mirror_ids: list[int] = field(default_factory=list)


@dataclass
class PendingCopy:
    position: Position
    move: int
    deal: int
    recorder: MatchRecorder


@dataclass
class GameRecord:
    row: dict
    recorder: MatchRecorder
    matchup: Matchup
    responses: dict[int, list] = field(default_factory=lambda: {0: [], 1: []})


class GroundTruthTable:
    def __init__(self, pool: EnginePool, encoder: ObservationEncoder, player: NetworkPolicy, rollout: NetworkPolicy,
                 settings: GroundTruthSettings, seed: int, deck_guesses: dict[Lineup, list[Counter]] | None, on_position):
        """player: samples like normal play; rollout: the same network, greedy. on_position(row, position) is called
        once per finished position."""
        self.pool = pool
        self.encoder = encoder
        self.player = player
        self.rollout = rollout
        self.settings = settings
        self.seed = seed
        self.deck_guesses = deck_guesses
        self.on_position = on_position
        self.rng = random.Random(seed)
        self.games: dict[int, GameRecord] = {}
        self.copies: dict[int, PendingCopy] = {}
        self.positions: list[Position] = []  # open positions, in the order they were taken
        self.mirrors: dict[int, MirrorReplay] = {}  # mirror game id -> its replay, until it reaches the position
        self.counts = Counter()

    def start_game(self, matchup: Matchup, seed: int, sampled_seat: int, row: dict) -> None:
        game_id = self.pool.create_game(*matchup.decks, seed=seed, equipment=matchup.equipment)
        row.update(game_id=game_id, sampled_seat=sampled_seat, seed=seed, position_taken=False,
                   opponent_lineup=tuple(sorted(matchup.decks[1 - sampled_seat].characters)))
        self.games[game_id] = GameRecord(row, MatchRecorder(matchup.decks, matchup.opponent_deck_known), matchup)

    @property
    def busy(self) -> bool:
        return bool(self.games)

    def step(self, timeout: float | None = None) -> None:
        real_pending, rollout_pending = [], []
        for event in self.pool.poll(timeout):
            if event.game_id in self.games:
                game = self.games[event.game_id]
                if isinstance(event, Decision):
                    real_pending.append((event, self.encoder.encode(event, game.recorder.on_decision(event))))
                else:
                    self.counts["games ended without a position"] += 1
                    del self.games[event.game_id]
            elif event.game_id in self.mirrors:
                self._replay_mirror(event)
            elif event.game_id in self.copies:
                pending = self.copies[event.game_id]
                if isinstance(event, Decision):
                    rollout_pending.append((event, self.encoder.encode(event, pending.recorder.on_decision(event))))
                    continue
                del self.copies[event.game_id]
                self._copy_ended(pending, event)
        for (decision, encoded), choice in zip(real_pending, self.player.choose(real_pending)):
            self._answer_real(decision, encoded, choice)
        for (decision, encoded), choice in zip(rollout_pending, self.rollout.choose(rollout_pending)):
            pending = self.copies.get(decision.game_id)
            if pending is None:
                continue  # its position settled during this step and stopped its leftover copies
            pending.recorder.on_response(decision, choice.response)
            self.pool.respond(decision, choice.response)
        self._launch()

    def _answer_real(self, decision: Decision, encoded: EncodedDecision, choice: NetworkChoice) -> None:
        game = self.games[decision.game_id]
        row = game.row
        if (not row["position_taken"] and decision.who == row["sampled_seat"] and decision.kind == "action"
                and len(choice.probabilities) > 1 and choice.probabilities.max() < self.settings.unsure_probability
                and self.rng.random() < self.settings.sample_probability):
            moves = candidate_moves(decision, encoded, choice.probabilities, self.settings.likely_moves)
            if len(moves) > 1:
                row["position_taken"] = True
                row["responses_before"] = {who: list(responses) for who, responses in game.responses.items()}
                position = Position(decision.game_id, decision, game.recorder, encoded, choice, moves,
                                    MoveComparison(len(moves)), anchors=[decision])
                self.positions.append(position)
                self._start_mirrors(position, game)
                return  # the game waits here until the position is done
        game.recorder.on_response(decision, choice.response)
        game.responses[decision.who].append(choice.response)
        self.pool.respond(decision, choice.response)

    def _start_mirrors(self, position: Position, game: GameRecord) -> None:
        own_worker = self.pool.worker_index_of(position.game_id)
        for worker_index in range(len(self.pool.workers)):
            if worker_index != own_worker:
                mirror_id = self.pool.create_game(*game.matchup.decks, seed=game.row["seed"], equipment=game.matchup.equipment,
                                                  worker_index=worker_index)
                self.mirrors[mirror_id] = MirrorReplay.of(position, game.row["responses_before"])
                position.mirror_ids.append(mirror_id)

    def _replay_mirror(self, event) -> None:
        replay = self.mirrors[event.game_id]
        if not isinstance(event, Decision):
            del self.mirrors[event.game_id]
            self.counts["mirrors failed"] += 1
            return
        response = replay.next_response(event)
        if response is not None:
            self.pool.respond(event, response)
            return
        del self.mirrors[event.game_id]
        if replay.reached(event):
            replay.position.anchors.append(event)
            self.counts["mirrors reached"] += 1
        else:
            self.pool.stop_game(event.game_id)
            self.counts["mirrors diverged"] += 1

    def _wants_deal(self, position: Position) -> bool:
        return (not position.settled and position.deals_started < self.settings.max_deals
                and position.deals_started - position.deals_completed < self.settings.deals_in_flight)

    def _launch(self) -> None:
        """Starts deals until the process's copy limit, oldest position first, so positions finish one after another
        (sharing deals out evenly kept every position open until the end, and a crash lost them all)."""
        for position in self.positions:
            while self._wants_deal(position):
                if len(self.copies) >= self.settings.max_open_copies:
                    return
                self._start_deal(position)

    def _start_deal(self, position: Position) -> None:
        decision = position.decision
        row = self.games[position.game_id].row
        tracker = position.recorder.trackers[decision.who]
        guesses = self.deck_guesses.get(row["opponent_lineup"]) if self.deck_guesses else None
        opponent_view = decision.state["player"][1 - decision.who]
        hand_ids = [card["id"] for card in opponent_view.get("handCard", [])]
        hidden_count = len(hand_ids) + len(opponent_view.get("pileCard", []))
        deal = position.deals_started
        deal_seed = hash((self.seed, position.game_id, deal)) & 0x7FFFFFFF
        guess = tracker.opponent_deck if tracker.opponent_deck is not None else (guesses[deal % len(guesses)] if guesses else None)
        opponent_cards = hidden_cards(guess, tracker, hidden_count, self.rng) if guess is not None else None
        for move_index, move in enumerate(position.moves):
            recorder = position.recorder.copy_for_redeal(decision, opponent_cards, hand_ids)
            recorder.on_response(decision, move.response)
            anchor = position.anchors[deal % len(position.anchors)]
            copy_id = self.pool.copy_game(anchor, move.response, deal_seed, opponent_cards)
            self.copies[copy_id] = PendingCopy(position, move_index, deal, recorder)
            position.open_copies_by_deal[deal] += 1
        position.deals_started += 1

    def _copy_ended(self, pending: PendingCopy, event) -> None:
        position = pending.position
        if isinstance(event, GameFinished):
            position.comparison.add(pending.move, pending.deal, 1.0 if won(event, position.decision.who) else -1.0)
            self.counts["copies played out"] += 1
        else:
            self.counts["copies failed"] += 1
        position.open_copies_by_deal[pending.deal] -= 1
        if position.open_copies_by_deal[pending.deal]:
            return
        del position.open_copies_by_deal[pending.deal]
        position.deals_completed += 1
        comparison = position.comparison
        if position.deals_completed >= self.settings.minimum_deals and (comparison.separated(self.settings.separation)
                                                                         or comparison.all_identical()):
            position.settled = True
        if position.deals_completed >= self.settings.max_deals:
            position.settled = True
        if position.settled:
            self._finish(position)

    def _finish(self, position: Position) -> None:
        """Stops the position's copies still running, writes it and stops its game."""
        for copy_id in [copy_id for copy_id, pending in self.copies.items() if pending.position is position]:
            del self.copies[copy_id]
            self.pool.stop_game(copy_id)
            self.counts["copies stopped unneeded"] += 1
        for mirror_id in position.mirror_ids:
            self.mirrors.pop(mirror_id, None)
            self.pool.stop_game(mirror_id)
        self.positions.remove(position)
        game = self.games.pop(position.game_id)
        self.pool.stop_game(position.game_id)
        self.counts["positions"] += 1
        self.on_position(game.row, position)
