"""Keeps many games running and collects the learner's decisions until enough finished steps exist.

Network seats are answered in one batched forward pass per poll; scripted seats are answered directly.
Reference games are started in mirror pairs (same seed, decks swapped, the learner in the same seat); a pair's
mean learner score is the skill measure for the matchup category (training/adaptive_difficulty.py).
Games keep running across `collect` calls, so a few steps at each update come from the previous weights;
PPO's stored behavior log-probabilities make that correct, just slightly more off-policy.
"""

import random

from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.bridge.messages import Decision, GameFailed, GameFinished, RepliesReady
from autogitcg.bridge.reply_views import policy_reads_replies, reply_views
from autogitcg.deck_guessing.reply_guess_candidates import ReplyGuessCandidates
from autogitcg.model.observation_batch import compact
from autogitcg.observation.match_recorder import MatchRecorder
from autogitcg.observation.observation_encoder import ObservationEncoder
from autogitcg.policies.heuristic_policy import HeuristicPolicy
from autogitcg.policies.network_policy import NetworkChoice, NetworkPolicy
from autogitcg.policies.random_policy import random_response
from autogitcg.policies.reply_selection import FirstLook, ReplySelection, checked_actions, choose_full, interest_target, on_demand
from autogitcg.training.active_game import LEARNER, ActiveGame
from autogitcg.training.game_record import game_record, move_row
from autogitcg.training.game_result import SPEED_PENALTY_PER_ROUND, final_reward, won
from autogitcg.training.matchup_sampler import Matchup, MatchupSampler, swapped
from autogitcg.training.move_summary import summarize_move
from autogitcg.training.opponent_league import HEURISTIC, RANDOM, REFERENCE, SELF, SNAPSHOT, OpponentLeague
from autogitcg.training.card_play_bonus import card_play_bonus
from autogitcg.training.position_potential import position_potential
from autogitcg.training.rollout_stats import RolloutStats
from autogitcg.training.trajectory import Step, Trajectory


class RolloutCollector:
    def __init__(self, pool: EnginePool, encoder: ObservationEncoder, learner: NetworkPolicy, snapshot: NetworkPolicy,
                 matchups: MatchupSampler, league: OpponentLeague, concurrent_games: int, gae_lambda: float, rng: random.Random,
                 reference: NetworkPolicy | None = None, reply_selection: ReplySelection | None = None):
        """reply_selection: replies come on demand for network seats, checked by the learned selector (None: every action)."""
        self.pool = pool
        self.encoder = encoder
        self.policies = {LEARNER: learner, SNAPSHOT: snapshot}
        if reference is not None:
            self.policies[REFERENCE] = reference
        # Blind seats' card replies are guessed from the training tournament decks (never the locked test lineups).
        self.reply_guesses = ReplyGuessCandidates(matchups.pools.tournament_decks) if pool.replies else None
        self.reply_selection = reply_selection if pool.replies else None
        self.first_looks: dict[tuple[int, int], FirstLook] = {}  # (game, seat) -> a decision waiting for its replies
        self.heuristic = HeuristicPolicy()
        self.matchups = matchups
        self.league = league
        self.concurrent_games = concurrent_games
        self.gae_lambda = gae_lambda
        self.rng = rng
        self.games: dict[int, ActiveGame] = {}
        self.policy_version = 0  # learner weights version, stamped on every step
        self.snapshot_version: int | None = None  # update of the loaded snapshot opponent
        self.actor_index = 0
        self.speed_penalty = SPEED_PENALTY_PER_ROUND  # per round of a win's reward (game_result.py)
        self.shaping_weight = 0.0  # reward shaping by the position score (position_potential.py)
        self.card_play_bonus = 0.0  # immediate reward per action card played (card_play_bonus.py)
        self.card_kind_bonus: dict[str, float] | None = None  # extra per card kind (data/card_kinds.py)
        self.pair_scores: dict[int, list[float]] = {}  # mirror pair id -> learner scores so far; dropped if a game fails
        self.next_pair_id = 0

    def collect(self, step_count: int) -> tuple[list[Step], RolloutStats, list[dict]]:
        """Returns finished learner steps, counters, and a game_record for every finished game."""
        steps: list[Step] = []
        stats = RolloutStats()
        records: list[dict] = []
        while len(steps) < step_count:
            while len(self.games) < self.concurrent_games:
                self._start_game()
            pending: dict[str, list[tuple[Decision, object]]] = {controller: [] for controller in self.policies}
            first: dict[str, list[tuple[Decision, object, object]]] = {controller: [] for controller in self.policies}
            for event in self.pool.poll():
                game = self.games.get(event.game_id)
                if isinstance(event, RepliesReady):
                    self._replies_arrived(game, event, pending)
                elif isinstance(event, Decision):
                    stats.decisions += 1
                    self._handle_decision(game, event, pending, first)
                elif isinstance(event, GameFinished):
                    steps.extend(self._finish(game, event, stats))
                    records.append(game_record(game, event, self.policy_version, self.actor_index))
                    del self.games[event.game_id]
                    self._forget_first_looks(event.game_id)
                elif isinstance(event, GameFailed):
                    summary = event.message.splitlines()[0][:120]
                    stats.failures[summary] += 1
                    stats.failure_examples.setdefault(summary, _failure_report(game, event))
                    self.pair_scores.pop(game.pair_id, None)
                    del self.games[event.game_id]
                    self._forget_first_looks(event.game_id)
            for controller, decisions in first.items():
                self._first_look(controller, decisions)
            for controller, decisions in pending.items():
                self._answer_with_network(controller, decisions)
        return steps, stats, records

    def _start_game(self) -> None:
        matchup = self.matchups.sample()
        opponent = self.league.sample_opponent()
        if opponent == SELF:
            seats = (LEARNER, LEARNER)
        else:
            seats = (LEARNER, opponent) if self.rng.random() < 0.5 else (opponent, LEARNER)
        seed = self.rng.randrange(2**31)
        if opponent != REFERENCE:
            self._create(matchup, seats, seed, self.snapshot_version if opponent == SNAPSHOT else None, None)
            return
        pair_id = self.next_pair_id
        self.next_pair_id += 1
        self.pair_scores[pair_id] = []
        for each in (matchup, swapped(matchup)):
            self._create(each, seats, seed, None, pair_id)

    def _create(self, matchup: Matchup, seats: tuple[str, str], seed: int, snapshot_version: int | None, pair_id: int | None) -> None:
        views = None
        if self.pool.replies:
            reads = tuple(seat in self.policies and policy_reads_replies(self.policies[seat]) for seat in seats)
            views = reply_views(reads, matchup.opponent_deck_known, matchup.decks, self.reply_guesses)
            if self.reply_selection:
                views = [on_demand(view) for view in views]
        game_id = self.pool.create_game(*matchup.decks, seed=seed, equipment=matchup.equipment, reply_views=views)
        game = ActiveGame(matchup, seats, MatchRecorder(matchup.decks, matchup.opponent_deck_known), seed, snapshot_version, pair_id=pair_id)
        game.trajectories = {who: Trajectory() for who in (0, 1) if seats[who] == LEARNER}
        self.games[game_id] = game

    def _handle_decision(self, game: ActiveGame, decision: Decision, pending: dict, first: dict) -> None:
        tracker = game.recorder.on_decision(decision)
        controller = game.seats[decision.who]
        if controller in pending and decision.replies_on_demand:
            first[controller].append((decision, self.encoder.encode(decision, tracker), tracker))
            return
        if controller in pending:
            pending[controller].append((decision, self.encoder.encode(decision, tracker)))
            return
        if controller == HEURISTIC:
            response = self.heuristic.respond(decision, self.rng)
        elif controller == RANDOM:
            response = random_response(decision, self.rng)
        else:
            raise ValueError(f"unknown controller {controller}")
        self._respond(game, decision, response)

    def _first_look(self, controller: str, decisions: list) -> None:
        """Score decisions without replies, then ask for the replies of the actions the selector picks."""
        choices = self.policies[controller].choose([(decision, encoded) for decision, encoded, _ in decisions])
        for (decision, encoded, tracker), choice in zip(decisions, choices):
            full = controller == LEARNER and choose_full(self.reply_selection, self.rng)
            self.first_looks[decision.game_id, decision.who] = FirstLook(decision, tracker, choice.probabilities, full)
            self.pool.request_replies(decision, checked_actions(choice.reply_interest, encoded.request_indices,
                                                                self.reply_selection.checks, full))

    def _forget_first_looks(self, game_id: int) -> None:
        for key in [key for key in self.first_looks if key[0] == game_id]:
            del self.first_looks[key]

    def _replies_arrived(self, game: ActiveGame | None, event: RepliesReady, pending: dict) -> None:
        look = self.first_looks.get((event.game_id, event.who))
        if game is None or look is None:
            return
        look.decision.replies, look.decision.replies_checked = event.replies, event.checked
        pending[game.seats[event.who]].append((look.decision, self.encoder.encode(look.decision, look.tracker)))

    def _answer_with_network(self, controller: str, decisions: list) -> None:
        for (decision, encoded), choice in zip(decisions, self.policies[controller].choose(decisions)):
            game = self.games[decision.game_id]
            look = self.first_looks.pop((decision.game_id, decision.who), None)
            if controller == LEARNER:
                game.trajectories[decision.who].steps.append(Step(
                    compact(encoded.arrays), choice.kind, choice.choice_index, choice.choice_bits, choice.log_prob, choice.value,
                    self.policy_version, policy_log_prob=choice.policy_log_prob,
                    potential=position_potential(decision.state, decision.who),
                    bonus=card_play_bonus(decision, choice.response, self.card_play_bonus, self.card_kind_bonus),
                    reply_target=interest_target(look.probabilities, choice.probabilities) if look and look.full else None))
            self._respond(game, decision, choice.response, choice)

    def _respond(self, game: ActiveGame, decision: Decision, response: dict, choice: NetworkChoice | None = None) -> None:
        game.moves.append(move_row(decision.who, summarize_move(decision, response), choice))
        game.recorder.on_response(decision, response)
        self.pool.respond(decision, response)

    def _finish(self, game: ActiveGame, finished: GameFinished, stats: RolloutStats) -> list[Step]:
        opponent = game.opponent
        stats.games[opponent] += 1
        stats.round_limit += finished.winner is None
        if game.pair_id in self.pair_scores:
            learner_seat = game.seats.index(LEARNER)
            scores = self.pair_scores[game.pair_id]
            scores.append(float(won(finished, learner_seat)))
            if len(scores) == 2:
                self.matchups.record_result(game.matchup.category, sum(scores) / 2)
                del self.pair_scores[game.pair_id]
        steps = []
        for who, trajectory in game.trajectories.items():
            if opponent != SELF and won(finished, who):
                stats.learner_wins[opponent] += 1
            steps.extend(trajectory.finish(final_reward(finished, who, self.speed_penalty), won(finished, who), self.gae_lambda,
                                           self.shaping_weight))
        return steps


def _failure_report(game: ActiveGame, failed: GameFailed) -> str:
    matchup = game.matchup
    return (f"source {matchup.source}, seats {game.seats}, origins {matchup.deck_origins}, seed {game.seed}\n"
            f"decks {[(deck.characters, deck.cards) for deck in matchup.decks]}\n"
            f"equipment {matchup.equipment}\n{failed.message[:4000]}")
