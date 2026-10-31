"""Many games at once between a look-ahead searcher and a network opponent, in one engine pool.

At its action decisions the searcher (when unsure) copies the game once per candidate move and re-deal, plays every
copy `horizon` decisions ahead (or to a later round, settings.horizon_rounds) with its own network's most likely moves for both players, scores the position
with the network's win chance (2p - 1 from the searcher's side) or value (settings.score_by; a finished copy scores
+1 or -1), and picks the best candidate (decision_search.py). The win chance was chosen because it should be
symmetric between the players, but in practice the two players' win outputs disagree badly; with
settings.own_view_only every copy is scored at the searcher's own decision. Against a blind opponent with
`deck_guesses`, copies deal the opponent's hidden cards from a guessed deck (deck_guess.py); otherwise from the
opponent's real remaining cards (fair only when their deck list is known). Copies run in the same engine pool
as the real games, and all decisions waiting on the network (real and copies) are answered in shared batches.

`on_searcher_move(decision, encoded, own_choice, chosen_index, search)` is called once per searcher decision,
when its move is final: the network's own choice, the choice index actually played, and for searched decisions
a SearchResult (each candidate's choice index and mean score), else None.
`on_game_end(row, event, recorder)` is called when a real game ends (GameFinished or GameFailed), with its recorder.

A game can have a person in one seat (`human_seat`, e.g. a web board): their decisions go to `on_human_decision(row,
decision)` and their engine notifications to `on_notification(row, notification)`; answer with `respond_human`.
A game with `searcher_seat=None` has no searcher: every seat that isn't the person is played by `opponent`.
A game started with `searchers` ({seat: (searcher, rollout)}) has those seats search, each with its own networks
(e.g. two searching networks against each other); `searcher_seat` is then None.
`on_real_move(row, decision, response_value, search)` is called for every answered decision of a real game, any seat
(search: the SearchResult when the searcher searched it, else None), e.g. to log whole games.
"""

import random
import time
from collections import Counter
from dataclasses import dataclass

from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.bridge.messages import Decision, GameFailed, GameFinished, Notification, RepliesReady
from autogitcg.observation.match_recorder import MatchRecorder
from autogitcg.observation.observation_encoder import EncodedDecision, ObservationEncoder
from autogitcg.policies.grouped_choice import choose_grouped
from autogitcg.policies.network_policy import NetworkChoice, NetworkPolicy
from autogitcg.policies.network_responses import response_from_choice
from autogitcg.policies.reply_selection import ReplySelection, checked_actions, on_demand
from autogitcg.bridge.reply_views import policy_reads_replies, reply_views
from autogitcg.deck_guessing.reply_guess_candidates import ReplyGuessCandidates
from autogitcg.deck_guessing.tournament_deck_guesses import TournamentDeckGuesses, opponent_guesses
from autogitcg.search.deck_guess import Lineup, hidden_cards
from autogitcg.search.decision_search import DecisionSearch, GameCopy, candidate_choices, new_search
from autogitcg.search.lookahead_settings import LookaheadSettings
from autogitcg.training.game_result import final_reward, won
from autogitcg.training.matchup_sampler import Matchup


@dataclass
class SearchResult:
    candidate_indices: list[int]  # choice indices tried; the first is the network's own choice
    mean_scores: list[float | None]  # per candidate (None: too few of its copies finished)
    paired_scores: list[list[float | None]]  # per candidate, per re-deal (decision_search.DecisionSearch.paired_scores)


def ignore_move(decision, encoded, own_choice, chosen_index, search) -> None:
    pass


def ignore_game_end(row, event, recorder) -> None:
    pass


def ignore_human_event(row, event) -> None:
    pass


def ignore_real_move(row, decision, response_value, search) -> None:
    pass


class LookaheadTable:
    def __init__(self, pool: EnginePool, encoder: ObservationEncoder, searcher: NetworkPolicy, rollout: NetworkPolicy,
                 opponent: NetworkPolicy, settings: LookaheadSettings, seed: int,
                 deck_guesses: dict[Lineup, list[Counter]] | None = None, on_searcher_move=ignore_move,
                 on_game_end=ignore_game_end, on_human_decision=ignore_human_event, on_notification=ignore_human_event,
                 on_real_move=ignore_real_move, reply_selection: ReplySelection | None = None):
        """searcher: samples like normal play; rollout: the same network, greedy. reply_selection: real games' replies
        come on demand for network seats that read them, checked by the learned selector (policies/reply_selection.py)."""
        self.pool = pool
        self.encoder = encoder
        self.searcher = searcher
        self.rollout = rollout
        self.opponent = opponent
        self.settings = settings
        self.seed = seed
        self.deck_guesses = deck_guesses
        self.on_searcher_move = on_searcher_move
        self.on_game_end = on_game_end
        self.on_human_decision = on_human_decision
        self.on_notification = on_notification
        self.on_real_move = on_real_move
        self.rng = random.Random(seed)
        self.games: dict[int, tuple[dict, MatchRecorder]] = {}
        self.copies: dict[int, GameCopy] = {}
        self.waiting: dict[int, tuple[EncodedDecision, NetworkChoice, list[int], float]] = {}  # game id -> searched decision
        self.counts = Counter()
        self.search_seconds: list[float] = []  # wall time of every concluded search (decision to answer)
        # Blind seats' card replies are guessed from the same public tournament decks as the look-ahead's copies.
        self.reply_guesses = ReplyGuessCandidates(deck_guesses.decks) if isinstance(deck_guesses, TournamentDeckGuesses) else None
        self.reply_selection = reply_selection
        self.first_looks: dict[int, tuple[Decision, object]] = {}  # real game id -> (decision, tracker) waiting for replies
        self.searchers: dict[int, dict[int, tuple[NetworkPolicy, NetworkPolicy]]] = {}  # real game id -> {seat: (searcher, rollout)}
        self.judges: dict[int, dict[int, LookaheadSettings]] = {}  # real game id -> {seat: its own settings}, else self.settings

    def start_game(self, matchup: Matchup, seed: int, searcher_seat: int | None, row: dict, human_seat: int | None = None,
                   searchers: dict[int, tuple[NetworkPolicy, NetworkPolicy]] | None = None,
                   judges: dict[int, LookaheadSettings] | None = None) -> int:
        """row: the game's record, completed with the result (or failure) when it ends. searchers: seats that search
        with their own (searcher, rollout) networks, instead of searcher_seat. judges: searching seats' own settings
        (the table's settings for the others). Returns the game id."""
        if searchers is None:
            searchers = {searcher_seat: (self.searcher, self.rollout)} if searcher_seat is not None else {}
        views = None
        if self.pool.replies:
            players = [None if seat == human_seat else searchers[seat][0] if seat in searchers else self.opponent for seat in (0, 1)]
            reads = tuple(player is not None and policy_reads_replies(player) for player in players)
            views = reply_views(reads, matchup.opponent_deck_known, matchup.decks, self.reply_guesses)
            if self.reply_selection:
                views = [on_demand(view) for view in views]
        game_id = self.pool.create_game(*matchup.decks, seed=seed, equipment=matchup.equipment, human_seat=human_seat, reply_views=views)
        row.update(game_id=game_id, searcher_seat=searcher_seat, human_seat=human_seat, seed=seed,
                   seat_lineups=[tuple(sorted(deck.characters)) for deck in matchup.decks])
        if searcher_seat is not None:
            row.update(searcher_deck=matchup.deck_origins[searcher_seat], opponent_deck=matchup.deck_origins[1 - searcher_seat],
                       opponent_lineup=tuple(sorted(matchup.decks[1 - searcher_seat].characters)))
        self.games[game_id] = (row, MatchRecorder(matchup.decks, matchup.opponent_deck_known))
        self.searchers[game_id] = searchers
        self.judges[game_id] = judges or {}
        return game_id

    def adopt_game(self, game_id: int, row: dict, recorder: MatchRecorder) -> None:
        """A game started elsewhere (e.g. a replayed ground-truth position) is played from here on. row: as
        start_game fills it (searcher_seat, human_seat, opponent_lineup)."""
        self.games[game_id] = (row | {"game_id": game_id}, recorder)
        self.searchers[game_id] = {row["searcher_seat"]: (self.searcher, self.rollout)} if row["searcher_seat"] is not None else {}

    def respond_human(self, decision: Decision, response_value: dict) -> None:
        """The person's answer to a decision passed to on_human_decision."""
        if decision.game_id in self.games:
            self._answer_real(decision, response_value)

    def stop_game(self, game_id: int) -> None:
        """Ends a real game without a result (a person left or conceded); its copies finish unused."""
        if self.games.pop(game_id, None) is not None:
            self.searchers.pop(game_id, None)
            self.judges.pop(game_id, None)
            self.waiting.pop(game_id, None)
            self.first_looks.pop(game_id, None)
            self.pool.stop_game(game_id)

    @property
    def game_count(self) -> int:
        return len(self.games)

    def step(self, timeout: float | None = None) -> list[dict]:
        """Handles one poll of engine events (waiting at most `timeout` seconds for the first). Returns the rows of
        games that ended (finished or failed)."""
        ended: list[dict] = []
        searcher_pending, opponent_pending, rollout_pending = [], [], []
        searcher_first, opponent_first = [], []  # decisions whose replies come on demand: scored without them first
        finished_searches: list[DecisionSearch] = []
        for event in self.pool.poll(timeout):
            if event.game_id in self.games:
                row, recorder = self.games[event.game_id]
                if isinstance(event, Notification):
                    self.on_notification(row, event)
                    continue
                if isinstance(event, RepliesReady):
                    if event.game_id in self.first_looks:
                        decision, tracker = self.first_looks.pop(event.game_id)
                        decision.replies, decision.replies_checked = event.replies, event.checked
                        target = searcher_pending if decision.who in self.searchers[event.game_id] else opponent_pending
                        target.append((decision, self.encoder.encode(decision, tracker)))
                    continue
                if isinstance(event, Decision) and event.who == row["human_seat"]:
                    recorder.on_decision(event)
                    self.on_human_decision(row, event)
                    continue
                if isinstance(event, Decision):
                    tracker = recorder.on_decision(event)
                    encoded = self.encoder.encode(event, tracker)
                    if event.replies_on_demand:
                        self.first_looks[event.game_id] = (event, tracker)
                        (searcher_first if event.who in self.searchers[event.game_id] else opponent_first).append((event, encoded))
                        continue
                    target = searcher_pending if event.who in self.searchers[event.game_id] else opponent_pending
                    target.append((event, encoded))
                    continue
                seat = row["searcher_seat"]
                if isinstance(event, GameFinished):
                    row.update(winner=event.winner, rounds=event.rounds)
                    if seat is not None:
                        row.update(searcher_won=event.winner == seat, searcher_reward=final_reward(event, seat))
                elif isinstance(event, GameFailed):
                    row.update(failure=event.message.splitlines()[0][:200])
                del self.games[event.game_id]
                del self.searchers[event.game_id]
                self.judges.pop(event.game_id, None)
                self.waiting.pop(event.game_id, None)
                self.first_looks.pop(event.game_id, None)
                self.on_game_end(row, event, recorder)
                ended.append(row)
            elif event.game_id in self.copies:
                game_copy = self.copies[event.game_id]
                if isinstance(event, Decision):
                    encoded = self.encoder.encode(event, game_copy.recorder.on_decision(event))
                    game_copy.decisions_seen += 1
                    rollout_pending.append((event, encoded))
                    continue
                del self.copies[event.game_id]
                seat = game_copy.search.decision.who
                score = (1.0 if won(event, seat) else -1.0) if isinstance(event, GameFinished) else None
                self.counts["copies finished" if score is not None else "copies failed"] += 1
                game_copy.search.record(game_copy, score)
                if game_copy.search.finished:
                    finished_searches.append(game_copy.search)
            # anything else is a stopped copy's leftover event

        for first, choices in ((searcher_first, self._searcher_choices(searcher_first)), (opponent_first, self.opponent.choose(opponent_first))):
            for (decision, encoded), choice in zip(first, choices):
                self.pool.request_replies(decision, checked_actions(choice.reply_interest, encoded.request_indices,
                                                                    self.reply_selection.checks, full=False))
        for (decision, encoded), choice in zip(searcher_pending, self._searcher_choices(searcher_pending)):
            self._answer_searcher(decision, encoded, choice)
        for (decision, encoded), choice in zip(opponent_pending, self.opponent.choose(opponent_pending)):
            self._answer_real(decision, choice.response)
        for (decision, encoded), choice in zip(rollout_pending, self._rollout_choices(rollout_pending)):
            search = self._continue_copy(decision, choice)
            if search is not None and search.finished:
                finished_searches.append(search)
        for search in finished_searches:
            self._conclude(search)
        return ended

    def _searcher_choices(self, pending: list[tuple[Decision, EncodedDecision]]) -> list[NetworkChoice]:
        """Real-game decisions of searching seats, each answered by its seat's searcher network."""
        return choose_grouped(pending, lambda decision: self.searchers[decision.game_id][decision.who][0])

    def _rollout_choices(self, pending: list[tuple[Decision, EncodedDecision]]) -> list[NetworkChoice]:
        """Copy decisions (both players), answered by the rollout network of the seat whose search made the copy
        (the table's rollout once the real game is gone: the copy's result is unused then)."""
        def policy_of(decision: Decision) -> NetworkPolicy:
            searched = self.copies[decision.game_id].search.decision
            return self.searchers.get(searched.game_id, {}).get(searched.who, (None, self.rollout))[1]
        return choose_grouped(pending, policy_of)

    def _answer_real(self, decision: Decision, response_value: dict, search: SearchResult | None = None) -> None:
        """Plays a real game's move: its recorder, the engine, and on_real_move all see it."""
        row, recorder = self.games[decision.game_id]
        recorder.on_response(decision, response_value)
        self.pool.respond(decision, response_value)
        self.on_real_move(row, decision, response_value, search)

    def _settings_of(self, decision: Decision) -> LookaheadSettings:
        """The search settings of the seat that decides (a real game's decision)."""
        return self.judges.get(decision.game_id, {}).get(decision.who, self.settings)

    def _answer_searcher(self, decision: Decision, encoded: EncodedDecision, choice: NetworkChoice) -> None:
        settings = self._settings_of(decision)
        recorder = self.games[decision.game_id][1]
        move_kinds = [decision.request["action"][request_index]["action"]["$case"] for request_index in encoded.request_indices] \
            if decision.kind == "action" else None
        candidates = candidate_choices(choice.choice_index, choice.probabilities, settings, move_kinds) \
            if decision.kind == "action" else []
        if not candidates:
            self.on_searcher_move(decision, encoded, choice, choice.choice_index, None)
            self._answer_real(decision, choice.response)
            return
        self.counts["searched decisions"] += 1
        self.waiting[decision.game_id] = (encoded, choice, candidates, time.perf_counter())
        search = new_search(decision, recorder, [response_from_choice(decision, encoded, index, []) for index in candidates])
        tracker = recorder.trackers[decision.who]
        row = self.games[decision.game_id][0]
        opponent_lineup = row["seat_lineups"][1 - decision.who] if "seat_lineups" in row else row["opponent_lineup"]
        guesses = opponent_guesses(self.deck_guesses, opponent_lineup, tracker.opponent_played_cards)
        opponent_view = decision.state["player"][1 - decision.who]
        hand_ids = [card["id"] for card in opponent_view.get("handCard", [])]
        hidden_count = len(hand_ids) + len(opponent_view.get("pileCard", []))
        for redeal in range(settings.redeals):
            redeal_seed = hash((self.seed, decision.game_id, self.counts["searched decisions"], redeal)) & 0x7FFFFFFF
            # The opponent's hidden cards: from their deck list minus what they've revealed when it is known, else
            # from a guessed deck. (With neither, the engine pools their real cards: only fair with a known list.)
            guess = tracker.opponent_deck if tracker.opponent_deck is not None else (guesses[redeal % len(guesses)] if guesses else None)
            opponent_cards = hidden_cards(guess, tracker, hidden_count, self.rng) if guess is not None else None
            for candidate in range(len(candidates)):
                copy_id = self.pool.copy_game(decision, search.responses[candidate], redeal_seed, opponent_cards)
                self.copies[copy_id] = search.start_copy(candidate, redeal, opponent_cards, hand_ids)

    def _continue_copy(self, decision: Decision, choice: NetworkChoice) -> DecisionSearch | None:
        """Plays the copy's next move, or scores it by value once it has seen `horizon` decisions. Returns its
        search when the copy was scored."""
        game_copy = self.copies.get(decision.game_id)
        if game_copy is None:
            return None
        searcher = game_copy.search.decision.who
        settings = self._settings_of(game_copy.search.decision)
        reached = game_copy.decisions_seen >= settings.horizon or self._past_round_horizon(game_copy, decision, settings)
        if not reached or (settings.own_view_only and decision.who != searcher):
            game_copy.recorder.on_response(decision, choice.response)
            self.pool.respond(decision, choice.response)
            return None
        del self.copies[decision.game_id]
        self.pool.stop_game(decision.game_id)
        self.counts["copies scored by value"] += 1
        score = choice.value if settings.score_by == "value" else 2 * choice.win_probability - 1
        game_copy.search.record(game_copy, score if decision.who == searcher else -score)
        return game_copy.search

    def _past_round_horizon(self, game_copy: GameCopy, decision: Decision, settings: LookaheadSettings) -> bool:
        if settings.horizon_rounds <= 0:
            return False
        start_round = game_copy.search.decision.state.get("roundNumber", 0)
        return decision.state.get("roundNumber", 0) >= start_round + settings.horizon_rounds

    def _conclude(self, search: DecisionSearch) -> None:
        decision = search.decision
        if decision.game_id not in self.games:
            return  # the real game failed while its copies ran
        settings = self._settings_of(decision)
        chosen = search.best(settings)
        self.counts["overridden decisions"] += chosen != 0
        encoded, choice, candidates, started = self.waiting.pop(decision.game_id)
        self.search_seconds.append(time.perf_counter() - started)
        result = SearchResult(candidates, search.means(settings), search.paired_scores(settings))
        self.on_searcher_move(decision, encoded, choice, candidates[chosen], result)
        self._answer_real(decision, search.responses[chosen], result)
