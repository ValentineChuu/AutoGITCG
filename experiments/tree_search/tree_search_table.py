"""Many games at once between a tree searcher and a network opponent, in one engine pool (experiments/tree_search/notes.md).

At its action decisions the searcher (when unsure) runs a TreeSearch: `iterations` game copies, at most `parallel` at
a time, each with the hidden cards dealt again (from a guessed deck against a blind opponent, deck_guess.py; from
the opponent's real remaining cards when their deck list is known). Copies are played by the rollout network
(greedy) wherever the tree doesn't choose. All decisions waiting on the network, real and copies, are answered in
shared batches. Only for tests for now: the playtest site still plays with lookahead_table.py.

`on_searcher_move(decision, encoded, own_choice, played_response, search)` is called once per searcher decision when
its move is final (search: the TreeSearch, or None when the network played directly).
"""

import random
import time
from collections import Counter

from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.bridge.messages import Decision, GameFailed, GameFinished
from autogitcg.observation.match_recorder import MatchRecorder
from autogitcg.observation.observation_encoder import EncodedDecision, ObservationEncoder
from autogitcg.policies.network_policy import NetworkChoice, NetworkPolicy
from autogitcg.deck_guessing.tournament_deck_guesses import opponent_guesses
from autogitcg.search.deck_guess import Lineup, hidden_cards
from experiments.tree_search.move_options import move_options
from experiments.tree_search.tree_search import TreeSearch, Walk
from experiments.tree_search.tree_search_settings import TreeSearchSettings
from autogitcg.training.game_result import final_reward, won
from autogitcg.training.matchup_sampler import Matchup


def ignore_move(decision, encoded, own_choice, played_response, search) -> None:
    pass


class TreeSearchTable:
    def __init__(self, pool: EnginePool, encoder: ObservationEncoder, searcher: NetworkPolicy, rollout: NetworkPolicy,
                 opponent: NetworkPolicy, settings: TreeSearchSettings, seed: int,
                 deck_guesses: dict[Lineup, list[Counter]] | None = None, on_searcher_move=ignore_move):
        """searcher: samples like normal play; rollout: the same network, greedy."""
        self.pool = pool
        self.encoder = encoder
        self.searcher = searcher
        self.rollout = rollout
        self.opponent = opponent
        self.settings = settings
        self.seed = seed
        self.deck_guesses = deck_guesses
        self.on_searcher_move = on_searcher_move
        self.rng = random.Random(seed)
        self.games: dict[int, tuple[dict, MatchRecorder]] = {}
        self.walks: dict[int, Walk] = {}  # copy id -> its walk
        self.searches: dict[int, tuple[TreeSearch, EncodedDecision, NetworkChoice, float]] = {}  # game id -> open search
        self.counts = Counter()
        self.search_seconds: list[float] = []

    def start_game(self, matchup: Matchup, seed: int, searcher_seat: int, row: dict) -> int:
        game_id = self.pool.create_game(*matchup.decks, seed=seed, equipment=matchup.equipment)
        row.update(game_id=game_id, searcher_seat=searcher_seat, seed=seed,
                   searcher_deck=matchup.deck_origins[searcher_seat], opponent_deck=matchup.deck_origins[1 - searcher_seat],
                   opponent_lineup=tuple(sorted(matchup.decks[1 - searcher_seat].characters)))
        self.games[game_id] = (row, MatchRecorder(matchup.decks, matchup.opponent_deck_known))
        return game_id

    def adopt_game(self, game_id: int, row: dict, recorder: MatchRecorder) -> None:
        """A game started elsewhere (e.g. a replayed ground-truth position) is played from here on. row: as
        start_game fills it (searcher_seat, opponent_lineup)."""
        self.games[game_id] = (row | {"game_id": game_id}, recorder)

    def stop_game(self, game_id: int) -> None:
        """Ends a real game without a result; walks of its open search finish unused."""
        if self.games.pop(game_id, None) is not None:
            self.searches.pop(game_id, None)
            self.pool.stop_game(game_id)

    @property
    def game_count(self) -> int:
        return len(self.games)

    def step(self, timeout: float | None = None) -> list[dict]:
        """Handles one poll of engine events. Returns the rows of real games that ended (finished or failed)."""
        ended: list[dict] = []
        searcher_pending, opponent_pending, rollout_pending = [], [], []
        touched: set[int] = set()  # game ids whose searches had a walk end
        for event in self.pool.poll(timeout):
            if event.game_id in self.games:
                row, recorder = self.games[event.game_id]
                if isinstance(event, Decision):
                    encoded = self.encoder.encode(event, recorder.on_decision(event))
                    (searcher_pending if event.who == row["searcher_seat"] else opponent_pending).append((event, encoded))
                    continue
                if isinstance(event, GameFinished):
                    seat = row["searcher_seat"]
                    row.update(winner=event.winner, rounds=event.rounds, searcher_won=event.winner == seat,
                               searcher_reward=final_reward(event, seat))
                elif isinstance(event, GameFailed):
                    row.update(failure=event.message.splitlines()[0][:200])
                del self.games[event.game_id]
                self.searches.pop(event.game_id, None)
                ended.append(row)
            elif event.game_id in self.walks:
                walk = self.walks[event.game_id]
                if isinstance(event, Decision):
                    encoded = self.encoder.encode(event, walk.recorder.on_decision(event))
                    rollout_pending.append((event, encoded))
                    continue
                del self.walks[event.game_id]
                score = (1.0 if won(event, walk.search.decision.who) else -1.0) if isinstance(event, GameFinished) else None
                self.counts["copies played out" if score is not None else "copies failed"] += 1
                walk.search.finish_walk(walk, score)
                touched.add(walk.search.decision.game_id)
            # anything else is a stopped copy's leftover event

        for (decision, encoded), choice in zip(searcher_pending, self.searcher.choose(searcher_pending)):
            self._answer_searcher(decision, encoded, choice)
        for (decision, encoded), choice in zip(opponent_pending, self.opponent.choose(opponent_pending)):
            self._answer_real(decision, choice.response)
        for (decision, encoded), choice in zip(rollout_pending, self.rollout.choose(rollout_pending)):
            game_id = self._continue_copy(decision, encoded, choice)
            if game_id is not None:
                touched.add(game_id)
        for game_id in touched:
            self._advance(game_id)
        return ended

    def _answer_real(self, decision: Decision, response_value: dict) -> None:
        recorder = self.games[decision.game_id][1]
        recorder.on_response(decision, response_value)
        self.pool.respond(decision, response_value)

    def _answer_searcher(self, decision: Decision, encoded: EncodedDecision, choice: NetworkChoice) -> None:
        if decision.kind != "action" or len(choice.probabilities) < 2 or choice.probabilities.max() >= self.settings.confident_probability:
            self.on_searcher_move(decision, encoded, choice, choice.response, None)
            self._answer_real(decision, choice.response)
            return
        self.counts["searched decisions"] += 1
        recorder = self.games[decision.game_id][1]
        search = TreeSearch(decision, recorder, move_options(decision, encoded, choice.probabilities), self.settings)
        self.searches[decision.game_id] = (search, encoded, choice, time.perf_counter())
        self._advance(decision.game_id)

    def _advance(self, game_id: int) -> None:
        """Starts the search's next walks, or answers the real decision once every walk has finished."""
        if game_id not in self.searches:
            return  # the real game ended while its copies ran
        search, encoded, choice, started = self.searches[game_id]
        while search.wants_walk():
            self._start_walk(search)
        if search.finished:
            del self.searches[game_id]
            self.search_seconds.append(time.perf_counter() - started)
            chosen = search.chosen()
            self.counts["overridden decisions"] += chosen.response["chosenActionIndex"] != choice.response["chosenActionIndex"]
            self.counts.update(search.counts)
            self.on_searcher_move(search.decision, encoded, choice, chosen.response, search)
            self._answer_real(search.decision, chosen.response)

    def _start_walk(self, search: TreeSearch) -> None:
        decision = search.decision
        row, recorder = self.games[decision.game_id]
        tracker = recorder.trackers[decision.who]
        guesses = opponent_guesses(self.deck_guesses, row["opponent_lineup"], tracker.opponent_played_cards)
        opponent_view = decision.state["player"][1 - decision.who]
        hand_ids = [card["id"] for card in opponent_view.get("handCard", [])]
        hidden_count = len(hand_ids) + len(opponent_view.get("pileCard", []))
        redeal_seed = hash((self.seed, decision.game_id, self.counts["searched decisions"], search.launched)) & 0x7FFFFFFF
        guess = tracker.opponent_deck if tracker.opponent_deck is not None else (guesses[search.launched % len(guesses)] if guesses else None)
        opponent_cards = hidden_cards(guess, tracker, hidden_count, self.rng) if guess is not None else None
        walk = search.start_walk(opponent_cards, hand_ids)
        root_response = search.options[walk.path[0][1]].response
        copy_id = self.pool.copy_game(decision, root_response, redeal_seed, opponent_cards)
        self.walks[copy_id] = walk

    def _continue_copy(self, decision: Decision, encoded: EncodedDecision, choice: NetworkChoice) -> int | None:
        """Plays the copy's next move: chosen in the tree, else by the network. Returns the real game's id when the
        copy was scored here (leaf horizon reached)."""
        walk = self.walks.get(decision.game_id)
        if walk is None:
            return None
        if decision.kind == "action":
            options = move_options(decision, encoded, choice.probabilities)
            key = walk.tree_move(decision, options, self.settings)
            if key is not None:
                self._respond_copy(walk, decision, options[key].response)
                return None
        if not walk.in_tree and self._past_leaf_horizon(walk, decision):
            del self.walks[decision.game_id]
            self.pool.stop_game(decision.game_id)
            self.counts["copies scored by win chance"] += 1
            searcher = walk.search.decision.who
            win_probability = choice.win_probability if decision.who == searcher else 1 - choice.win_probability
            walk.search.finish_walk(walk, 2 * win_probability - 1)
            return walk.search.decision.game_id
        self._respond_copy(walk, decision, choice.response)
        return None

    def _respond_copy(self, walk: Walk, decision: Decision, response_value: dict) -> None:
        walk.recorder.on_response(decision, response_value)
        self.pool.respond(decision, response_value)

    def _past_leaf_horizon(self, walk: Walk, decision: Decision) -> bool:
        if self.settings.leaf_horizon_rounds <= 0:
            return False
        start_round = walk.search.decision.state.get("roundNumber", 0)
        return decision.state.get("roundNumber", 0) >= start_round + self.settings.leaf_horizon_rounds
