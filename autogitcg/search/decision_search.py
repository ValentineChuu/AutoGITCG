"""The search for one action decision: candidate moves, the game copies playing each, and their scores.

The network's own (sampled) choice is always a candidate, and stays the answer unless another candidate's copies
score better by the settings' margin. So with an infinite margin the searcher plays exactly like the network.
"""

from dataclasses import dataclass, field

import numpy as np

from autogitcg.bridge.messages import Decision
from autogitcg.observation.match_recorder import MatchRecorder
from autogitcg.search.lookahead_settings import LookaheadSettings


@dataclass
class GameCopy:
    search: "DecisionSearch"
    candidate: int  # index into search.responses
    redeal: int  # which re-deal of the hidden cards (the same re-deals are shared by every candidate)
    recorder: MatchRecorder  # the real game's trackers, then this copy's decisions
    decisions_seen: int = 0


@dataclass
class DecisionSearch:
    decision: Decision
    recorder: MatchRecorder  # the real game's
    responses: list[dict]  # candidate answers; index 0 is the network's own choice
    scores: list[dict[int, float]] = field(default_factory=list)  # per candidate: re-deal -> its copy's score
    open_copies: int = 0

    def start_copy(self, candidate: int, redeal: int, opponent_cards: list[int] | None = None,
                   opponent_hand_ids: list[int] = ()) -> GameCopy:
        """opponent_cards: as in MatchRecorder.copy_for_redeal."""
        recorder = self.recorder.copy_for_redeal(self.decision, opponent_cards, opponent_hand_ids)
        recorder.on_response(self.decision, self.responses[candidate])
        self.open_copies += 1
        return GameCopy(self, candidate, redeal, recorder)

    def record(self, game_copy: GameCopy, score: float | None) -> None:
        """score: the copy's result from the searching player's side, or None if the copy failed."""
        if score is not None:
            self.scores[game_copy.candidate][game_copy.redeal] = score
        self.open_copies -= 1

    @property
    def finished(self) -> bool:
        return self.open_copies == 0

    def means(self, settings: LookaheadSettings) -> list[float | None]:
        """Per candidate: the mean score of its copies, or None if fewer than half of them finished."""
        return [float(np.mean(list(scores.values()))) if len(scores) * 2 >= settings.redeals else None for scores in self.scores]

    def paired_scores(self, settings: LookaheadSettings) -> list[list[float | None]]:
        """Per candidate, its score on each re-deal in order (None where that copy failed): comparing candidates
        re-deal by re-deal removes the luck the re-deals share."""
        return [[scores.get(redeal) for redeal in range(settings.redeals)] for scores in self.scores]

    def best(self, settings: LookaheadSettings) -> int:
        """The chosen candidate index (0 = the network's own choice)."""
        means = self.means(settings)
        if means[0] is None:
            return 0
        best = max(range(len(means)), key=lambda index: means[index] if means[index] is not None else -np.inf)
        return best if means[best] - means[0] >= settings.margin else 0


def candidate_choices(own_choice: int, probabilities: np.ndarray, settings: LookaheadSettings,
                      move_kinds: list[str] | None = None) -> list[int]:
    """Choice indices to try: the network's own choice first, then its most likely others, then (with
    settings.every_move_kind and move_kinds, each choice's action kind) the most likely move of every kind not tried
    yet, however unlikely. Empty = don't search."""
    if len(probabilities) < 2 or probabilities.max() >= settings.confident_probability:
        return []
    ranked = [int(index) for index in np.argsort(-probabilities)]
    likely = [index for index in ranked[:settings.candidates] if probabilities[index] >= settings.minimum_probability]
    others = [index for index in likely if index != own_choice][:settings.candidates - 1]
    if settings.every_move_kind and move_kinds is not None:
        kinds_tried = {move_kinds[index] for index in [own_choice] + others}
        for index in ranked:
            if move_kinds[index] not in kinds_tried:
                others.append(index)
                kinds_tried.add(move_kinds[index])
    return [own_choice] + others if others else []


def new_search(decision: Decision, recorder: MatchRecorder, responses: list[dict]) -> DecisionSearch:
    return DecisionSearch(decision, recorder, responses, scores=[{} for _ in responses])
