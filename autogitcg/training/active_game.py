"""Bookkeeping for one running training game: who controls each seat, trackers, and learner trajectories."""

from dataclasses import dataclass, field

from autogitcg.observation.match_recorder import MatchRecorder
from autogitcg.training.matchup_sampler import Matchup
from autogitcg.training.opponent_league import SELF
from autogitcg.training.trajectory import Trajectory

LEARNER = "learner"


@dataclass
class ActiveGame:
    matchup: Matchup
    seats: tuple[str, str]  # LEARNER or an opponent name from opponent_league
    recorder: MatchRecorder
    seed: int
    snapshot_version: int | None = None  # update the snapshot opponent was taken at, for snapshot games
    pair_id: int | None = None  # reference games: both games of a mirror pair share it
    moves: list[list] = field(default_factory=list)  # game_record.move_row per move, both seats
    trajectories: dict[int, Trajectory] = field(default_factory=dict)  # learner seats only

    @property
    def opponent(self) -> str:
        """The non-learner seat's controller, or SELF for self-play."""
        return next((seat for seat in self.seats if seat != LEARNER), SELF)
