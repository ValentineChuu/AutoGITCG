"""Settings for building the ground-truth set: the true best move at sampled decisions (experiments/tree_search/notes.md)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GroundTruthSettings:
    likely_moves: int = 6  # the network's most likely moves compared, plus its most likely move of every other kind
    deals_in_flight: int = 64  # deals of one position running at once (each deal is played once with every compared move)
    minimum_deals: int = 64  # completed deals before a position may stop early
    max_deals: int = 512  # stop here even if the best move isn't clearly ahead (also stops: every move had
    # identical results on every deal so far, e.g. a lost position)
    separation: float = 3.0  # stop once the best move leads the runner-up by this many standard errors (paired)
    sample_probability: float = 0.12  # chance that an unsure decision of the sampled seat becomes the game's position
    unsure_probability: float = 0.9  # only decisions whose top move the network gives less than this
    max_open_copies: int = 1000  # game copies running at once in this process
