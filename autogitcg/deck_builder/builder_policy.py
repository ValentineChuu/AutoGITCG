"""Builds decks with the builder network: all tasks advance in lockstep, one batched forward pass per pick.

Each pick is recorded as a BuildingStep (observation, choice, log-probability, value estimate) so training can
credit every pick with the finished deck's reward.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from autogitcg.data.deck import Deck
from autogitcg.deck_builder.builder_network import BuilderNetwork
from autogitcg.deck_builder.building_batch import pad_and_stack
from autogitcg.deck_builder.building_encoder import BuildingEncoder
from autogitcg.deck_builder.building_state import BuildingState
from autogitcg.deck_builder.building_task import BuildingTask
from autogitcg.deck_builder.token_budget_batches import observation_length, token_budget_batches
from autogitcg.model.observation_batch import compact, to_device

INFERENCE_TOKEN_BUDGET = 131072  # padded tokens per forward pass (no gradients kept)


@dataclass
class BuildingStep:
    observation: dict[str, np.ndarray]  # compact storage, natural lengths
    choice_index: int  # position among the candidate tokens
    log_prob: float
    value: float


@dataclass
class BuiltDeck:
    task: BuildingTask
    deck: Deck
    steps: list[BuildingStep] = field(default_factory=list)
    entropy: float = 0.0  # mean over the picks


class BuilderPolicy:
    def __init__(self, network: BuilderNetwork, encoder: BuildingEncoder, device: torch.device, greedy: bool = False):
        self.network = network
        self.encoder = encoder
        self.device = device
        self.greedy = greedy

    @torch.no_grad()
    def build(self, tasks: list[BuildingTask], generator: torch.Generator | None = None,
              started: list[BuildingState] | None = None) -> list[BuiltDeck]:
        """started: optional decks already partly picked (one per task, e.g. a guess forced to hold the cards an
        opponent has shown); the builder makes only the remaining picks."""
        self.network.eval()
        states = started if started is not None else [BuildingState(task) for task in tasks]
        steps: list[list[BuildingStep]] = [[] for _ in tasks]
        entropy_totals = np.zeros(len(tasks))
        while unfinished := [row for row, state in enumerate(states) if not state.finished]:
            candidates = {row: states[row].candidates() for row in unfinished}
            observations = {row: self.encoder.encode(states[row], candidates[row]) for row in unfinished}
            lengths = [observation_length(observations[row]) for row in unfinished]
            for batch_rows in token_budget_batches(lengths, INFERENCE_TOKEN_BUDGET):
                rows = [unfinished[index] for index in batch_rows]
                chosen, chosen_log_probs, values, entropy = self._pick([observations[row] for row in rows], generator)
                for position, row in enumerate(rows):
                    states[row].apply(candidates[row][chosen[position]])
                    steps[row].append(BuildingStep(compact(observations[row]), int(chosen[position]),
                                                   float(chosen_log_probs[position]), float(values[position])))
                    entropy_totals[row] += entropy[position]
        return [BuiltDeck(task, state.deck(), task_steps, float(entropy_totals[row] / max(len(task_steps), 1)))
                for row, (task, state, task_steps) in enumerate(zip(tasks, states, steps))]

    def _pick(self, observations: list[dict[str, np.ndarray]], generator: torch.Generator | None) -> tuple[np.ndarray, ...]:
        """(chosen candidate, its log-probability, value estimate, entropy) per observation."""
        batch = to_device(pad_and_stack(observations), self.device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            output = self.network(batch)
        log_probabilities = torch.log_softmax(output.candidate_logits, dim=1)
        probabilities = log_probabilities.exp()
        if self.greedy:
            chosen = probabilities.argmax(dim=1)
        else:
            chosen = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
        chosen_log_probs = log_probabilities.gather(1, chosen.unsqueeze(1)).squeeze(1)
        entropy = -(probabilities * log_probabilities.nan_to_num(neginf=0.0)).sum(dim=1)
        return (chosen.cpu().numpy(), chosen_log_probs.cpu().numpy(), output.value.cpu().numpy(), entropy.cpu().numpy())
