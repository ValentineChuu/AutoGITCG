"""Training pieces: advantages, the choice distribution, padding invariance, and network play through the bridge."""

import dataclasses
import random
from collections import Counter

import numpy as np
import pytest
import torch

from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.bridge.messages import Decision, GameFailed, GameFinished
from autogitcg.data.npc_decks import load_npc_decks
from autogitcg.model.combat_network import CombatNetwork
from autogitcg.model.network_config import NetworkConfig, network_config_for
from autogitcg.model.observation_batch import compact, stack, to_device
from autogitcg.model.policy_distribution import categorical_logits, evaluate_choices, sample_choices
from autogitcg.observation.enums import DECISION_KINDS
from autogitcg.observation.match_recorder import MatchRecorder
from autogitcg.observation.observation_encoder import ObservationEncoder
from autogitcg.policies.heuristic_policy import HeuristicPolicy
from autogitcg.policies.network_policy import NetworkPolicy
from autogitcg.policies.random_policy import random_response
from autogitcg.training.checkpoints import network_config_from
from autogitcg.training.ppo_config import PPOConfig
from autogitcg.training.ppo_update import ppo_update
from autogitcg.training.trajectory import Step, Trajectory

CPU = torch.device("cpu")


def _step(value: float) -> Step:
    return Step({}, 0, 0, np.zeros(16, dtype=np.float32), 0.0, value)


def test_advantages_with_full_lambda_are_reward_minus_value():
    trajectory = Trajectory([_step(0.2), _step(-0.5), _step(0.9)])
    steps = trajectory.finish(1.0, True, gae_lambda=1.0)
    assert [round(step.advantage, 6) for step in steps] == [0.8, 1.5, 0.1]
    assert all(abs(step.value_target - 1.0) < 1e-6 and step.won == 1.0 for step in steps)


def test_advantages_with_zero_lambda_are_one_step_differences():
    steps = Trajectory([_step(0.2), _step(-0.5), _step(0.9)]).finish(-1.0, False, gae_lambda=0.0)
    assert [round(step.advantage, 6) for step in steps] == [-0.7, 1.4, -1.9]


@pytest.fixture(scope="module")
def encoded_decisions():
    """(decision, encoded) pairs from a few random games, covering every decision kind that occurs."""
    rng = random.Random(5)
    decks = list(load_npc_decks().values())
    encoder = ObservationEncoder()
    recorders, pairs = {}, []
    with EnginePool(worker_count=4) as pool:
        for seed in range(6):
            deck_pair = tuple(rng.sample(decks, 2))
            recorders[pool.create_game(*deck_pair, seed=seed)] = MatchRecorder(deck_pair, (True, False))
        while pool.has_active_games():
            for event in pool.poll():
                if isinstance(event, Decision):
                    pairs.append((event, encoder.encode(event, recorders[event.game_id].on_decision(event))))
                    response = random_response(event, rng)
                    recorders[event.game_id].on_response(event, response)
                    pool.respond(event, response)
    return encoder, pairs


@pytest.fixture(scope="module")
def network(encoded_decisions):
    encoder, _ = encoded_decisions
    torch.manual_seed(0)
    return CombatNetwork(network_config_for(encoder, width=32, layers=2), encoder.catalog_features.table).eval()


def _batch(pairs):
    batch = to_device(stack([compact(encoded.arrays) for _, encoded in pairs]), CPU)
    kinds = torch.tensor([DECISION_KINDS.index(decision.kind) for decision, _ in pairs])
    return batch, kinds


def test_sampled_choices_are_legal_and_log_probs_match(encoded_decisions, network):
    _, pairs = encoded_decisions
    assert {decision.kind for decision, _ in pairs} >= {"action", "rerollDice", "switchHands", "chooseActive"}
    batch, kinds = _batch(pairs)
    with torch.no_grad():
        output = network(batch)
        sampled = sample_choices(output, batch, kinds)
        log_prob, entropy = evaluate_choices(output, batch, kinds, sampled.choice_index, sampled.choice_bits)
    assert torch.allclose(log_prob, sampled.log_prob, atol=1e-5)
    assert torch.isfinite(log_prob).all() and (entropy >= -1e-6).all()
    categorical = (kinds != DECISION_KINDS.index("rerollDice")) & (kinds != DECISION_KINDS.index("switchHands"))
    chosen_mask = batch["action_mask"].gather(1, sampled.choice_index.unsqueeze(1)).squeeze(1)
    assert (chosen_mask[categorical] > 0).all()
    rerolls = kinds == DECISION_KINDS.index("rerollDice")
    assert (sampled.choice_bits[rerolls] <= batch["dice_mask"][rerolls]).all()


def test_padding_does_not_change_outputs(encoded_decisions, network):
    """A row scored alone (trimmed to its own length) must match the same row inside a batch of longer rows."""
    _, pairs = encoded_decisions
    batch, kinds = _batch(pairs[:64])
    shortest = int(batch["entity_mask"].sum(1).argmin())
    alone, _ = _batch(pairs[shortest:shortest + 1])
    with torch.no_grad():
        together_output, alone_output = network(batch), network(alone)
    assert torch.allclose(together_output.value[shortest], alone_output.value[0], atol=1e-4)
    real = alone["action_mask"][0] > 0
    assert torch.allclose(together_output.action_logits[shortest][real], alone_output.action_logits[0][real], atol=1e-4)


def test_network_plays_complete_games(encoded_decisions, network):
    encoder, _ = encoded_decisions
    policy = NetworkPolicy(network, CPU)
    rng = random.Random(9)
    decks = list(load_npc_decks().values())
    recorders, outcomes = {}, Counter()
    with EnginePool(worker_count=4) as pool:
        for seed in range(8):
            deck_pair = tuple(rng.sample(decks, 2))
            recorders[pool.create_game(*deck_pair, seed=seed)] = MatchRecorder(deck_pair, (True, True))
        while pool.has_active_games():
            pending = []
            for event in pool.poll():
                if isinstance(event, Decision):
                    pending.append((event, encoder.encode(event, recorders[event.game_id].on_decision(event))))
                else:
                    outcomes[type(event).__name__] += 1
            for (decision, _), choice in zip(pending, policy.choose(pending)):
                recorders[decision.game_id].on_response(decision, choice.response)
                pool.respond(decision, choice.response)
    assert outcomes == Counter({GameFinished.__name__: 8}), outcomes
    assert GameFailed.__name__ not in outcomes


def test_heuristic_beats_random():
    rng = random.Random(3)
    decks = list(load_npc_decks().values())
    heuristic = HeuristicPolicy()
    heuristic_seat, wins = {}, 0
    with EnginePool(worker_count=4) as pool:
        for seed in range(40):
            heuristic_seat[pool.create_game(*rng.sample(decks, 2), seed=seed)] = seed % 2
        while pool.has_active_games():
            for event in pool.poll():
                if isinstance(event, Decision):
                    mine = event.who == heuristic_seat[event.game_id]
                    pool.respond(event, heuristic.respond(event, rng) if mine else random_response(event, rng))
                elif isinstance(event, GameFinished):
                    wins += event.winner == heuristic_seat[event.game_id]
    assert wins >= 32, wins


def test_identity_embeddings_start_small(network):
    weights = network.definition_embedding.weight.detach()
    assert float(weights[0].abs().max()) == 0.0
    assert float(weights[1:].norm(dim=1).mean()) < 0.1 * weights.shape[1] ** 0.5


def test_speed_reward_prefers_fast_wins_and_never_rewards_losing():
    from autogitcg.bridge.messages import GameFinished
    from autogitcg.training.game_result import final_reward
    fast, slow = GameFinished(0, 0, 4), GameFinished(0, 0, 9)
    assert final_reward(fast, 0) > final_reward(slow, 0) > 0
    assert final_reward(fast, 1) == final_reward(slow, 1) == -1.0
    assert final_reward(GameFinished(0, None, 15), 0) == -1.0


def test_matchups_record_deck_origins_and_swap_consistently():
    from autogitcg.training.collection_pool import simulate_collections
    from autogitcg.training.deck_pools import load_deck_pools
    from autogitcg.training.matchup_sampler import MatchupSampler, swapped
    from autogitcg.progression.stage_catalog import build_stage_catalog
    stages, _ = build_stage_catalog()
    sampler = MatchupSampler(load_deck_pools(simulate_collections(stages, 2, 2, seed=0)), random.Random(1))
    matchups = [sampler.sample() for _ in range(400)]
    assert {matchup.source for matchup in matchups} == {"stage", "synergy", "npc_decks", "collections", "tournament", "chudledee"}
    origin_kinds = {"collection", "npc_deck", "stage", "synergy", "tournament", "chudledee", "chudledee_stage"}
    for matchup in matchups:
        assert matchup.decks[matchup.focus_seat] is not None and matchup.category.split("/")[0] == matchup.source
        assert all(origin.split(":")[0] in origin_kinds for origin in matchup.deck_origins)
        if matchup.category.endswith("/chudledee"):  # Chudledee's stage deck was built for this very stage
            player, stage = sorted(matchup.deck_origins)
            assert player.split(":")[1] == stage.split(":")[1]
        if matchup.equipment:  # pre-equipped talents belong to the stage side
            stage_seat = next(seat for seat, origin in enumerate(matchup.deck_origins) if origin.startswith("stage:"))
            assert all(item.who == stage_seat for item in matchup.equipment)
            assert all(item.who != stage_seat for item in swapped(matchup).equipment)


def test_adaptive_difficulty_shifts_weight_to_lost_categories():
    from autogitcg.training.adaptive_difficulty import AdaptiveDifficulty
    difficulty = AdaptiveDifficulty({"easy": 0.5, "hard": 0.5})
    assert difficulty.weights() == pytest.approx({"easy": 0.5, "hard": 0.5})  # equal skill keeps the base mix
    for _ in range(300):
        difficulty.record("easy", 1.0)
        difficulty.record("hard", 0.0)
    weights = difficulty.weights()
    assert abs(sum(weights.values()) - 1) < 1e-9
    assert weights["hard"] > 0.7 and weights["easy"] >= 0.25


def test_tournament_test_lineups_never_reach_training():
    from autogitcg.data.tournament_test_set import hashed_into_test, lineup_of, test_lineups
    from autogitcg.training.deck_pools import load_training_tournament_decks
    locked = test_lineups()
    assert locked and all(hashed_into_test(lineup) for lineup in locked)
    training = load_training_tournament_decks()
    assert training and not {lineup_of(deck) for deck in training} & locked


def test_frozen_benchmark_matchups_load():
    from autogitcg.deck_building.deck_validation import deck_problems
    from autogitcg.training.benchmark_matchups import load_benchmark_matchups
    matchups = load_benchmark_matchups()
    assert len(matchups) == 150
    first = matchups[0]
    assert first.deck_origins == ("npc_deck:130204", "synergy:0")  # as recorded in alpha 2's benchmark
    synergy_decks = [deck for matchup in matchups for deck, origin in zip(matchup.decks, matchup.deck_origins)
                     if origin.startswith("synergy:")]
    assert synergy_decks and all(not deck_problems(deck) for deck in synergy_decks)


def test_kickstarting_divergence_is_zero_only_for_the_teacher_itself(encoded_decisions, network):
    from autogitcg.training.kickstarting import kickstarting_loss, teacher_divergence
    encoder, pairs = encoded_decisions
    batch, kinds = _batch(pairs)
    torch.manual_seed(1)
    other = CombatNetwork(network_config_for(encoder, width=32, layers=2), encoder.catalog_features.table).eval()
    with torch.no_grad():
        teacher_output, other_output = network(batch), other(batch)
        same = teacher_divergence(teacher_output, teacher_output, batch, kinds)
        different = teacher_divergence(other_output, teacher_output, batch, kinds)
        divergence, value_difference = kickstarting_loss(other_output, teacher_output, batch, kinds)
    assert same.abs().max() < 1e-5
    # Two random networks both output near-uniform choices, so how far apart they are depends on the initialization;
    # the point is only that a different network isn't at zero.
    assert (different >= -1e-5).all() and different.mean() > 1e-4
    assert torch.isfinite(divergence) and value_difference > 0


def test_networks_saved_before_appended_features_load_unchanged(encoded_decisions, network):
    """Features appended to a group later (e.g. plunging, history order) get zero input weights when an older network
    loads, so it plays exactly as when it was saved (network_config.grown_feature_widths)."""
    encoder, pairs = encoded_decisions
    old_widths = {**network.config.feature_widths, "global": network.config.feature_widths["global"] - 1,
                  "history": network.config.feature_widths["history"] - 1}
    old_config = NetworkConfig(network.config.vocabulary_size, network.config.catalog_width, old_widths, 32, 2)
    torch.manual_seed(1)
    old_network = CombatNetwork(old_config, encoder.catalog_features.table).eval()
    grown = CombatNetwork(network_config_from({"network_config": dataclasses.asdict(old_config)}), encoder.catalog_features.table).eval()
    grown.load_state_dict(old_network.state_dict())
    batch, _ = _batch(pairs[:64])
    old_batch = {**batch, "global_features": batch["global_features"][..., :-1], "history_features": batch["history_features"][..., :-1]}
    with torch.no_grad():
        assert torch.allclose(grown(batch).action_logits, old_network(old_batch).action_logits, atol=1e-5, equal_nan=True)
        assert torch.allclose(grown(batch).value, old_network(old_batch).value, atol=1e-5)


def test_ppo_update_measures_kl_from_the_start_of_the_update(encoded_decisions):
    """The early stop compares with the policy at the start of the update, not the (stale) behaviour policy, so even
    steps recorded with very different log-probabilities don't stop an update that hasn't moved yet."""
    encoder, pairs = encoded_decisions
    torch.manual_seed(2)
    learner = CombatNetwork(network_config_for(encoder, width=32, layers=2), encoder.catalog_features.table)
    pending = pairs[:64]
    choices = NetworkPolicy(learner, CPU).choose(pending)
    steps = [Step(compact(encoded.arrays), choice.kind, choice.choice_index, choice.choice_bits, choice.log_prob - 1.0,
                  choice.value, 0, advantage=0.0, value_target=0.0) for (_, encoded), choice in zip(pending, choices)]
    config = PPOConfig(epochs=1, minibatch_size=64, learning_rate=0.0, target_kl=0.01)
    optimizer = torch.optim.Adam(learner.parameters(), lr=0.0)
    metrics = ppo_update(learner, optimizer, steps, config, CPU, np.random.default_rng(0))
    assert metrics["approx_kl"] > 0.1  # behaviour log-probs were shifted by 1
    assert abs(metrics["update_kl"]) < 1e-3  # no learning rate: the policy didn't move
    assert metrics["epochs_completed"] == 1


def test_exploration_samples_unlikely_moves_with_behaviour_log_probs(encoded_decisions, network):
    """With exploration, every legal action has at least exploration / legal count probability, and the returned
    log-probability is the mixture's (what PPO must correct for)."""
    _, pairs = encoded_decisions
    actions = [(decision, encoded) for decision, encoded in pairs if decision.kind == "action"][:64]
    batch, kinds = _batch(actions)
    with torch.no_grad():
        output = network(batch)
        sampled = sample_choices(output, batch, kinds, exploration=0.5)
    legal = batch["action_mask"] > 0
    policy = categorical_logits(output, batch).softmax(dim=1)
    behaviour = 0.5 * policy + 0.5 * legal.float() / legal.float().sum(dim=1, keepdim=True)
    expected = behaviour.gather(1, sampled.choice_index.unsqueeze(1)).squeeze(1).log()
    assert torch.allclose(sampled.log_prob, expected, atol=1e-5)
    own = policy.gather(1, sampled.choice_index.unsqueeze(1)).squeeze(1).log()
    assert torch.allclose(sampled.policy_log_prob, own, atol=1e-5)  # PPO's ratio uses this one
    assert bool(legal.gather(1, sampled.choice_index.unsqueeze(1)).all())


def test_card_play_bonus_adds_to_the_return_of_its_step_and_earlier_ones():
    """With λ = 1 and zero values, each step's return is the final reward plus the bonuses from that step on."""
    steps = [Step({}, 0, 0, np.zeros(1), 0.0, 0.0, bonus=bonus) for bonus in (0.0, 0.01, 0.0, 0.01)]
    finished = Trajectory(steps).finish(1.0, True, gae_lambda=1.0)
    assert [step.advantage for step in finished] == pytest.approx([1.02, 1.02, 1.01, 1.01])
    assert all(step.result == 1.0 for step in finished)


def test_card_play_bonus_adds_the_kind_bonus_for_played_cards_only():
    from autogitcg.bridge.messages import Decision
    from autogitcg.data.card_kinds import card_kind
    from autogitcg.training.card_play_bonus import card_play_bonus
    resonance = next(card for card in range(331000, 332000) if card_kind(card) == "resonances")
    options = [{"action": {"$case": "playCard", "value": {"cardDefinitionId": resonance}}},
               {"action": {"$case": "declareEnd", "value": {}}}]
    decision = Decision(0, 0, "action", {"action": options}, {}, {})
    kinds = {"resonances": 0.02, "events": 0.5}
    assert card_play_bonus(decision, {"chosenActionIndex": 0}, 0.01, kinds) == pytest.approx(0.03)
    assert card_play_bonus(decision, {"chosenActionIndex": 1}, 0.01, kinds) == 0.0
    assert card_play_bonus(decision, {"chosenActionIndex": 0}, 0.0) == 0.0


def test_reward_shaping_telescopes_to_the_first_potential():
    """Potential-based shaping: with λ = 1 and zero values, the first step's return is the final reward minus
    weight x its own potential (the shaping in between cancels), and the last step sees weight x (0 - its potential)."""
    steps = [Step({}, 0, 0, np.zeros(1), 0.0, 0.0, potential=potential) for potential in (0.0, 0.3, -0.2, 0.5)]
    finished = Trajectory(steps).finish(1.0, True, gae_lambda=1.0, shaping_weight=0.5)
    assert finished[0].advantage == pytest.approx(1.0 - 0.5 * 0.0)
    assert finished[3].advantage == pytest.approx(1.0 + 0.5 * (0.0 - 0.5))
    assert finished[1].advantage == pytest.approx(1.0 - 0.5 * 0.3)


def test_training_tournament_decks_can_start_at_a_version():
    from autogitcg.training.deck_pools import load_training_tournament_decks

    every = load_training_tournament_decks()
    recent = load_training_tournament_decks("6-4")
    assert 0 < len(recent) < len(every)
    assert {tuple(deck.cards) for deck in recent} <= {tuple(deck.cards) for deck in every}
