"""Answers a batch of decisions that belong to different networks: one batched call per network, choices in order."""

from autogitcg.policies.network_policy import NetworkChoice, NetworkPolicy


def choose_grouped(pending: list, policy_of) -> list[NetworkChoice]:
    """pending: (decision, encoded) pairs; policy_of(decision) -> the NetworkPolicy that answers it."""
    positions_by_policy: dict[int, tuple[NetworkPolicy, list[int]]] = {}
    for position, (decision, _) in enumerate(pending):
        policy = policy_of(decision)
        positions_by_policy.setdefault(id(policy), (policy, []))[1].append(position)
    choices: list[NetworkChoice | None] = [None] * len(pending)
    for policy, positions in positions_by_policy.values():
        for position, choice in zip(positions, policy.choose([pending[position] for position in positions])):
            choices[position] = choice
    return choices
