"""Win-chance points a move loses against a stored position's best move, without the winner's curse: the best move
by measured mean is partly the luckiest one, so the best is chosen on the even deals and the regret measured on the
odd deals (only deals where every move has a result)."""


def held_out_regret(line: dict, move: int) -> float | None:
    results = line["results"]  # per move, per deal: +1, -1, or 0 (no result)
    deals = range(len(results[0]))
    complete = [deal for deal in deals if all(results[other][deal] != 0 for other in range(len(results)))]
    choose, measure = complete[0::2], complete[1::2]
    if len(choose) < 8 or len(measure) < 8:
        return None
    best = max(range(len(results)), key=lambda other: sum(results[other][deal] for deal in choose))
    return sum(results[best][deal] - results[move][deal] for deal in measure) / len(measure) / 2 * 100
