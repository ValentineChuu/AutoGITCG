"""A game between two networks as readable text: the board at each round's first move, every move with who made it,
and for searched moves what the search tried (candidates and their mean scores, the network's own choice first)."""

from autogitcg.analysis.game_text.board_description import area_lines, character_lines
from autogitcg.analysis.game_text.option_labels import option_labels
from autogitcg.bridge.messages import Decision

SHOWN_KINDS = ("action", "chooseActive", "selectCard")


class ReadableGame:
    def __init__(self, title: str, names: dict[int, str]):
        """names: seat -> the network's name."""
        self.names = names
        self.lines = [title, ""]
        self.round = None

    def on_move(self, decision: Decision, response_value: dict, search) -> None:
        if decision.kind not in SHOWN_KINDS:
            return
        state = decision.state
        if decision.kind == "action" and state.get("roundNumber") != self.round:
            self.round = state.get("roundNumber")
            self.lines += ["", f"--- Round {self.round} ---"]
            for seat in (decision.who, 1 - decision.who):
                self.lines.append(f"{self.names[seat]} (seat {seat}):")
                player = state["player"][seat]
                self.lines += character_lines(player, decision.counters) + area_lines(player, decision.counters)
        labels = option_labels(decision.kind, decision.request, state, decision.who)
        played = self._choice_index(decision, response_value)
        line = f"{self.names[decision.who]}: {labels[played] if played is not None and played < len(labels) else decision.kind}"
        if search is not None:
            tried = [f"{labels[index]} {score:+.2f}" if score is not None else f"{labels[index]} ?"
                     for index, score in zip(search.candidate_indices, search.mean_scores)]
            line += f"\n      searched: " + " | ".join(tried)
            if played != search.candidate_indices[0]:
                line += "\n      (search overrode the network's own choice)"
        self.lines.append(line)

    def finish(self, row: dict) -> str:
        if "failure" in row:
            self.lines += ["", f"Engine failure: {row['failure']}"]
        else:
            winner = self.names.get(row.get("winner"), "nobody (draw)")
            self.lines += ["", f"Winner: {winner} after {row.get('rounds')} rounds"]
        return "\n".join(self.lines) + "\n"

    @staticmethod
    def _choice_index(decision: Decision, response_value: dict) -> int | None:
        """The played option's position among the labels (valid actions in request order, or candidates)."""
        request = decision.request
        if decision.kind == "action":
            chosen = response_value["chosenActionIndex"]
            valid = [index for index, action in enumerate(request["action"]) if action.get("validity", 0) == 0]
            return valid.index(chosen) if chosen in valid else None
        if decision.kind == "chooseActive":
            return request["candidateIds"].index(response_value["activeCharacterId"])
        return request["candidateDefinitionIds"].index(response_value["selectedDefinitionId"])
