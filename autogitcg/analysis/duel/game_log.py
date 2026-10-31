"""A whole game, move by move, in a .jsonl.gz file (one JSON object per line); scripts/evaluation/judge_duel.py writes one per game.

- header: seed, decks and equipment per seat, the players' names and settings. With the seed and every response the
  game replays exactly (the engine is seeded and decks are shuffled from the seed). Answers given to decisions asked
  of both seats at once (rerolls, opening hands) must be replayed in the logged order.
- move: every answered decision of either seat: who, human or bot, the player's own view of the state, the request
  (the options offered), the response and the chosen action, seconds since the game started, a person's thinking
  time, and for searched moves each candidate's choice index and mean score.
- end: winner and rounds, or how it ended early (conceded, left, engine failure).
"""

import gzip
import json
import time
from pathlib import Path

from autogitcg.bridge.messages import Decision
from autogitcg.search.lookahead_table import SearchResult


class GameLog:
    def __init__(self, path: Path, header: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = gzip.open(path, "wt")
        self.started = time.monotonic()
        self.moves = 0
        self.asked_at: dict[int, float] = {}  # who -> when the person's pending decision arrived
        self._write({"type": "header", **header})

    def on_asked(self, decision: Decision) -> None:
        self.asked_at[decision.who] = time.monotonic()

    def on_move(self, decision: Decision, response_value: dict, search: SearchResult | None, human: bool) -> dict:
        """Writes the move and returns its record."""
        now = time.monotonic()
        asked = self.asked_at.pop(decision.who, None) if human else None
        chosen = (decision.request["action"][response_value["chosenActionIndex"]]["action"]
                  if decision.kind == "action" else None)
        record = {"type": "move", "index": self.moves, "who": decision.who, "actor": "human" if human else "bot",
                  "kind": decision.kind, "seconds": round(now - self.started, 3),
                  "thinking_seconds": round(now - asked, 3) if asked is not None else None,
                  "state": decision.state, "request": decision.request, "response": response_value, "chosen": chosen,
                  "search": {"candidates": search.candidate_indices, "mean_scores": search.mean_scores} if search else None}
        self._write(record)
        self.moves += 1
        return record

    def end(self, result: dict) -> None:
        if self.file.closed:
            return
        self._write({"type": "end", "moves": self.moves, "seconds": round(time.monotonic() - self.started, 3), **result})
        self.file.close()

    def _write(self, record: dict) -> None:
        self.file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.file.flush()
