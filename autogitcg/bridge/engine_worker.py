"""One engine worker process, talking newline-delimited JSON over stdio: the C++ engine (engine/, build: make -C engine
build/release/worker), or with AUTOGITCG_ENGINE=node its JavaScript reference (engine/reference/: same protocol,
identical games, 20-40 times slower)."""

import json
import os
import queue
import subprocess
import threading
from pathlib import Path

import orjson

from autogitcg.bridge.messages import Event, GameFailed, parse_event

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKER_SCRIPT = PROJECT_ROOT / "engine" / "reference" / "bridge" / "worker.mjs"
CPP_WORKER = PROJECT_ROOT / "engine" / "build" / "release" / "worker"


def worker_command() -> list[str]:
    """The environment variable reaches spawned actor processes too, so one setting covers a whole run."""
    engine = os.environ.get("AUTOGITCG_ENGINE", "cpp")
    if engine == "cpp":
        if not CPP_WORKER.exists():
            raise FileNotFoundError(f"{CPP_WORKER} is missing: make -C engine build/release/worker")
        return [str(CPP_WORKER)]
    if engine != "node":
        raise ValueError(f"AUTOGITCG_ENGINE must be node or cpp, not {engine!r}")
    return ["node", str(WORKER_SCRIPT)]


class EngineWorker:
    def __init__(self, events: queue.Queue[Event]):
        """Start the worker; its events are pushed onto the shared `events` queue."""
        self.events = events
        self.active_game_ids: set[int] = set()
        self.closing = False  # after shutdown, a message cut off by the worker's exit is expected and ignored
        self.process = subprocess.Popen(
            worker_command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.reader = threading.Thread(target=self._read_events, daemon=True)
        self.reader.start()

    def send(self, message: dict) -> None:
        self.process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        self.process.stdin.flush()

    def _read_events(self) -> None:
        # Raw bytes into orjson: about 3x faster than json.loads on str, and this runs for every decision of every
        # game copy during look-ahead (profiled: reading messages took a sixth of a recording process's time).
        for line in self.process.stdout.buffer:
            try:
                self.events.put(parse_event(orjson.loads(line)))
            except orjson.JSONDecodeError:
                if self.closing:
                    return
                raise
        # The worker exited: fail every game it still owned so callers don't wait forever.
        for game_id in list(self.active_game_ids):
            self.events.put(GameFailed(game_id, f"engine worker exited with code {self.process.wait()}"))

    def close(self) -> None:
        self.closing = True
        if self.process.poll() is None:
            try:
                self.send({"type": "shutdown"})
            except BrokenPipeError:
                pass
        self.process.wait(timeout=10)
