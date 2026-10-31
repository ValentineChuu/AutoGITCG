"""Which machine ran which part of a run: one line in machines.jsonl each time training starts or resumes.

Runs can move between machines (docs/cloud-training.md); this explains speed changes in the timeline.
"""

import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path

import torch


def record_machine(run_directory: Path, starting_update: int) -> None:
    run_directory.mkdir(parents=True, exist_ok=True)
    entry = {
        "starting_update": starting_update,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "logical_cores": os.cpu_count(),
    }
    with (run_directory / "machines.jsonl").open("a") as history:
        history.write(json.dumps(entry) + "\n")
