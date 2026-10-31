"""Build the model release: every trained network as a small frozen file (no optimizer state), plus manifest.json.

Reads the run folders named in model_release_plan.json (next to this script):
- combat runs: the network at every 50-update milestone, from <run>/weights/update_NNNNN.pt and the run's network
  config. Named "<family> <run>.<update / 50>": theta 1 at update 650 is theta-1.13.pt. A run that stopped before its
  first milestone gives its last network instead, named with the updates past the last milestone: eta 6 stopped at
  update 30, so its network is eta-6.0.30.pt;
- students: their final epoch;
- Chudledee: every saved checkpoint (one per 50 updates), named the same way: chudledee v1 at update 300 is
  chudledee-v1.6.pt. Chudledee's last milestone is that version's final network (v1.6, v2.4, v3.4).

This file is the export from a training machine's run folders, which this repository doesn't include; to use a
single network of your own, scripts/training/freeze_combat_network.py is enough.

Every written file is loaded back into its network before it is listed, so a broken file can't reach the release; a
run whose networks the current code can't load is left out and listed under "not_loadable" with the reason.

    python scripts/training/export_model_release.py --runs runs --output ../AutoGITCG-models
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch

from autogitcg.deck_builder.builder_setup import BuilderSetup
from autogitcg.model.combat_network_file import load_combat_network
from autogitcg.observation.observation_encoder import ObservationEncoder

PLAN = Path(__file__).resolve().parent / "model_release_plan.json"
MILESTONE = 50


def versioned_name(base: str, update: int) -> str:
    """("theta 1", 650) -> "theta 1.13"; ("eta 6", 30) -> "eta 6.0.30" (30 updates past milestone 0)."""
    milestone, past = divmod(update, MILESTONE)
    return f"{base}.{milestone}" + (f".{past}" if past else "")


def file_name(name: str) -> str:
    """"theta 1.13" -> "theta-1.13.pt"."""
    return name.replace(" ", "-") + ".pt"


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def network_config(run: Path) -> dict:
    """The run's network config, from any of its checkpoints (all of a run's networks share it)."""
    checkpoint = next(iter(sorted(run.glob("checkpoint_*.pt"))), None)
    if checkpoint is None:
        raise FileNotFoundError(f"{run} has no checkpoint to read the network config from")
    return torch.load(checkpoint, map_location="cpu", weights_only=False)["network_config"]


def combat_milestones(entry: dict, runs: Path, folder: Path, encoder: ObservationEncoder) -> list[dict]:
    run = runs / entry["run"]
    config = network_config(run)
    rows = []
    saved = sorted((run / "weights").glob("update_*.pt"))
    updates = [int(re.search(r"update_(\d+)", weights.name).group(1)) for weights in saved]
    chosen = [(weights, update) for weights, update in zip(saved, updates) if update and not update % MILESTONE]
    if not chosen and updates and updates[-1]:
        chosen = [(saved[-1], updates[-1])]  # a run that stopped before its first milestone: its last network
    for weights, update in chosen:
        name = versioned_name(entry["name"], update)
        path = folder / file_name(name)
        torch.save({"network": torch.load(weights, map_location="cpu", weights_only=False), "network_config": config,
                    "progress": {"update": update}, "frozen_from": f"{entry['name']} update {update}"}, path)
        load_combat_network(path, encoder, torch.device("cpu"))
        rows.append({"name": name, "file": path.name, "kind": "combat", "family_run": entry["name"],
                     "update": update, "idea": entry["idea"], "note": entry.get("headline", {}).get(str(update // MILESTONE))})
    return rows


def student(entry: dict, runs: Path, folder: Path, encoder: ObservationEncoder) -> list[dict]:
    epochs = sorted((runs / entry["run"]).glob("student_epoch_*.pt"))
    if not epochs:
        return []
    checkpoint = torch.load(epochs[-1], map_location="cpu", weights_only=False)
    path = folder / file_name(entry["name"])
    torch.save({"network": checkpoint["network"], "network_config": checkpoint["network_config"],
                "progress": checkpoint.get("progress", {"update": 0}), "frozen_from": f"{entry['name']}, epoch {len(epochs)}"}, path)
    load_combat_network(path, encoder, torch.device("cpu"))
    return [{"name": entry["name"], "file": path.name, "kind": "combat (student)", "family_run": entry["name"],
             "epoch": len(epochs), "idea": entry["idea"], "note": None}]


def chudledee(entry: dict, runs: Path, folder: Path, setup: BuilderSetup) -> list[dict]:
    rows = []
    checkpoints = sorted((runs / entry["run"]).glob("checkpoint_[0-9]*.pt"))
    for source in checkpoints:
        update = int(re.search(r"checkpoint_(\d+)", source.name).group(1))
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        frozen = {"builder": checkpoint["builder"], "builder_config": checkpoint["builder_config"],
                  "progress": checkpoint.get("progress", {"update": update}), "frozen_from": f"{entry['name']} update {update}"}
        name = versioned_name(entry["name"], update)
        path = folder / file_name(name)
        torch.save(frozen, path)
        setup.network_from(torch.load(path, map_location="cpu", weights_only=False))
        rows.append({"name": name, "file": path.name, "kind": "deck builder", "family_run": entry["name"],
                     "update": update, "idea": entry["idea"],
                     "note": f"{entry['name']}'s final network" if source == checkpoints[-1] else None})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    encoder, setup = ObservationEncoder(), BuilderSetup()
    rows, missing, unloadable = [], [], []
    for section, export in (("combat", combat_milestones), ("students", student), ("chudledee", chudledee)):
        for entry in plan[section]:
            folder = arguments.output  # flat: the release unpacks straight into models/, where the scripts look
            folder.mkdir(parents=True, exist_ok=True)
            if not (arguments.runs / entry["run"]).exists():
                missing.append(entry["run"])
                continue
            try:
                made = export(entry, arguments.runs, folder, setup if section == "chudledee" else encoder)
            except RuntimeError as error:  # the current code can't load it (e.g. an output whose shape changed since)
                for written in folder.glob(file_name(entry["name"]).removesuffix(".pt") + "[.-]*.pt"):
                    written.unlink()
                unloadable.append({"run": entry["name"], "reason": str(error).split(": ", 1)[-1][:300]})
                print(f"{entry['name']:14} not loadable by the current code, left out", flush=True)
                continue
            rows += made
            print(f"{entry['name']:14} {len(made):3} files", flush=True)
    for row in rows:
        path = arguments.output / row["file"]
        row |= {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (arguments.output / "manifest.json").write_text(json.dumps({"networks": rows, "runs_not_found": missing, "not_loadable": unloadable}, indent=1))
    print(f"{len(rows)} files, {sum(row['bytes'] for row in rows) / 1024**2:.0f} MB; runs not found: {missing}")


if __name__ == "__main__":
    main()
