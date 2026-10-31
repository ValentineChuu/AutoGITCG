"""Label files: each recording process writes its labels in shards (gzipped pickle lists of TeacherLabel) under
<run>/labels/, and the parent lists every shard with its counts in <run>/labels/index.jsonl."""

import gzip
import json
import pickle
from pathlib import Path

from experiments.lookahead_students.teacher_label import TeacherLabel

LABEL_FOLDER = "labels"
INDEX_FILE = "index.jsonl"


def write_shard(path: Path, labels: list[TeacherLabel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    with gzip.open(temporary, "wb", compresslevel=3) as file:
        pickle.dump(labels, file, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.rename(path)


def read_shard(path: Path) -> list[TeacherLabel]:
    with gzip.open(path, "rb") as file:
        return pickle.load(file)


def append_index(run: Path, entry: dict) -> None:
    with open(run / LABEL_FOLDER / INDEX_FILE, "a") as file:
        file.write(json.dumps(entry) + "\n")


def read_index(run: Path) -> list[dict]:
    path = run / LABEL_FOLDER / INDEX_FILE
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
