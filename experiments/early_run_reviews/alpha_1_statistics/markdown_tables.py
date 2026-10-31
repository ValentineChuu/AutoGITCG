"""Markdown table text for the report, and the collector that writes all generated tables to one file."""

from pathlib import Path


def markdown_table(header: list[str], rows: list[list]) -> str:
    def cell(value) -> str:
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(cell(value) for value in row) + " |" for row in rows]
    return "\n".join(lines)


class TableCollector:
    """Accumulates titled tables and notes; `write` saves them as one Markdown file."""

    def __init__(self):
        self.sections: list[str] = []

    def add(self, title: str, header: list[str], rows: list[list], note: str = "") -> None:
        text = f"### {title}\n\n"
        if note:
            text += f"{note}\n\n"
        self.sections.append(text + markdown_table(header, rows))
        print(f"\n{self.sections[-1]}", flush=True)

    def note(self, title: str, text: str) -> None:
        self.sections.append(f"### {title}\n\n{text}")
        print(f"\n{self.sections[-1]}", flush=True)

    def write(self, path: Path, heading: str) -> None:
        path.write_text(f"# {heading}\n\n" + "\n\n".join(self.sections) + "\n", encoding="utf-8")
