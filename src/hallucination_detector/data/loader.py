import json
from pathlib import Path
from typing import Any


class JSONLLoadError(Exception):
    """Raised when a JSONL file cannot be loaded correctly."""


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """
    Load a JSON Lines file.

    Each non-empty line must contain one valid JSON object.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise JSONLLoadError(f"Expected a file: {path}")

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JSONLLoadError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise JSONLLoadError(
                    f"Expected JSON object at {path}:{line_number}, "
                    f"got {type(record).__name__}"
                )

            records.append(record)

    return records