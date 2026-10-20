"""Atomic text exports: preserve the previous artifact if writing fails."""

from contextlib import contextmanager
from pathlib import Path
import uuid


@contextmanager
def atomic_text(path):
    """Write beside the destination, replace on success, and remove partial files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            yield stream
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
