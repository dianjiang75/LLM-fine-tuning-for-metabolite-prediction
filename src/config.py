"""Read optional local credentials without replacing exported environment variables."""

import os
from pathlib import Path


def load_local_env(path: Path) -> None:
    """Load simple KEY=value entries if the file exists. Never print their values."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def require_env(name: str) -> str:
    """Return a required credential or explain how to configure it."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Set {name} in your environment or the repository's .env file before "
            "running this service-backed step. Offline replay needs no API key."
        )
    return value
