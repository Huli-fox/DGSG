"""Minimal .env file loader and saver (no external dependencies)."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def read_env_file(env_path: Path = DEFAULT_ENV_PATH) -> dict[str, str]:
    """Parse KEY=VALUE pairs from a .env file."""
    result: dict[str, str] = {}
    if not env_path.is_file():
        return result
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key:
            result[key] = _unquote(value)
    return result


def load_env(env_path: Path = DEFAULT_ENV_PATH, override: bool = False) -> dict[str, str]:
    """Load variables from .env into os.environ."""
    values = read_env_file(env_path)
    for key, value in values.items():
        if override or not os.environ.get(key):
            os.environ[key] = value
    return values


def get_api_key(env_path: Path = DEFAULT_ENV_PATH) -> str:
    """Return DEEPGRAM_API_KEY from environment or .env file."""
    env_val = os.environ.get("DEEPGRAM_API_KEY", "").strip()
    if env_val:
        return env_val
    return read_env_file(env_path).get("DEEPGRAM_API_KEY", "").strip()


def save_api_key(api_key: str, env_path: Path = DEFAULT_ENV_PATH) -> None:
    """Save or update DEEPGRAM_API_KEY in .env while preserving other lines."""
    clean_key = api_key.strip()
    lines: list[str] = []
    if env_path.is_file():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    updated = False
    new_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            k, _, _ = stripped.partition("=")
            if k.strip() == "DEEPGRAM_API_KEY":
                new_lines.append(f"DEEPGRAM_API_KEY={clean_key}")
                updated = True
                continue
        new_lines.append(line)

    if not updated:
        if new_lines and new_lines[-1].strip():
            new_lines.append("")
        new_lines.append("# Deepgram")
        new_lines.append(f"DEEPGRAM_API_KEY={clean_key}")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    os.environ["DEEPGRAM_API_KEY"] = clean_key
