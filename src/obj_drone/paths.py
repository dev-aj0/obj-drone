"""Filesystem locations.

Kept dependency-free and separate from :mod:`obj_drone.config` so that low-level
modules (e.g. vision.debug) can resolve paths without importing the config
module, which imports the controller package, which imports them back.
"""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Repository root — the directory containing config/, models/ and src/."""
    return Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path | None) -> Path | None:
    """Resolve a config path relative to the project root unless absolute."""
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else project_root() / p
