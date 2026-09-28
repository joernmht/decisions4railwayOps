"""Load API keys into the process environment without ever echoing them.

The repository is public, so keys never live in it. They come from, in order:
real environment variables, a secrets file named by ``D4R_SECRETS_FILE``, and a
git-ignored ``.env`` in the working directory. Nothing in this module prints or
returns a key value; callers only learn *whether* a key is present.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["KEY_NAMES", "has_key", "load_secrets"]

#: Environment variables the engines read.
KEY_NAMES = ("TYPESAFE_API_KEY", "DEEPSEEK_API_KEY")


def load_secrets() -> None:
    """Populate ``os.environ`` from the configured secrets sources (no override)."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv is a core dependency
        return
    extra = os.environ.get("D4R_SECRETS_FILE")
    if extra and Path(extra).is_file():
        load_dotenv(extra, override=False)
    load_dotenv(Path.cwd() / ".env", override=False)


def has_key(name: str) -> bool:
    """Return whether ``name`` is set to a non-empty value (the value itself is never exposed)."""
    return bool(os.environ.get(name))
