"""Named engine configurations used in the experiments.

One place defines what each engine id means, so run records, tables and the paper use the same
names. Paid engines get a per-engine JSONL cache under ``cache_dir``; ``salt`` separates repeated
calls for the test-retest experiment (P2-H5) from the main cache.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from d4r.engines.base import DefaultEngine, Engine, RandomEngine
from d4r.engines.cache import ResponseCache

__all__ = ["ENGINE_IDS", "PAID", "make_engine"]

#: Engines that call a paid API.
PAID = frozenset(
    {
        "jev",
        "jev-raw",
        "deepseek-fast",
        "deepseek-fast-raw",
        "deepseek-think",
        "deepseek-lp",
        "deepseek-lp-think",
    }
)


def _cache(cache_dir: Path | None, name: str, salt: str) -> ResponseCache:
    if cache_dir is None:
        return ResponseCache(None)
    suffix = f".{salt}" if salt else ""
    return ResponseCache(cache_dir / f"{name}{suffix}.jsonl")


def make_engine(engine_id: str, cache_dir: Path | None = None, salt: str = "") -> Engine:
    """Instantiate an engine by id (see ``ENGINE_IDS``)."""
    from d4r.engines.jev import JevEngine
    from d4r.engines.llm import DeepSeekEngine
    from d4r.engines.milp import DeepSeekLPEngine, FixedMilpEngine
    from d4r.engines.oracle import RolloutOracle
    from d4r.engines.rules import SlackPriorityEngine

    c = _cache(cache_dir, engine_id, salt)
    factories: dict[str, Callable[[], Engine]] = {
        "dla-default": DefaultEngine,
        "random": RandomEngine,
        "rule-slack": SlackPriorityEngine,
        "fixed-milp": FixedMilpEngine,
        "oracle": RolloutOracle,
        "jev": lambda: JevEngine(cache=c),
        "jev-raw": lambda: JevEngine(variant="raw", cache=c),
        "deepseek-fast": lambda: DeepSeekEngine(thinking=False, cache=c),
        "deepseek-fast-raw": lambda: DeepSeekEngine(thinking=False, variant="raw", cache=c),
        "deepseek-think": lambda: DeepSeekEngine(thinking=True, cache=c),
        "deepseek-lp": lambda: DeepSeekLPEngine(thinking=False, cache=c),
        "deepseek-lp-think": lambda: DeepSeekLPEngine(thinking=True, cache=c),
    }
    if engine_id not in factories:
        raise KeyError(f"unknown engine {engine_id!r}; known: {sorted(factories)}")
    eng = factories[engine_id]()
    eng.name = engine_id  # type: ignore[misc]
    return eng


ENGINE_IDS = (
    "dla-default",
    "random",
    "rule-slack",
    "fixed-milp",
    "jev",
    "jev-raw",
    "deepseek-fast",
    "deepseek-fast-raw",
    "deepseek-think",
    "deepseek-lp",
    "deepseek-lp-think",
    "oracle",
)
