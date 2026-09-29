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
        "jev-cost",
        "jev-gate-lp",
        "jev-gate-think",
        "jev-ril",
        "deepseek-fast-ril",
        "deepseek-think-ril",
        "jev",
        "jev-raw",
        "deepseek-fast",
        "deepseek-fast-raw",
        "deepseek-think",
        "deepseek-lp",
        "deepseek-lp-think",
    }
)


#: Bayes threshold c_FP/(c_FP+c_FN) from scenario-A game-set labels (19.6 / (19.6 + 54.5)).
JEV_COST_THETA = 0.265
#: Escalation threshold on Jev's p_top, fitted on scenario A (escalation capped at 50 %).
JEV_GATE_THETA = 0.61


def _cache(cache_dir: Path | None, name: str, salt: str) -> ResponseCache:
    if cache_dir is None:
        return ResponseCache(None)
    suffix = f".{salt}" if salt else ""
    return ResponseCache(cache_dir / f"{name}{suffix}.jsonl")


def make_engine(engine_id: str, cache_dir: Path | None = None, salt: str = "") -> Engine:
    """Instantiate an engine by id (see ``ENGINE_IDS``)."""
    from d4r.engines.composite import CostSensitiveEngine, GatedEngine
    from d4r.engines.jev import JevEngine
    from d4r.engines.llm import DeepSeekEngine
    from d4r.engines.milp import DeepSeekLPEngine, FixedMilpEngine
    from d4r.engines.oracle import RolloutOracle
    from d4r.engines.rules import RilPriorityEngine, SlackPriorityEngine
    from d4r.rules import rules_text

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
        # Ril 420 rule study (mixed traffic, P2-H9)
        "rule-ril420": RilPriorityEngine,
        "jev-ril": lambda: JevEngine(cache=c, guidance=rules_text()),
        "deepseek-fast-ril": lambda: DeepSeekEngine(thinking=False, cache=c, guidance=rules_text()),
        "deepseek-think-ril": lambda: DeepSeekEngine(thinking=True, cache=c, guidance=rules_text()),
        # pilot2 composites; thresholds fixed on scenario A (lab notebook, 2026-09-28)
        "jev-cost": lambda: CostSensitiveEngine(
            JevEngine(cache=_cache(cache_dir, "jev", salt)), theta=JEV_COST_THETA
        ),
        "jev-gate-lp": lambda: GatedEngine(
            JevEngine(cache=_cache(cache_dir, "jev", salt)),
            DeepSeekLPEngine(thinking=False, cache=_cache(cache_dir, "deepseek-lp", salt)),
            theta=JEV_GATE_THETA,
        ),
        "jev-gate-think": lambda: GatedEngine(
            JevEngine(cache=_cache(cache_dir, "jev", salt)),
            DeepSeekEngine(thinking=True, cache=_cache(cache_dir, "deepseek-think", salt)),
            theta=JEV_GATE_THETA,
        ),
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
    "jev-cost",
    "jev-gate-lp",
    "jev-gate-think",
    "rule-ril420",
    "jev-ril",
    "deepseek-fast-ril",
    "deepseek-think-ril",
    "oracle",
)
