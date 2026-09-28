"""Offline benchmark: every engine answers the labelled cards of a game set.

Records are written one per line and the run is resumable: cards already answered in the output
file are skipped. API engines are called from a small thread pool; answers are cached, so a rerun
reproduces the same decisions.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from d4r.contest.gameset import LabelledCard
from d4r.engines.base import Decision, Engine

__all__ = ["bench_record", "run_bench"]


def bench_record(lc: LabelledCard, d: Decision, engine: str) -> dict[str, Any]:
    """Flat JSON record for one answered card."""
    values: dict[str, float] = lc["values"]
    best_v = max(values.values())
    chosen_v = values.get(d.option_id, min(values.values()))
    card = lc["card"]
    return {
        "engine": engine,
        "card_id": card["card_id"],
        "scenario": card["scenario"],
        "seed": card["seed"],
        "step": card["step"],
        "kind": card["kind"],
        "default_option": card["default_option"],
        "values": values,
        "best": lc["best"],
        "spread": lc["spread"],
        "option": d.option_id,
        "regret": round(best_v - chosen_v, 3),
        "hit": d.option_id in lc["best"],
        "probabilities": d.probabilities,
        "latency_ms": d.latency_ms,
        "input_tokens": d.input_tokens,
        "output_tokens": d.output_tokens,
        "cost_usd": d.cost_usd,
        "model": d.model,
        "error": d.error,
        "meta": {k: v for k, v in d.meta.items() if k != "cache_key"},
    }


def run_bench(
    cards: Sequence[LabelledCard], engine: Engine, out: Path, workers: int = 1
) -> list[dict[str, Any]]:
    """Answer ``cards`` with ``engine``; append records to ``out``; return all records."""
    out.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict[str, Any]] = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["card_id"]] = rec
    todo = [lc for lc in cards if lc["card"]["card_id"] not in done]

    def one(lc: LabelledCard) -> dict[str, Any]:
        d = engine.decide([lc.card])[0]
        return bench_record(lc, d, engine.name)

    with out.open("a", encoding="utf-8") as f:
        if workers <= 1:
            results = map(one, todo)
            for rec in results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                done[rec["card_id"]] = rec
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for rec in pool.map(one, todo):
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    done[rec["card_id"]] = rec
    order = [lc["card"]["card_id"] for lc in cards]
    return [done[c] for c in order if c in done]
