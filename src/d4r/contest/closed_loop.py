"""Closed-loop contest: an engine dispatches whole episodes.

One JSONL line per (scenario, seed): the episode result and all decision records. Resumable: seeds
already present in the output file are skipped.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from d4r.dispatch.controller import run_episode
from d4r.engines.base import Engine
from d4r.sim.scenario import Scenario

__all__ = ["run_contest"]


def run_contest(
    scenario: Scenario, seeds: Iterable[int], engine: Engine, out: Path, **controller_kwargs: Any
) -> list[dict[str, Any]]:
    """Run ``engine`` on every seed and append one record per episode to ``out``."""
    out.parent.mkdir(parents=True, exist_ok=True)
    done: dict[tuple[str, int], dict[str, Any]] = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[(rec["result"]["scenario"], rec["result"]["seed"])] = rec
    rows = []
    with out.open("a", encoding="utf-8") as f:
        for seed in seeds:
            key = (scenario.name, seed)
            if key in done:
                rows.append(done[key])
                continue
            result, records = run_episode(scenario, seed, engine, **controller_kwargs)
            rec = {
                "result": result.to_dict(),
                "controller": controller_kwargs,
                "decisions": [r.to_dict() for r in records],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            rows.append(rec)
    return rows
