"""Test–retest consistency (P2-H5) and joint coordination (P2-H7) from run records.

- **Test–retest.** The same card answered by the same engine in two independent runs (different
  tags, no shared cache) is compared: share of flipped choices and mean absolute change of the
  probability of the non-default option.
- **Joint coordination.** In closed-loop records, two ``meet`` decisions taken in the same step for
  trains that are oncoming to each other form a *meeting pair*. Both choosing ``HOLD`` means both
  wait for each other (a joint inconsistency: neither goes); both choosing ``PROCEED`` leaves the
  order to the interlocking (consistent with the default). We report the rate of both-hold pairs.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

__all__ = ["meeting_pairs", "retest"]


def _load(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["card_id"]] = r
    return out


def _p_nd(r: dict[str, Any]) -> float | None:
    p = r.get("probabilities")
    if not p:
        return None
    nd = [o for o in r["values"] if o != r["default_option"]]
    return sum(p.get(o, 0.0) for o in nd)


def retest(a: Path, b: Path) -> dict[str, float]:
    """Compare two bench files of the same engine on their common cards."""
    ra, rb = _load(a), _load(b)
    common = sorted(set(ra) & set(rb))
    if not common:
        return {"n": 0}
    flips = [ra[c]["option"] != rb[c]["option"] for c in common]
    dp = [
        abs(pa - pb)
        for c in common
        if (pa := _p_nd(ra[c])) is not None and (pb := _p_nd(rb[c])) is not None
    ]
    reg_a = statistics.fmean(ra[c]["regret"] for c in common)
    reg_b = statistics.fmean(rb[c]["regret"] for c in common)
    return {
        "n": len(common),
        "flip_rate": sum(flips) / len(common),
        "mean_abs_dp_nondefault": statistics.fmean(dp) if dp else float("nan"),
        "regret_run_a": reg_a,
        "regret_run_b": reg_b,
    }


def meeting_pairs(contest_file: Path) -> dict[str, float]:
    """Joint-consistency statistics of same-step meeting pairs in closed-loop records."""
    pairs = both_hold = both_proceed = mixed = 0
    for line in contest_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        by_step: dict[int, list[dict[str, Any]]] = {}
        for d in rec["decisions"]:
            if d["card"]["kind"] == "meet":
                by_step.setdefault(d["card"]["step"], []).append(d)
        for ds in by_step.values():
            for i in range(len(ds)):
                for j in range(i + 1, len(ds)):
                    ti, tj = ds[i]["card"]["train"]["id"], ds[j]["card"]["train"]["id"]
                    oi = {o["id"] for o in ds[i]["card"]["others"]}
                    oj = {o["id"] for o in ds[j]["card"]["others"]}
                    if tj in oi and ti in oj:
                        pairs += 1
                        a, b = ds[i]["applied_option"], ds[j]["applied_option"]
                        if a == b == "HOLD":
                            both_hold += 1
                        elif a == b == "PROCEED":
                            both_proceed += 1
                        else:
                            mixed += 1
    return {
        "pairs": pairs,
        "both_hold": both_hold,
        "both_proceed": both_proceed,
        "mixed": mixed,
        "both_hold_rate": both_hold / pairs if pairs else float("nan"),
    }
