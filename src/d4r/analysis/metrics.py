"""Decision-level metrics over bench records (see :mod:`d4r.contest.bench`).

All metrics are computed on *consequential* cards by default (cards whose options differ in rollout
value); on the others every choice has zero regret. Probabilistic scores use the engine's option
distribution against the oracle's best set (ties share the target mass).
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "auroc",
    "calibration_bins",
    "gate_curve",
    "load_records",
    "summarize",
]

Record = dict[str, Any]


def load_records(path: Path) -> list[Record]:
    """Read a bench JSONL file."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _target(rec: Record) -> dict[str, float]:
    best = rec["best"]
    return {o: (1.0 / len(best) if o in best else 0.0) for o in rec["values"]}


def _probs(rec: Record) -> dict[str, float] | None:
    p = rec.get("probabilities")
    if not p:
        return None
    z = sum(p.get(o, 0.0) for o in rec["values"])
    if z <= 0:
        return None
    return {o: p.get(o, 0.0) / z for o in rec["values"]}


def _pct(xs: Sequence[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Area under the ROC curve (probability that a random positive outranks a random negative)."""
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def calibration_bins(records: Iterable[Record], n_bins: int = 10) -> list[dict[str, float]]:
    """Reliability table of p_top (probability of the chosen option) vs hit rate."""
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for r in records:
        p = _probs(r)
        if p is None:
            continue
        pt = p.get(r["option"], 0.0)
        i = min(n_bins - 1, int(pt * n_bins))
        bins[i].append((pt, bool(r["hit"])))
    out = []
    for i, b in enumerate(bins):
        if b:
            out.append(
                {
                    "bin": i,
                    "lo": i / n_bins,
                    "hi": (i + 1) / n_bins,
                    "n": len(b),
                    "mean_p": sum(p for p, _ in b) / len(b),
                    "hit_rate": sum(h for _, h in b) / len(b),
                }
            )
    return out


def summarize(records: Sequence[Record], consequential_only: bool = True) -> dict[str, Any]:
    """Headline metrics for one engine."""
    rs = [r for r in records if (r["spread"] > 0 or not consequential_only)]
    n = len(rs)
    if n == 0:
        return {"n": 0}
    regrets = [r["regret"] for r in rs]
    hits = [bool(r["hit"]) for r in rs]
    nondefault = [r["option"] != r["default_option"] for r in rs]
    lat = [r["latency_ms"] for r in rs if r.get("latency_ms") is not None]
    cost = [r.get("cost_usd", 0.0) for r in rs]
    out: dict[str, Any] = {
        "n": n,
        "hit_rate": sum(hits) / n,
        "mean_regret": statistics.fmean(regrets),
        "total_regret": sum(regrets),
        "share_nondefault": sum(nondefault) / n,
        "errors": sum(1 for r in rs if r.get("error")),
        "latency_ms_p50": _pct(lat, 0.5),
        "latency_ms_p95": _pct(lat, 0.95),
        "cost_usd_per_decision": statistics.fmean(cost) if cost else 0.0,
        "input_tokens_mean": statistics.fmean([r.get("input_tokens", 0) for r in rs]),
        "output_tokens_mean": statistics.fmean([r.get("output_tokens", 0) for r in rs]),
    }
    dec = [r for r in rs if r.get("ril_decisive")]
    if dec:
        out["ril_decisive_n"] = len(dec)
        out["ril_adherence"] = sum(r["option"] == r["ril_option"] for r in dec) / len(dec)
    rw = [r["regret_weighted"] for r in records if r.get("spread_weighted", 0) > 0]
    if rw:
        out["mean_regret_weighted"] = statistics.fmean(rw)
    probs = [(r, _probs(r)) for r in rs]
    probs = [(r, p) for r, p in probs if p is not None]
    if probs:
        brier, logloss, ptop = [], [], []
        for r, p in probs:
            t = _target(r)
            brier.append(sum((p[o] - t[o]) ** 2 for o in p))
            pb = sum(p[o] for o in r["best"])
            logloss.append(-math.log(max(pb, 1e-6)))
            ptop.append(p.get(r["option"], 0.0))
        bins = calibration_bins([r for r, _ in probs])
        ece = sum(b["n"] * abs(b["mean_p"] - b["hit_rate"]) for b in bins) / len(probs)
        out |= {
            "n_prob": len(probs),
            "brier": statistics.fmean(brier),
            "log_loss": statistics.fmean(logloss),
            "ece": ece,
            "p_top_mean": statistics.fmean(ptop),
            "auroc_ptop_hit": auroc(ptop, [bool(r["hit"]) for r, _ in probs]),
        }
    return out


def gate_curve(
    fast: Sequence[Record], slow: Sequence[Record], thresholds: Sequence[float]
) -> list[dict[str, float]]:
    """Confidence-gated escalation (P2-H4): use ``fast`` if its p_top >= θ, else ``slow``.

    For each θ the escalation rate r is reported with three variants: gated, *random* escalation at
    the same rate (exact expectation) and *oracle* escalation of the same number of cards (escalate
    where the slow engine is better first). Latency and cost add the fast call to every escalated
    decision.
    """
    s_by = {r["card_id"]: r for r in slow}
    pairs = [(f, s_by[f["card_id"]]) for f in fast if f["card_id"] in s_by]
    n = len(pairs)
    out = []
    for th in thresholds:
        esc = []
        for f, _s in pairs:
            p = _probs(f)
            pt = p.get(f["option"], 0.0) if p else 1.0
            esc.append(pt < th)
        k = sum(esc)
        rate = k / n if n else 0.0
        reg = [s["regret"] if e else f["regret"] for (f, s), e in zip(pairs, esc, strict=True)]
        hit = [s["hit"] if e else f["hit"] for (f, s), e in zip(pairs, esc, strict=True)]
        rnd = [(1 - rate) * f["regret"] + rate * s["regret"] for f, s in pairs]
        gains = sorted((f["regret"] - s["regret"] for f, s in pairs), reverse=True)
        oracle_total = sum(f["regret"] for f, _ in pairs) - sum(g for g in gains[:k] if g > 0)
        lat = [
            f["latency_ms"] + (s["latency_ms"] if e else 0.0)
            for (f, s), e in zip(pairs, esc, strict=True)
        ]
        cost = [
            f.get("cost_usd", 0.0) + (s.get("cost_usd", 0.0) if e else 0.0)
            for (f, s), e in zip(pairs, esc, strict=True)
        ]
        out.append(
            {
                "theta": th,
                "escalation_rate": rate,
                "mean_regret": statistics.fmean(reg) if reg else float("nan"),
                "hit_rate": sum(hit) / n if n else float("nan"),
                "random_escalation_mean_regret": statistics.fmean(rnd) if rnd else float("nan"),
                "oracle_escalation_mean_regret": oracle_total / n if n else float("nan"),
                "latency_ms_mean": statistics.fmean(lat) if lat else float("nan"),
                "cost_usd_per_decision": statistics.fmean(cost) if cost else float("nan"),
            }
        )
    return out
