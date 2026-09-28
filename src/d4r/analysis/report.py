"""Regenerate result tables for a run tag from ``runs/`` (never edit results by hand).

Writes ``results/<tag>/`` with:

- ``bench_summary.md`` / ``.json`` — decision-level metrics per engine (consequential cards);
- ``bench_by_kind.md`` — the same split by decision kind;
- ``calibration.json`` — reliability bins per engine;
- ``gating.md`` — confidence-gated escalation curves (Jev → each slow engine);
- ``contest_summary.md`` / ``.json`` — closed-loop episode metrics and paired tests vs DLA.
"""

from __future__ import annotations

import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any

from d4r.analysis.metrics import calibration_bins, gate_curve, load_records, summarize

__all__ = ["wilcoxon_signed_rank", "write_report"]

RUNS = Path("runs")
RESULTS = Path("results")

_BENCH_COLS = [
    ("n", "n", "{:d}"),
    ("hit_rate", "agree", "{:.3f}"),
    ("mean_regret", "regret", "{:.2f}"),
    ("total_regret", "Σregret", "{:.0f}"),
    ("share_nondefault", "non-default", "{:.2f}"),
    ("brier", "Brier", "{:.3f}"),
    ("ece", "ECE", "{:.3f}"),
    ("auroc_ptop_hit", "AUROC(p_top)", "{:.2f}"),
    ("latency_ms_p50", "p50 ms", "{:.0f}"),
    ("latency_ms_p95", "p95 ms", "{:.0f}"),
    ("cost_usd_per_decision", "$/decision", "{:.6f}"),
    ("errors", "errors", "{:d}"),
]


def _fmt(v: Any, f: str) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    try:
        return f.format(v)
    except (ValueError, TypeError):
        return str(v)


def _table(rows: list[tuple[str, dict[str, Any]]], cols: list[tuple[str, str, str]]) -> str:
    head = "| engine | " + " | ".join(c[1] for c in cols) + " |"
    sep = "|---|" + "|".join("---:" for _ in cols) + "|"
    body = [
        f"| {name} | " + " | ".join(_fmt(s.get(k), f) for k, _, f in cols) + " |"
        for name, s in rows
    ]
    return "\n".join([head, sep, *body])


def wilcoxon_signed_rank(diffs: list[float]) -> dict[str, float]:
    """Two-sided Wilcoxon signed-rank test (exact for n ≤ 16, normal approximation otherwise)."""
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n == 0:
        return {"n": 0, "W+": 0.0, "p": 1.0}
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    w_plus = sum(r for r, x in zip(ranks, d, strict=True) if x > 0)
    total = n * (n + 1) / 2
    if n <= 16:
        count = 0
        extreme = min(w_plus, total - w_plus)
        for signs in itertools.product((0, 1), repeat=n):
            w = sum(r for r, s in zip(ranks, signs, strict=True) if s)
            if w <= extreme + 1e-9:
                count += 1
        p = min(1.0, 2 * count / 2**n)
    else:
        mu = total / 2
        sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z = (abs(w_plus - mu) - 0.5) / sigma
        p = math.erfc(z / math.sqrt(2))
    return {"n": n, "W+": w_plus, "p": p}


def _bootstrap_ci(diffs: list[float], n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap 95 % CI of the mean (deterministic)."""
    import random

    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def _bench(tag: str, out: Path) -> dict[str, Any]:
    d = RUNS / "bench" / tag
    if not d.exists():
        return {}
    files = sorted(d.glob("*.jsonl"))
    recs = {f.stem: load_records(f) for f in files}
    summ = {k: summarize(v) for k, v in recs.items()}
    order = sorted(summ, key=lambda k: summ[k].get("mean_regret", 1e9))
    lines = [
        f"# Bench summary — {tag}",
        "",
        "Consequential cards only. Regret in Flatland reward units (≈ delay steps).",
        "",
    ]
    lines.append(_table([(k, summ[k]) for k in order], _BENCH_COLS))
    (out / "bench_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "bench_summary.json").write_text(
        json.dumps(summ, indent=1, default=float), encoding="utf-8"
    )

    kinds = sorted({r["kind"] for v in recs.values() for r in v})
    kl = [f"# Bench by decision kind — {tag}", ""]
    for kind in kinds:
        rows = [(k, summarize([r for r in recs[k] if r["kind"] == kind])) for k in order]
        kl += [f"## {kind}", "", _table(rows, _BENCH_COLS[:5]), ""]
    (out / "bench_by_kind.md").write_text("\n".join(kl), encoding="utf-8")

    cal = {k: calibration_bins([r for r in v if r["spread"] > 0]) for k, v in recs.items()}
    (out / "calibration.json").write_text(json.dumps(cal, indent=1), encoding="utf-8")

    # paired bootstrap CIs of mean regret differences against reference engines
    pl = [
        f"# Paired regret differences — {tag}",
        "",
        "Mean regret(engine) − regret(reference) on common consequential cards; 95 % percentile bootstrap CI (10 000 resamples, seed 0). Negative = engine better.",
        "",
    ]
    for ref in [k for k in ("jev", "dla-default", "deepseek-lp") if k in recs]:
        rref = {r["card_id"]: r for r in recs[ref] if r["spread"] > 0}
        pl += [
            f"## vs {ref}",
            "",
            "| engine | n | Δ mean regret | 95 % CI | by scenario |",
            "|---|---:|---:|---|---|",
        ]
        for k in order:
            if k == ref:
                continue
            rk = {r["card_id"]: r for r in recs[k] if r["spread"] > 0}
            common = sorted(set(rk) & set(rref))
            if not common:
                continue
            diffs = [rk[c]["regret"] - rref[c]["regret"] for c in common]
            lo, hi = _bootstrap_ci(diffs)
            by_sc: dict[str, list[float]] = {}
            for c in common:
                by_sc.setdefault(rk[c]["scenario"], []).append(rk[c]["regret"] - rref[c]["regret"])
            sc = ", ".join(f"{s_}: {statistics.fmean(v):+.2f}" for s_, v in sorted(by_sc.items()))
            pl.append(
                f"| {k} | {len(common)} | {statistics.fmean(diffs):+.2f} | [{lo:+.2f}, {hi:+.2f}] | {sc} |"
            )
        pl.append("")
    (out / "bench_pairs.md").write_text("\n".join(pl), encoding="utf-8")

    gl = [f"# Confidence-gated escalation — {tag}", ""]
    fast_ids = [k for k in recs if k.startswith("jev")]
    slow_ids = [k for k in recs if not k.startswith("jev") and k not in ("random", "dla-default")]
    for fk in fast_ids:
        fast = [r for r in recs[fk] if r["spread"] > 0]
        for sk in slow_ids:
            slow = [r for r in recs[sk] if r["spread"] > 0]
            curve = gate_curve(fast, slow, [0.0, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.01])
            gl += [
                f"## {fk} → {sk}",
                "",
                "| θ | escalated | regret | random-esc. regret | oracle-esc. regret | agree | ms/decision | $/decision |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for c in curve:
                gl.append(
                    f"| {c['theta']:.2f} | {c['escalation_rate']:.2f} | {c['mean_regret']:.2f} | {c['random_escalation_mean_regret']:.2f}"
                    f" | {c['oracle_escalation_mean_regret']:.2f} | {c['hit_rate']:.3f} | {c['latency_ms_mean']:.0f} | {c['cost_usd_per_decision']:.6f} |"
                )
            gl.append("")
    (out / "gating.md").write_text("\n".join(gl), encoding="utf-8")
    return summ


def _contest(tag: str, out: Path) -> dict[str, Any]:
    d = RUNS / "contest" / tag
    if not d.exists():
        return {}
    by: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for f in sorted(d.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)["result"]
                by.setdefault(r["engine"], {})[(r["scenario"], r["seed"])] = r
    base = by.get("dla-default", {})
    summ: dict[str, Any] = {}
    for eng, eps in by.items():
        keys = sorted(eps)
        s: dict[str, Any] = {
            "episodes": len(keys),
            "arrival_share": statistics.fmean(eps[k]["arrival_share"] for k in keys),
            "normalized_reward": statistics.fmean(eps[k]["normalized_reward"] for k in keys),
            "total_arrival_delay": statistics.fmean(eps[k]["total_arrival_delay"] for k in keys),
            "deadlocked_trains": statistics.fmean(eps[k]["deadlocked_trains"] for k in keys),
            "decisions": statistics.fmean(sum(eps[k]["decisions"].values()) for k in keys),
            "fallbacks": sum(eps[k]["fallbacks"] for k in keys),
            "latency_s_per_episode": statistics.fmean(
                eps[k]["engine_latency_ms_total"] / 1000 for k in keys
            ),
            "cost_usd_per_episode": statistics.fmean(eps[k]["cost_usd"] for k in keys),
        }
        common = [k for k in keys if k in base]
        if common and eng != "dla-default":
            diffs = [eps[k]["total_reward"] - base[k]["total_reward"] for k in common]
            s["reward_gain_vs_dla"] = statistics.fmean(diffs)
            s["wins_ties_losses"] = (
                sum(x > 0 for x in diffs),
                sum(x == 0 for x in diffs),
                sum(x < 0 for x in diffs),
            )
            s["wilcoxon_vs_dla"] = wilcoxon_signed_rank(diffs)
        summ[eng] = s
    order = sorted(summ, key=lambda k: -summ[k]["normalized_reward"])
    lines = [
        f"# Closed-loop contest — {tag}",
        "",
        "| engine | episodes | arrived | norm. reward | Δreward vs DLA | W/T/L | p (Wilcoxon) | deadlocked | decisions/ep | fallbacks | engine s/ep | $/ep |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for k in order:
        s = summ[k]
        w = s.get("wilcoxon_vs_dla", {})
        lines.append(
            f"| {k} | {s['episodes']} | {s['arrival_share']:.3f} | {s['normalized_reward']:.4f} | {_fmt(s.get('reward_gain_vs_dla'), '{:+.1f}')}"
            f" | {'/'.join(map(str, s.get('wins_ties_losses', ()))) or '–'} | {_fmt(w.get('p'), '{:.3f}')} | {s['deadlocked_trains']:.2f}"
            f" | {s['decisions']:.1f} | {s['fallbacks']} | {s['latency_s_per_episode']:.1f} | {s['cost_usd_per_episode']:.4f} |"
        )
    (out / "contest_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "contest_summary.json").write_text(
        json.dumps(summ, indent=1, default=float), encoding="utf-8"
    )
    return summ


def write_report(tag: str) -> str:
    """Write all tables for ``tag``; return a short text summary."""
    out = RESULTS / tag
    out.mkdir(parents=True, exist_ok=True)
    b = _bench(tag, out)
    c = _contest(tag, out)
    return json.dumps(
        {"results": str(out), "bench_engines": sorted(b), "contest_engines": sorted(c)}
    )
