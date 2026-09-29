"""LaTeX tables for the paper, regenerated from ``runs/`` (never typed by hand).

``d4r paper-tables --out <dir>`` writes one ``.tex`` file per table (a ``tabular`` only; captions and
labels live in the paper). Engine names are mapped to the paper's display names here, once.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from d4r.analysis.metrics import load_records, summarize
from d4r.analysis.report import wilcoxon_signed_rank

__all__ = ["DISPLAY", "write_paper_tables"]

RUNS = Path("runs")

DISPLAY: dict[str, str] = {
    "jev": "Typed model (Jev)",
    "jev-raw": "Typed model, raw numbers",
    "jev-ril": "Typed model, rule-guided",
    "jev-cost": "Typed model + cost threshold",
    "jev-gate-lp": "Typed model $\\rightarrow$ LLM+LP gate",
    "deepseek-fast": "LLM, fast",
    "deepseek-fast-raw": "LLM, fast, raw numbers",
    "deepseek-fast-ril": "LLM, fast, rule-guided",
    "deepseek-think": "LLM, reasoning",
    "deepseek-think-ril": "LLM, reasoning, rule-guided",
    "deepseek-lp": "LLM $\\rightarrow$ MILP $\\rightarrow$ HiGHS",
    "fixed-milp": "Code-built MILP",
    "rule-slack": "Slack priority rule",
    "rule-ril420": "Ril 420 priority rules",
    "dla-default": "Interlocking only (default)",
    "random": "Random",
    "oracle": "Rollout oracle",
    "learned-lr": "Learned (logistic regression)",
}


def _f(v: Any, fmt: str) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    return fmt.format(v)


def _load_bench(tag: str) -> dict[str, list[dict[str, Any]]]:
    d = RUNS / "bench" / tag
    return {f.stem: load_records(f) for f in sorted(d.glob("*.jsonl")) if "." not in f.stem}


def _order(summ: dict[str, dict[str, Any]], key: str) -> list[str]:
    return sorted(summ, key=lambda k: summ[k].get(key, 1e9))


def _graded(records: Sequence[dict[str, Any]]) -> bool:
    """Does the engine return graded (not one-hot) option probabilities?"""
    return any(r.get("probabilities") and max(r["probabilities"].values()) < 0.999 for r in records)


def bench_table(
    tag: str, engines: Sequence[str] | None = None, robust_tag: str | None = None
) -> str:
    """Decision-level table: regret overall and per family, robustness set, agreement,
    calibration (graded engines only), latency, cost."""
    recs = _load_bench(tag)
    if engines:
        recs = {k: v for k, v in recs.items() if k in engines}
    robust = _load_bench(robust_tag) if robust_tag else {}
    summ = {k: summarize(v) for k, v in recs.items()}
    scen = sorted({r["scenario"] for v in recs.values() for r in v})
    rows = []
    for k in _order(summ, "mean_regret"):
        s = summ[k]
        per = []
        for sc in scen:
            rr = [r["regret"] for r in recs[k] if r["scenario"] == sc and r["spread"] > 0]
            per.append(_f(statistics.fmean(rr) if rr else None, "{:.1f}"))
        rob = summarize(robust[k])["mean_regret"] if k in robust else None
        graded = _graded(recs[k])
        cells = [DISPLAY.get(k, k), _f(s["mean_regret"], "{:.1f}"), *per]
        if robust_tag:
            cells.append(_f(rob, "{:.1f}"))
        cells += [
            _f(s["hit_rate"], "{:.2f}"),
            _f(s["share_nondefault"], "{:.2f}"),
            _f(s.get("ece") if graded else None, "{:.2f}"),
            _f(s.get("auroc_ptop_hit") if graded else None, "{:.2f}"),
            _f(s["latency_ms_p50"] / 1000 if s["latency_ms_p50"] else 0.0, "{:.2f}"),
            _f(s["cost_usd_per_decision"] * 1000, "{:.3f}"),
        ]
        rows.append(" & ".join(cells) + r" \\")
    head = (
        "Engine & Regret & "
        + " & ".join(f"{sc}" for sc in scen)
        + (" & Seeds 51--200" if robust_tag else "")
        + r" & Agree & Hold & ECE & AUROC & p50 [s] & \$/1k \\"
    )
    cols = "l" + "r" * (8 + len(scen) + (1 if robust_tag else 0))
    return "\n".join(
        [
            rf"\begin{{tabular}}{{{cols}}}",
            r"\toprule",
            head,
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


def rules_table(tag: str) -> str:
    """Scenario M: O0 regret, weighted regret, Ril adherence on decisive cards."""
    recs = _load_bench(tag)
    summ = {k: summarize(v) for k, v in recs.items()}
    rows = []
    for k in _order(summ, "mean_regret"):
        s = summ[k]
        rows.append(
            " & ".join(
                [
                    DISPLAY.get(k, k),
                    _f(s["mean_regret"], "{:.1f}"),
                    _f(s.get("mean_regret_weighted"), "{:.1f}"),
                    _f(s.get("ril_adherence"), "{:.2f}"),
                    _f(s["share_nondefault"], "{:.2f}"),
                ]
            )
            + r" \\"
        )
    return "\n".join(
        [
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Engine & Regret (O0) & Weighted regret & Rule adherence & Hold \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


def contest_table(tag: str, scenario: str) -> str:
    """Closed loop: arrival share, reward difference to the default, W/T/L, Wilcoxon p."""
    d = RUNS / "contest" / tag
    by: dict[str, dict[int, dict[str, Any]]] = {}
    for f in sorted(d.glob(f"{scenario}-*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)["result"]
                by.setdefault(r["engine"], {})[r["seed"]] = r
    base = by.get("dla-default", {})
    rows = []
    weighted = all(
        r.get("weighted_reward") is not None for eps in by.values() for r in eps.values()
    )
    order = sorted(
        by, key=lambda k: -statistics.fmean(r["normalized_reward"] for r in by[k].values())
    )
    for k in order:
        eps = by[k]
        seeds = sorted(eps)
        arr = statistics.fmean(eps[s]["arrival_share"] for s in seeds)
        cells = [DISPLAY.get(k, k), str(len(seeds)), f"{arr:.3f}"]
        if k != "dla-default" and base:
            common = [s for s in seeds if s in base]
            diffs = [eps[s]["total_reward"] - base[s]["total_reward"] for s in common]
            w = wilcoxon_signed_rank(diffs)
            wtl = f"{sum(x > 0 for x in diffs)}/{sum(x == 0 for x in diffs)}/{sum(x < 0 for x in diffs)}"
            cells += [
                f"{statistics.fmean(diffs):+.1f}",
                _f(
                    statistics.median([x for x in diffs if x != 0]) if any(diffs) else None,
                    "{:+.1f}",
                ),
                wtl,
                _f(w["p"], "{:.3f}"),
            ]
        else:
            cells += ["--", "--", "--", "--"]
        if weighted:
            cells.append(f"{statistics.fmean(eps[s]['weighted_reward'] for s in seeds):.1f}")
        cells.append(
            f"{statistics.fmean(eps[s]['engine_latency_ms_total'] / 1000 for s in seeds):.1f}"
        )
        rows.append(" & ".join(cells) + r" \\")
    head = (
        r"Engine & Episodes & Arrived & mean $\Delta$ & median $\Delta{\neq}0$ & W/T/L & $p$"
        + (" & Weighted reward" if weighted else "")
        + r" & Engine [s/ep] \\"
    )
    cols = "lrrrrrr" + ("r" if weighted else "") + "r"
    return "\n".join(
        [
            rf"\begin{{tabular}}{{{cols}}}",
            r"\toprule",
            head,
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


def latency_table(
    taus: Sequence[int],
    scenario: str = "B",
    paused_tag: str = "pilot2",
    seeds: Sequence[int] = range(51, 71),
) -> str:
    """Latency-charged clock: arrival share per engine and seconds per step τ (paused = ∞)."""

    def mean_arrival(tag: str, engine: str) -> float | None:
        f = RUNS / "contest" / tag / f"{scenario}-{engine}.jsonl"
        if not f.exists():
            return None
        rs = [
            json.loads(line)["result"]
            for line in f.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rs = [r for r in rs if r["seed"] in set(seeds)]
        return statistics.fmean(r["arrival_share"] for r in rs) if rs else None

    engines = [
        "dla-default",
        "jev",
        "deepseek-fast",
        "deepseek-lp",
        "jev-gate-lp",
        "deepseek-think",
    ]
    rows = []
    for e in engines:
        cells = [DISPLAY.get(e, e), _f(mean_arrival(paused_tag, e), "{:.3f}")]
        cells += [_f(mean_arrival(f"lat-tau{t}", e), "{:.3f}") for t in taus]
        rows.append(" & ".join(cells) + r" \\")
    head = r"Engine & paused & " + " & ".join(rf"$\tau={t}$\,s" for t in taus) + r" \\"
    return "\n".join(
        [
            r"\begin{tabular}{l" + "r" * (1 + len(taus)) + "}",
            r"\toprule",
            head,
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


def write_paper_tables(out: Path) -> list[str]:
    """Write every table the paper uses; returns the file names written."""
    out.mkdir(parents=True, exist_ok=True)
    written = []
    tables = {
        "tab_bench_abc.tex": lambda: bench_table(
            "pilot2",
            [
                "learned-lr",
                "jev",
                "jev-gate-lp",
                "fixed-milp",
                "deepseek-fast-raw",
                "deepseek-lp",
                "jev-raw",
                "dla-default",
                "deepseek-fast",
                "jev-cost",
                "deepseek-think",
                "rule-slack",
                "random",
            ],
            robust_tag="big",
        ),
        "tab_contest_b.tex": lambda: contest_table("pilot2", "B"),
        "tab_contest_fresh.tex": lambda: contest_table("fresh", "B"),
        "tab_rules_m.tex": lambda: rules_table("m1"),
        "tab_contest_m.tex": lambda: contest_table("m1", "M"),
        "tab_latency_b.tex": lambda: latency_table([10, 3, 1]),
    }
    for name, fn in tables.items():
        try:
            text = fn()
        except (FileNotFoundError, KeyError, statistics.StatisticsError) as exc:
            text = f"% table not available yet: {exc!r}"
        (out / name).write_text(
            "% generated by d4r.analysis.latex — do not edit\n" + text + "\n", encoding="utf-8"
        )
        written.append(name)
    return written
