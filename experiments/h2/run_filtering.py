"""P2-H2b filter half: Jev as a relevance filter ("driver of RAG") in front of an MILP generator.

Hypothesis labels (docs/hypotheses.md v0.1): P2-H2b covers requirement -> constraint-family routing
(experiments/h2/run_routing.py) and this relevance filtering; P2-H2c (semantic pre-solve gate) is
not yet tested.

Why: the claim under test is bounded. Jev does not generate the model; it decides which sentences
of a noisy problem statement reach the generator. That is worth something only if (1) its recall
on data-bearing sentences is essentially perfect at a threshold chosen on held-out statements,
(2) it removes noise that a digit regex cannot (the numeracy trap) and (3) the generator's model
built from the filtered statement is at least as often correct, with fewer prompt tokens.

Steps (all paid calls cached under ``runs/cache/h2-filtering-*.jsonl``, so a rerun replays and a
run cut short by a budget cap resumes where it stopped):

1. generate ``--n`` seeded statements (:mod:`d4r.h2.statements`); the first ``--n-dev`` form the
   dev split (threshold choice), the rest the test split;
2. score every statement with each filter arm (:mod:`d4r.h2.filtering`), sweep thresholds, pick
   the largest threshold with perfect recall of needed sentences on dev, report on test;
3. downstream on the first ``--downstream-n`` test statements: DeepSeek-flash writes the MILP
   (thinking and/or non-thinking, ``--downstream``) from the full statement and from filtered
   ones (Jev at the dev threshold, Jev at ``jev@0.3``, gold labels, DeepSeek's list); HiGHS
   solves; the objective is compared with the brute-force optimum. Identical filtered texts share
   one cached generation. ``jev@0.3`` is a *post-hoc* safety margin: it was not selected on dev
   (dev recall of needed sentences is 1 for every threshold up to 0.5) but fixed after the sweep
   over all 60 statements, test split included.

Writes ``results/h2/filtering_summary.{json,md}`` and ``results/h2/filtering_per_statement.json``
(aggregates and per-statement counts only; no API keys, no private data).

Usage::

    ~/.venvs/d4r/bin/python experiments/h2/run_filtering.py
    ~/.venvs/d4r/bin/python experiments/h2/run_filtering.py --offline       # replay check
    ~/.venvs/d4r/bin/python experiments/h2/run_filtering.py --render-only   # re-render the .md

``--offline`` replays every arm and the downstream step from the caches only (a request missing
from a cache stops the run; nothing reaches the network) and writes to ``results/h2/offline/``,
so it never overwrites the reported results. Its output equals the reported one except for the
run timestamp and the latencies measured live on replay (the regex arm and the HiGHS solve time
inside the downstream ``latency_ms_mean``). ``--render-only`` rewrites ``filtering_summary.md``
(and the static labels of ``filtering_summary.json``) from the reported JSON without any calls.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from d4r.engines.cache import ResponseCache
from d4r.engines.jev import JEV_PRICE_PER_INPUT_TOKEN
from d4r.engines.llm import DeepSeekChat
from d4r.h2.filtering import (
    Budget,
    DeepSeekSentenceFilter,
    EmbeddingSentenceFilter,
    FilterScores,
    GenerationResult,
    JevSentenceFilter,
    ReplayCache,
    aggregate,
    choose_floor_threshold,
    cost_latency,
    generate_and_solve,
    keep_all_scores,
    kept_ids,
    mcnemar_exact_p,
    objective_matches,
    regex_scores,
    sentence_metrics,
    sweep,
)
from d4r.h2.statements import Statement, make_statements, reference_optimum

ROOT = Path(__file__).resolve().parents[2]
JEV_GRID = (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
THRESHOLD_ARMS = ("jev", "jev-bare", "embed")
ALL_ARMS = ("jev", "jev-bare", "deepseek", "embed", "regex", "keep-all")
#: Static labels written into filtering_summary.json (and refreshed by --render-only).
LABELS: dict[str, Any] = {
    "experiment": (
        "P2-H2b, relevance-filtering half (Jev as a relevance filter in front of an MILP generator)"
    ),
    "hypothesis": (
        "docs/hypotheses.md v0.1: P2-H2b = requirement -> constraint-family routing"
        " (results/h2/routing_summary.md) plus relevance filtering (this file); P2-H2c"
        " (semantic pre-solve gate) is not yet tested"
    ),
    "downstream_conditions": {
        "full": "the full statement, no filter",
        "jev": (
            "Jev at the dev-selected threshold (largest threshold with recall of needed"
            " sentences = 1 on the dev split)"
        ),
        "jev@0.3": (
            "Jev at theta = 0.3: a post-hoc safety margin, NOT selected on dev (dev recall of"
            " needed sentences is 1 for every threshold up to 0.5); fixed after the sweep over"
            " all 60 statements, test split included"
        ),
        "gold": "the gold needed sentences (oracle filter)",
        "deepseek": "DeepSeek-flash's list of needed sentences",
    },
}
#: Downstream conditions whose threshold was not chosen on dev.
POST_HOC = frozenset({"jev@0.3"})
#: generator -> conditions (input texts) of the downstream step
DEFAULT_DOWNSTREAM = "fast:full,jev,jev@0.3,gold,deepseek;think:full,jev,jev@0.3,gold,deepseek"


def _setup_keys() -> None:
    os.environ.setdefault("D4R_SECRETS_FILE", os.path.expanduser("~/.config/raiLP/secrets.env"))
    from d4r.apikeys import load_secrets

    load_secrets()


def score_arm(
    arm: str,
    stmts: Sequence[Statement],
    cache_dir: Path,
    budget: Budget,
    chat: DeepSeekChat,
    workers: int,
    cache_cls: type[ResponseCache] = ResponseCache,
) -> dict[str, FilterScores]:
    """Per-statement scores of one arm (parallel for the hosted models, sequential for ScaDS).

    ``cache_cls=ReplayCache`` replays from the cache only (offline).
    """
    scorer: Callable[[Statement], FilterScores]
    if arm == "regex":
        scorer = regex_scores
    elif arm == "keep-all":
        scorer = keep_all_scores
    elif arm in ("jev", "jev-bare"):
        jev = JevSentenceFilter(
            criteria=arm == "jev",
            cache=cache_cls(cache_dir / f"h2-filtering-{arm}.jsonl"),
            budget=budget,
        )
        scorer = jev.score
    elif arm == "deepseek":
        ds = DeepSeekSentenceFilter(
            cache=cache_cls(cache_dir / "h2-filtering-deepseek.jsonl"), chat=chat, budget=budget
        )
        scorer = ds.score
    elif arm == "embed":
        emb = EmbeddingSentenceFilter(cache=cache_cls(cache_dir / "h2-filtering-embed.jsonl"))
        scorer = emb.score
        workers = 1
    else:
        raise ValueError(arm)
    if workers <= 1:
        results = [scorer(s) for s in stmts]
    else:
        with ThreadPoolExecutor(workers) as pool:
            results = list(pool.map(scorer, stmts))
    return {r.stmt_id: r for r in results}


def embed_grid(dev: Sequence[Statement], scores: dict[str, FilterScores]) -> list[float]:
    """Cosine thresholds at every 5th percentile of the dev scores."""
    vals = sorted(v for s in dev for v in scores[s.stmt_id].scores.values())
    qs = [vals[min(len(vals) - 1, int(p / 100 * len(vals)))] for p in range(0, 101, 5)]
    return sorted({round(q, 4) for q in qs})


def arm_report(
    arm: str,
    dev: Sequence[Statement],
    test: Sequence[Statement],
    scores: dict[str, FilterScores],
) -> dict[str, Any]:
    everyone = [*dev, *test]
    rep: dict[str, Any] = {"arm": arm, **cost_latency([scores[s.stmt_id] for s in everyone])}
    models = sorted({scores[s.stmt_id].model for s in everyone if scores[s.stmt_id].model})
    rep["model"] = ", ".join(models)
    if arm in THRESHOLD_ARMS:
        grid = list(JEV_GRID) if arm != "embed" else embed_grid(dev, scores)
        dev_rows = sweep(dev, scores, grid)
        th = choose_floor_threshold(dev_rows)
        rep["threshold"] = th
        rep["threshold_rule"] = "largest threshold with recall of needed sentences = 1 on dev"
        rep["dev_sweep"] = dev_rows
        rep["test_sweep"] = sweep(test, scores, grid)
        rep["all_sweep"] = sweep(everyone, scores, grid)
    else:
        th = 0.5
        rep["threshold"] = None
    for name, part in (("dev", dev), ("test", test), ("all", everyone)):
        ms = [sentence_metrics(s, kept_ids(scores[s.stmt_id], th)) for s in part]
        rep[name] = aggregate(ms)
    if arm in ("jev", "jev-bare"):
        ms = [sentence_metrics(s, kept_ids(scores[s.stmt_id], 0.5)) for s in test]
        rep["test_at_0.5"] = aggregate(ms)
    return rep


def parse_downstream(spec: str) -> dict[str, list[str]]:
    """``"think:full,jev;fast:full"`` -> {"think": ["full", "jev"], "fast": ["full"]}."""
    out: dict[str, list[str]] = {}
    for part in spec.split(";"):
        if not part.strip():
            continue
        gen, _, conds = part.partition(":")
        if gen.strip() not in ("think", "fast"):
            raise ValueError(f"unknown generator {gen!r}")
        out[gen.strip()] = [c.strip() for c in conds.split(",") if c.strip()]
    return out


def downstream(
    sub: Sequence[Statement],
    kept: dict[str, dict[str, frozenset[str] | None]],
    conditions: Sequence[str],
    thinking: bool,
    cache: ResponseCache,
    budget: Budget,
    chat: DeepSeekChat,
    workers: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Generate, solve and score the MILP for every (statement, condition) with one generator."""
    refs = {s.stmt_id: reference_optimum(s.instance).objective for s in sub}
    jobs = [
        (s.stmt_id, c, s.text(kept[c][s.stmt_id])) for s in sub for c in conditions if c in kept
    ]
    # Identical texts (e.g. a filter that keeps exactly the gold set) are generated once, so no two
    # threads race on the same cache key and a rerun replays exactly what was reported.
    texts = sorted({text for _, _, text in jobs})

    def run(text: str) -> GenerationResult:
        return generate_and_solve(text, chat, cache, budget=budget, thinking=thinking)

    with ThreadPoolExecutor(workers) as pool:
        gen_by_text = dict(zip(texts, pool.map(run, texts), strict=True))
    by: dict[str, dict[str, GenerationResult]] = {}
    for sid, c, text in jobs:
        by.setdefault(c, {})[sid] = gen_by_text[text]
    conds = [c for c in conditions if c in by]
    per: dict[str, dict[str, Any]] = {}
    for s in sub:
        ref = refs[s.stmt_id]
        row: dict[str, Any] = {"reference_objective": ref}
        for c in conds:
            g = by[c][s.stmt_id]
            keep = kept[c][s.stmt_id]
            row[c] = {
                "dropped_needed": 0 if keep is None else len(s.needed_ids() - keep),
                "correct": g.optimal and objective_matches(g.objective or 0.0, ref),
                "correct_first": g.first_optimal
                and objective_matches(g.first_objective or 0.0, ref),
                "optimal": g.optimal,
                "objective": g.objective,
                "attempts": len(g.attempts),
                "prompt_tokens_first": g.prompt_tokens_first,
                "problems": [a["problem"] for a in g.attempts if a["problem"]],
                "error": g.error,
            }
        per[s.stmt_id] = row
    summary: dict[str, Any] = {
        "generator": f"deepseek-flash[{'think' if thinking else 'fast'}]",
        "statements": len(sub),
        "unique_generations": len(texts),
        "cost_unique_usd": round(sum(g.cost_usd for g in gen_by_text.values()), 5),
        "conditions": {},
    }
    for c in conds:
        gs = [by[c][s.stmt_id] for s in sub]
        rows = [per[s.stmt_id][c] for s in sub]
        summary["conditions"][c] = {
            "first_valid": sum(g.first_valid for g in gs),
            "first_optimal": sum(g.first_optimal for g in gs),
            "optimal": sum(g.optimal for g in gs),
            "correct_first": sum(r["correct_first"] for r in rows),
            "correct": sum(r["correct"] for r in rows),
            "stmts_with_dropped_needed": sum(r["dropped_needed"] > 0 for r in rows),
            "correct_when_needed_dropped": sum(
                r["correct"] for r in rows if r["dropped_needed"] > 0
            ),
            "attempts_mean": round(statistics.fmean(len(g.attempts) for g in gs), 3),
            "prompt_tokens_first_mean": round(statistics.fmean(g.prompt_tokens_first for g in gs)),
            "prompt_tokens_total": sum(g.prompt_tokens for g in gs),
            "completion_tokens_total": sum(g.completion_tokens for g in gs),
            "cost_usd": round(sum(g.cost_usd for g in gs), 5),
            "latency_ms_mean": round(statistics.fmean(g.latency_ms for g in gs), 1),
            "api_errors": sum(g.error is not None for g in gs),
        }
    if "full" in by:
        for c in conds:
            if c == "full":
                continue
            b = sum(per[s]["full"]["correct"] and not per[s][c]["correct"] for s in per)
            cc = sum(not per[s]["full"]["correct"] and per[s][c]["correct"] for s in per)
            summary["conditions"][c]["vs_full"] = {
                "full_only_correct": b,
                "only_this_correct": cc,
                "mcnemar_exact_p": round(mcnemar_exact_p(b, cc), 4),
                "prompt_token_reduction_first": round(
                    1
                    - summary["conditions"][c]["prompt_tokens_first_mean"]
                    / summary["conditions"]["full"]["prompt_tokens_first_mean"],
                    4,
                ),
            }
    return summary, per


def _fmt(v: Any) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) < 1 else f"{v:g}"
    return str(v)


def write_markdown(path: Path, summary: dict[str, Any], example: dict[str, Any]) -> None:
    arms = summary["arms"]
    lines = [
        "# P2-H2b, relevance-filtering half: Jev as a relevance filter in front of an MILP "
        "generator",
        "",
        f"Generated by `experiments/h2/run_filtering.py` on {summary['generated_utc']} "
        f"(seed {summary['seed']}, {summary['n_statements']} statements: "
        f"{summary['n_dev']} dev / {summary['n_test']} test). Do not edit by hand.",
        "",
        "Hypothesis labels (`docs/hypotheses.md` v0.1): **P2-H2b** = requirement -> "
        "constraint-family routing (`results/h2/routing_summary.md`) plus relevance filtering "
        "(this file); **P2-H2c** (semantic pre-solve gate) is not yet tested.",
        "",
        "Statements are synthetic (single-track corridor, 2-4 trains, 2-3 sections) with gold "
        "needed/not-needed labels per sentence. *Data* = data-bearing sentences (times, headway, "
        "weights, rules, incidents); *needed* = data + frame (line, occupancy, overtaking, "
        "objective). *Trap* = not-needed sentences that contain a digit. Thresholded arms use the "
        "largest threshold with perfect recall of needed sentences on the dev split. Corpus: "
        f"`{json.dumps(summary['corpus'])}`.",
        "",
        "## Filter arms on the test split",
        "",
        "| arm | threshold | recall data | recall needed | precision | trap kept | near-miss kept "
        "| superseded kept | chars removed | stmts with all needed kept | latency p50 ms "
        "| USD / stmt |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for a in arms.values():
        t = a["test"]
        lines.append(
            f"| {a['arm']} | {_fmt(a['threshold'])} | {_fmt(t['recall_data'])} | "
            f"{_fmt(t['recall_needed'])} | {_fmt(t['precision'])} | {_fmt(t['trap_kept'])} | "
            f"{_fmt(t['near_miss_kept'])} | {_fmt(t['superseded_kept'])} | "
            f"{_fmt(t['char_reduction'])} | {_fmt(t['stmt_all_needed_kept'])} | "
            f"{a['latency_ms_p50'] or 0:.0f} | {a['cost_usd_per_statement'] or 0:.6f} |"
        )
    for arm in ("jev", "jev-bare"):
        if arm in arms and "test_at_0.5" in arms[arm]:
            t = arms[arm]["test_at_0.5"]
            lines.append(
                f"| {arm} @0.5 | 0.5 | {_fmt(t['recall_data'])} | {_fmt(t['recall_needed'])} | "
                f"{_fmt(t['precision'])} | {_fmt(t['trap_kept'])} | {_fmt(t['near_miss_kept'])} | "
                f"{_fmt(t['superseded_kept'])} | {_fmt(t['char_reduction'])} | "
                f"{_fmt(t['stmt_all_needed_kept'])} | | |"
            )
    for arm in THRESHOLD_ARMS:
        if arm not in arms:
            continue
        lines += [
            "",
            f"## Threshold sweep: {arm} (all {summary['n_statements']} statements)",
            "",
            "| threshold | recall data | recall needed | precision | trap kept | near-miss kept "
            "| superseded kept | chars removed | stmts all needed kept |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in arms[arm]["all_sweep"]:
            lines.append(
                f"| {r['threshold']} | {_fmt(r['recall_data'])} | {_fmt(r['recall_needed'])} | "
                f"{_fmt(r['precision'])} | {_fmt(r['trap_kept'])} | {_fmt(r['near_miss_kept'])} "
                f"| {_fmt(r['superseded_kept'])} | {_fmt(r['char_reduction'])} | "
                f"{_fmt(r['stmt_all_needed_kept'])} |"
            )
    lines += [
        "",
        "Latency is client wall-clock time per statement. Jev, DeepSeek and the embedding arm "
        "each send one request per statement (the embedding arm embeds all sentences of a "
        "statement in one batched request; its once-per-run query embedding and the rate-limit "
        "pauses are excluded). The regex arm is timed live and changes on every run.",
    ]
    conds_doc = summary.get("downstream_conditions") or {}
    for ds in (summary.get("downstream") or {}).values():
        lines += [
            "",
            f"## Downstream, {ds['generator']} writes the MILP ({ds['statements']} test "
            "statements)",
            "",
            "Correct = HiGHS optimum equals the brute-force optimum of the instance. Up to 3 "
            "attempts with error feedback (invalid JSON/schema, non-optimal status). Counts out of "
            f"{ds['statements']}. Prompt tokens include the fixed system prompt (the template), so "
            "the prompt-token cut is smaller than the statement's character cut. Identical texts "
            f"are generated once ({ds['unique_generations']} unique generations, "
            f"{ds['cost_unique_usd']:.4f} USD); the USD column attributes each to every condition "
            "that used it.",
            "",
            *(f"- `{c}`: {conds_doc[c]}." for c in ds["conditions"] if c in conds_doc),
            "",
            "| input | first valid | first optimal | correct (1st) | correct (≤3) | stmts with "
            "needed dropped | attempts | prompt tokens (1st) | USD | vs full: full-only / "
            "this-only (p) | prompt-token cut |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for c, v in ds["conditions"].items():
            vs = v.get("vs_full")
            vs_s = (
                f"{vs['full_only_correct']} / {vs['only_this_correct']} ({vs['mcnemar_exact_p']})"
                if vs
                else "–"
            )
            cut = _fmt(vs["prompt_token_reduction_first"]) if vs else "–"
            label = f"{c} (post-hoc margin)" if c in POST_HOC else c
            lines.append(
                f"| {label} | {v['first_valid']} | {v['first_optimal']} | {v['correct_first']} | "
                f"{v['correct']} | {v['stmts_with_dropped_needed']} | {v['attempts_mean']} | "
                f"{v['prompt_tokens_first_mean']} | {v['cost_usd']:.4f} | {vs_s} | {cut} |"
            )
    lines += [
        "",
        "## Cost",
        "",
        f"Fresh spend of the last invocation: `{json.dumps(summary['spend_fresh_usd'])}`. Cost of "
        f"all calls behind these numbers (fresh or cached): "
        f"`{json.dumps(summary['cost_all_calls_usd'])}`.",
        "",
        "## Example statement (first test statement; gold label and Jev noul per sentence)",
        "",
        "| id | kind | needed | Jev noul | sentence |",
        "|---|---|---|---|---|",
    ]
    for row in example["sentences"]:
        lines.append(
            f"| {row['sid']} | {row['kind']} | {'yes' if row['needed'] else 'no'} | "
            f"{_fmt(row['jev'])} | {row['text']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--n-dev", type=int, default=40)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--arms", default=",".join(ALL_ARMS))
    ap.add_argument("--downstream-n", type=int, default=20)
    ap.add_argument("--downstream", default=DEFAULT_DOWNSTREAM, help="gen:cond,...;gen:cond,...")
    ap.add_argument(
        "--offline",
        action="store_true",
        help="replay every arm from the caches only (a miss stops the run); writes to "
        "results/h2/offline/ unless --out-dir is given",
    )
    ap.add_argument(
        "--render-only",
        action="store_true",
        help="re-render filtering_summary.md from the existing filtering_summary.json; no calls",
    )
    ap.add_argument("--deepseek-cap", type=float, default=1.00)
    ap.add_argument("--jev-cap", type=float, default=0.20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "runs" / "cache")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default results/h2 (--offline: results/h2/offline)",
    )
    args = ap.parse_args(argv)
    if args.out_dir is None:
        args.out_dir = ROOT / "results" / "h2" / ("offline" if args.offline else "")
    if args.render_only:
        return render_only(args.out_dir)

    arms = [a for a in args.arms.split(",") if a]
    cache_cls: type[ResponseCache] = ReplayCache if args.offline else ResponseCache
    if not args.offline:
        _setup_keys()
    stmts = make_statements(args.n, args.seed)
    dev, test = stmts[: args.n_dev], stmts[args.n_dev :]
    budget = Budget(caps={"deepseek": args.deepseek_cap, "jev": args.jev_cap})
    chat = DeepSeekChat()

    scores: dict[str, dict[str, FilterScores]] = {}
    reports: dict[str, Any] = {}
    for arm in arms:
        scores[arm] = score_arm(
            arm, stmts, args.cache_dir, budget, chat, args.workers, cache_cls=cache_cls
        )
        reports[arm] = arm_report(arm, dev, test, scores[arm])
        t = reports[arm]["test"]
        print(
            f"{arm:9s} th={reports[arm]['threshold']} recall_data={t['recall_data']}"
            f" recall_needed={t['recall_needed']} precision={t['precision']}"
            f" trap_kept={t['trap_kept']} chars_removed={t['char_reduction']}"
            f" errors={reports[arm]['errors']}",
            flush=True,
        )

    ds_summary: dict[str, Any] = {}
    ds_rows: dict[str, dict[str, Any]] = {}
    if args.downstream_n > 0:
        sub = test[: args.downstream_n]
        plan = parse_downstream(args.downstream)
        kept: dict[str, dict[str, frozenset[str] | None]] = {}
        for c in sorted({c for conds in plan.values() for c in conds}):
            if c == "full":
                kept[c] = {s.stmt_id: None for s in sub}
            elif c == "gold":
                kept[c] = {s.stmt_id: s.needed_ids() for s in sub}
            elif "@" in c:  # an arm at a fixed threshold, e.g. jev@0.3 (a safety margin)
                arm, th_s = c.split("@")
                kept[c] = {s.stmt_id: kept_ids(scores[arm][s.stmt_id], float(th_s)) for s in sub}
            elif c in scores:
                th = reports[c]["threshold"] if reports[c]["threshold"] is not None else 0.5
                kept[c] = {s.stmt_id: kept_ids(scores[c][s.stmt_id], th) for s in sub}
            else:
                raise ValueError(f"unknown downstream condition {c!r}")
        milp_cache = cache_cls(args.cache_dir / "h2-filtering-milp.jsonl")
        for gen, conds in plan.items():
            summ, per = downstream(
                sub, kept, conds, gen == "think", milp_cache, budget, chat, args.workers
            )
            if "jev" in kept:
                summ["jev_filter_threshold"] = reports["jev"]["threshold"]
                summ["jev_filter_cost_usd"] = round(
                    sum(scores["jev"][s.stmt_id].cost_usd for s in sub), 6
                )
            ds_summary[gen] = summ
            for sid, row in per.items():
                ds_rows.setdefault(sid, {})[gen] = row
            for c, v in summ["conditions"].items():
                print(f"downstream {gen} {c:9s} {json.dumps(v)}", flush=True)

    per_stmt: list[dict[str, Any]] = []
    dev_ids = {s.stmt_id for s in dev}
    for s in stmts:
        row: dict[str, Any] = {
            "stmt_id": s.stmt_id,
            "split": "dev" if s.stmt_id in dev_ids else "test",
            "sentences": len(s.sentences),
            "needed": len(s.needed_ids()),
            "data": len(s.data_ids()),
            "trap": len(s.trap_ids()),
            "trains": len(s.instance["trains"]),
            "sections": len(s.instance["sections"]),
            "filters": {},
        }
        for arm in arms:
            th = reports[arm]["threshold"]
            m = sentence_metrics(s, kept_ids(scores[arm][s.stmt_id], th if th is not None else 0.5))
            row["filters"][arm] = {"kept": m.kept, "dropped_needed": m.fn, "kept_noise": m.fp}
        if s.stmt_id in ds_rows:
            row["downstream"] = ds_rows[s.stmt_id]
        per_stmt.append(row)

    ex = test[0] if test else stmts[0]
    example = {
        "stmt_id": ex.stmt_id,
        "sentences": [
            {
                "sid": x.sid,
                "kind": x.kind,
                "needed": x.needed,
                "jev": round(scores["jev"][ex.stmt_id].scores[x.sid], 3)
                if "jev" in scores
                else None,
                "text": x.text,
            }
            for x in ex.sentences
        ],
    }
    cost_all: dict[str, float] = {a: reports[a]["cost_usd"] for a in arms if reports[a]["cost_usd"]}
    for gen, summ in ds_summary.items():
        cost_all[f"downstream-{gen}"] = summ["cost_unique_usd"]
    summary = {
        **LABELS,
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": args.seed,
        "n_statements": len(stmts),
        "n_dev": len(dev),
        "n_test": len(test),
        "prices": {
            "jev_usd_per_input_token": JEV_PRICE_PER_INPUT_TOKEN,
            "deepseek": "d4r.engines.llm.deepseek_cost (off-peak list price, peak x2)",
            "scads_embeddings": "no charge (university endpoint)",
        },
        "corpus": {
            "sentences": sum(len(s.sentences) for s in stmts),
            "needed": sum(len(s.needed_ids()) for s in stmts),
            "data": sum(len(s.data_ids()) for s in stmts),
            "trap": sum(len(s.trap_ids()) for s in stmts),
            "near_miss": sum(x.kind == "near_miss" for s in stmts for x in s.sentences),
        },
        "arms": reports,
        "downstream": ds_summary,
        "spend_fresh_usd": {k: round(v, 6) for k, v in budget.spent.items()},
        "cost_all_calls_usd": {k: round(v, 6) for k, v in cost_all.items()},
        "example": example,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "filtering_summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    (args.out_dir / "filtering_per_statement.json").write_text(
        json.dumps(per_stmt, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    write_markdown(args.out_dir / "filtering_summary.md", summary, example)
    print(f"spend (fresh): {summary['spend_fresh_usd']}; wrote {args.out_dir}", flush=True)
    return 0


def render_only(out_dir: Path) -> int:
    """Refresh the static labels of filtering_summary.json and re-render the .md; no calls."""
    path = out_dir / "filtering_summary.json"
    old = json.loads(path.read_text(encoding="utf-8"))
    summary = {**LABELS, **{k: v for k, v in old.items() if k not in LABELS}}
    path.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    write_markdown(out_dir / "filtering_summary.md", summary, summary["example"])
    print(f"re-rendered {out_dir / 'filtering_summary.md'} from {path.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
