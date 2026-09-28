"""Run the P2-H2a and P2-H2b routing experiments end to end and write results/h2/routing_*.

Why a script and not a notebook: every number in results/h2/routing_summary.{md,json} must be
regenerable from the response caches (runs/cache/h2-routing-*.jsonl). A rerun replays cached
answers for free; only missing requests go to the APIs, under per-provider live-spend caps.

    python experiments/h2/run_routing.py                 # both parts, live where not cached
    python experiments/h2/run_routing.py --part h2b      # only the synthetic benchmark
    python experiments/h2/run_routing.py --offline       # cache only, no network
    python experiments/h2/run_routing.py --limit 5       # smoke test on a few items

P2-H2a reads the private literature catalogue from ``D4R_CATALOG_DIR`` (default
``~/LiteratureAssistant/optimization_model_catalog``) and is skipped if it is absent; only
aggregate metrics are written. Its class names and class descriptions are private as well: the
descriptions come from ``D4R_CLASS_DESCRIPTIONS`` (default
``~/.local/share/d4r/h2a_class_descriptions.json``), the results name classes by opaque ids
(``C01``, ``V01``, ``P01``, ``O01``, numbered by sorted class name per kind), and the id -> name
mapping is written only next to the private descriptions file. The H2a response caches carry
class labels and are gitignored. P2-H2b is fully synthetic; its statements are written out too.
Keys come from ``D4R_SECRETS_FILE`` (default ``~/.config/raiLP/secrets.env``) and are never printed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

from d4r.apikeys import load_secrets
from d4r.engines.cache import ResponseCache
from d4r.h2.requirements import (
    Statement,
    constraint_families,
    family_descriptions,
    generate,
    keyword_route,
)
from d4r.h2.routing import (
    DS_MODEL,
    EMB_MODEL,
    JEV_MODEL,
    Answer,
    Budget,
    CatalogItem,
    DeepSeekRouter,
    EmbeddingUnavailable,
    JevRouter,
    ScadsEmbedder,
    accuracy,
    calibration_bins,
    catalog_dir,
    class_descriptions,
    class_ids,
    class_ids_path,
    gate_curve,
    gate_cv,
    load_catalog,
    macro_f1,
    multilabel_scores,
    negation_scores,
    paired_bootstrap,
    per_class_f1,
    per_label_predict,
    percentile,
    tfidf_similarities,
    threshold_predict,
    topk_predict,
    tune_per_label_thresholds,
    tune_threshold,
)

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "runs" / "cache"
RESULTS = ROOT / "results" / "h2"
KINDS = ("constraints", "variables", "parameters", "objectives")
GATE_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
STYLES = ("modelling", "requirement")
SWEEP = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
GRID = tuple(round(0.01 * i, 2) for i in range(101))
EMB_INSTR_H2A = (
    "Instruct: Given an element of a railway-optimisation MILP model ({kind}), retrieve the"
    " taxonomy class it belongs to\nQuery: "
)
EMB_INSTR_H2B = (
    "Instruct: Given a railway rescheduling requirement statement, retrieve the constraint"
    " families an optimisation model needs to enforce it\nQuery: "
)

T = TypeVar("T")
R = TypeVar("R")


def pmap(fn: Callable[[T], R], xs: Sequence[T], workers: int) -> list[R]:
    """Order-preserving parallel map."""
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        return list(ex.map(fn, xs))


def _lat_cost(ans: Sequence[Answer]) -> dict[str, Any]:
    lat = [a.latency_ms for a in ans if a.latency_ms is not None]
    n = len(ans)
    models = Counter(a.model for a in ans if a.model)
    return {
        "p50_ms": percentile(lat, 50),
        "p95_ms": percentile(lat, 95),
        "usd_total": sum(a.cost_usd for a in ans),
        "usd_per_1k": 1000 * sum(a.cost_usd for a in ans) / n if n else None,
        "input_tokens": sum(a.input_tokens for a in ans),
        "output_tokens": sum(a.output_tokens for a in ans),
        "n_errors": sum(1 for a in ans if a.error),
        "served_model": models.most_common(1)[0][0] if models else "",
    }


def _argmax_answers(
    items: Sequence[str], labels: Sequence[str], sims: Sequence[tuple[list[float], float | None]]
) -> list[Answer]:
    out = []
    for iid, (s, ms) in zip(items, sims, strict=True):
        scores = dict(zip(labels, s, strict=True))
        out.append(Answer(iid, scores=scores, choice=topk_predict(scores, 1)[0], latency_ms=ms))
    return out


def _knn_loo(
    items: Sequence[CatalogItem], sims: Sequence[tuple[list[float], float | None]], k: int
) -> list[Answer]:
    out = []
    for i, (row, ms) in enumerate(sims):
        order = sorted((j for j in range(len(items)) if j != i), key=lambda j: (-row[j], j))[:k]
        votes = Counter(items[j].gold for j in order)
        top = max(votes.values())
        # tie -> the class of the nearest neighbour among the tied classes
        choice = next(items[j].gold for j in order if votes[items[j].gold] == top)
        out.append(Answer(items[i].item_id, choice=choice, latency_ms=ms))
    return out


# ---------------------------------------------------------------------------------------------
# P2-H2a
# ---------------------------------------------------------------------------------------------


def evaluate_single(
    y: Sequence[str], classes: Sequence[str], arms: dict[str, list[Answer]]
) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(y), "n_classes": len(classes), "arms": {}}
    for name, ans in arms.items():
        pred = [a.choice for a in ans]
        out["arms"][name] = {
            "accuracy": accuracy(y, pred),
            "macro_f1": macro_f1(y, pred, classes),
            "per_class_f1": per_class_f1(y, pred, classes),
            **_lat_cost(ans),
        }
    if "jev" in arms and "deepseek" in arms:
        J, D = arms["jev"], arms["deepseek"]
        jc = [a.choice for a in J]
        dc = [a.choice for a in D]
        jp = [a.p_top for a in J]
        cj = [a == b for a, b in zip(y, jc, strict=True)]
        cd = [a == b for a, b in zip(y, dc, strict=True)]

        def diff(idx: Sequence[int]) -> float:
            return (sum(cj[i] for i in idx) - sum(cd[i] for i in idx)) / len(idx)

        out["jev_minus_deepseek_acc"] = paired_bootstrap(diff, len(y), seed=0)
        out["discordant"] = {
            "jev_only_right": sum(a and not b for a, b in zip(cj, cd, strict=True)),
            "deepseek_only_right": sum(b and not a for a, b in zip(cj, cd, strict=True)),
            "agreement": sum(a == b for a, b in zip(jc, dc, strict=True)) / len(y),
        }
        out["gate"] = gate_curve(
            y,
            jc,
            jp,
            dc,
            classes,
            GATE_THRESHOLDS,
            fast_cost=[a.cost_usd for a in J],
            slow_cost=[a.cost_usd for a in D],
            fast_ms=[a.latency_ms for a in J],
            slow_ms=[a.latency_ms for a in D],
        )
        for row in out["gate"]:
            th = row["threshold"]
            cc = [d if (p is None or p < th) else j for j, d, p in zip(cj, cd, jp, strict=True)]

            def cdiff(idx: Sequence[int], cc: list[bool] = cc) -> float:
                return (sum(cc[i] for i in idx) - sum(cd[i] for i in idx)) / len(idx)

            row["cascade_minus_deepseek"] = paired_bootstrap(cdiff, len(y), n_boot=2000, seed=0)
        out["jev_calibration"] = calibration_bins(jp, cj, (0.0, 0.4, 0.6, 0.8, 0.9, 1.0))
        # Held-out gate estimate: theta chosen on the other folds, never on the scored items.
        out["gate_cv"] = gate_cv(
            y,
            jc,
            jp,
            dc,
            GATE_THRESHOLDS,
            k=5,
            n_splits=200,
            seed=0,
            fast_cost=[a.cost_usd for a in J],
            slow_cost=[a.cost_usd for a in D],
        )
    return out


def _with_class_ids(ev: dict[str, Any], ids: dict[str, str]) -> dict[str, Any]:
    """Replace private class names by their opaque ids in an evaluation (per-class F1 keys)."""
    for arm in ev["arms"].values():
        if "per_class_f1" in arm:
            arm["per_class_f1"] = {ids[c]: v for c, v in arm["per_class_f1"].items()}
    return ev


def _write_private_class_ids(id_maps: dict[str, dict[str, str]]) -> Path:
    """Write kind -> id -> class name next to the private descriptions file (mode 600)."""
    path = class_ids_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    data = {kind: {i: c for c, i in m.items()} for kind, m in id_maps.items()}
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    path.chmod(0o600)
    return path


def run_h2a(
    args: argparse.Namespace, jev: JevRouter, ds: DeepSeekRouter, emb: ScadsEmbedder
) -> dict[str, Any] | None:
    cdir = catalog_dir()
    sets: dict[str, Any] = {}
    pooled_y: list[str] = []
    pooled: dict[str, list[Answer]] = {}
    notes: list[str] = []
    id_maps: dict[str, dict[str, str]] = {}
    for kind in KINDS:
        items = load_catalog(kind, cdir)
        if items is None:
            notes.append(f"{kind}: catalogue file absent, skipped")
            continue
        # Opaque ids from the full class set, so a --limit smoke run uses the same ids.
        id_maps[kind] = class_ids(kind, [it.gold for it in items])
        if args.limit:
            items = items[: args.limit]
        classes = sorted({it.gold for it in items})
        desc = class_descriptions(kind, classes)
        undescribed = [c for c, d in desc.items() if d == c]
        if undescribed:
            notes.append(f"{kind}: {len(undescribed)} class(es) without a written description")
        ids = [it.item_id for it in items]
        arms: dict[str, list[Answer]] = {}
        if "jev" in args.arms:
            arms["jev"] = pmap(partial(jev.route_catalog, descriptions=desc), items, args.workers)
        if "deepseek" in args.arms:
            arms["deepseek"] = pmap(
                partial(ds.route_catalog, descriptions=desc), items, args.workers
            )
        docs = [f"{c}: {d}" for c, d in desc.items()]
        if "emb" in args.arms:
            try:
                instr = EMB_INSTR_H2A.format(kind=kind[:-1])
                sims = emb.similarities([instr + it.text for it in items], docs)
                arms["emb-nearest"] = _argmax_answers(ids, classes, sims)
                texts = [it.text for it in items]
                knn = emb.similarities(texts, texts)
                arms["emb-knn1-loo"] = _knn_loo(items, knn, 1)
                arms["emb-knn5-loo"] = _knn_loo(items, knn, 5)
            except EmbeddingUnavailable as exc:
                notes.append(f"{kind}: embedding arm unavailable ({exc})")
        tf = tfidf_similarities([it.text for it in items], docs)
        arms["tfidf-nearest"] = _argmax_answers(ids, classes, [(s, None) for s in tf])
        maj = sorted(Counter(it.gold for it in items).items(), key=lambda kv: (-kv[1], kv[0]))[0]
        arms["majority"] = [Answer(i, choice=maj[0]) for i in ids]
        y = [it.gold for it in items]
        sets[kind] = _with_class_ids(evaluate_single(y, classes, arms), id_maps[kind])
        pooled_y += [f"{kind}:{c}" for c in y]
        for name, ans in arms.items():
            if name == "majority":
                continue
            pooled.setdefault(name, [])
            pooled[name] += [
                replace(
                    a,
                    choice=f"{kind}:{a.choice}" if a.choice else None,
                    scores={f"{kind}:{k}": v for k, v in a.scores.items()} if a.scores else None,
                )
                for a in ans
            ]
        print(f"[h2a] {kind}: n={len(items)} " + _progress(arms), flush=True)
    if not sets:
        return {"skipped": True, "notes": notes}
    _write_private_class_ids(id_maps)
    full = [a for a, v in pooled.items() if len(v) == len(pooled_y)]
    pooled_eval = evaluate_single(pooled_y, sorted(set(pooled_y)), {a: pooled[a] for a in full})
    for arm in pooled_eval["arms"].values():
        arm.pop("per_class_f1", None)
    return {
        "catalog_dir_env": "D4R_CATALOG_DIR",
        "class_descriptions_env": "D4R_CLASS_DESCRIPTIONS",
        "class_ids": (
            "opaque ids (C01.., V01.., P01.., O01..) numbered by sorted class name per kind;"
            " class names, descriptions and the id -> name mapping are private"
        ),
        "sets": sets,
        "pooled": pooled_eval,
        "notes": notes,
    }


def _progress(arms: dict[str, list[Answer]]) -> str:
    return " ".join(
        f"{k}(live={sum(a.live for a in v)},err={sum(1 for a in v if a.error)})"
        for k, v in arms.items()
    )


# ---------------------------------------------------------------------------------------------
# P2-H2b
# ---------------------------------------------------------------------------------------------


def _eval_multi(
    stmts: Sequence[Statement],
    preds: Sequence[Sequence[str]],
    fams: Sequence[str],
    ans: Sequence[Answer] | None,
) -> dict[str, Any]:
    gold = [s.gold for s in stmts]
    out = multilabel_scores(gold, preds, fams)
    out.update(negation_scores(gold, [s.flipped for s in stmts], [s.waived for s in stmts], preds))
    if ans is not None:
        out.update(_lat_cost(ans))
    else:
        out.update({"p50_ms": None, "p95_ms": None, "usd_per_1k": 0.0, "usd_total": 0.0})
    return out


def _noul_profile(stmts: Sequence[Statement], ans: Sequence[Answer]) -> dict[str, Any]:
    """Mean Jev Noul probability by the role a family plays in the statement."""
    buckets: dict[str, list[float]] = {
        "gold_plain": [],
        "gold_prohibition": [],
        "waived": [],
        "absent": [],
    }
    for s, a in zip(stmts, ans, strict=True):
        if not a.scores:
            continue
        for f, p in a.scores.items():
            if f in s.flipped:
                buckets["gold_prohibition"].append(p)
            elif f in s.gold:
                buckets["gold_plain"].append(p)
            elif f in s.waived:
                buckets["waived"].append(p)
            else:
                buckets["absent"].append(p)
    return {
        k: {"n": len(v), "mean": sum(v) / len(v) if v else None, "p50": percentile(v, 50)}
        for k, v in buckets.items()
    }


def _sim_dicts(fams: Sequence[str], sims: Sequence[Sequence[float]]) -> list[dict[str, float]]:
    return [dict(zip(fams, x, strict=True)) for x in sims]


def _bootstrap_f1(
    test: Sequence[Statement],
    a: Sequence[Sequence[str]],
    b: Sequence[Sequence[str]],
    fams: Sequence[str],
) -> dict[str, float]:
    gold = [s.gold for s in test]

    def diff(idx: Sequence[int]) -> float:
        g = [gold[i] for i in idx]
        return (
            multilabel_scores(g, [a[i] for i in idx], fams)["micro_f1"]
            - multilabel_scores(g, [b[i] for i in idx], fams)["micro_f1"]
        )

    return paired_bootstrap(diff, len(test), n_boot=2000, seed=0)


def run_h2b(
    args: argparse.Namespace, jev: JevRouter, ds: DeepSeekRouter, emb: ScadsEmbedder
) -> dict[str, Any]:
    """All arms x both description styles; style, thresholds and top-k are chosen on dev only."""
    fams = list(constraint_families())
    descs = {st: family_descriptions(st) for st in STYLES}
    n_test = args.limit or args.n_test
    n_dev = args.limit or args.n_dev
    test = generate(n_test, args.seed, "test")
    dev = generate(n_dev, args.seed, "dev")
    notes: list[str] = []
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not args.limit:
        (RESULTS / "routing_h2b_statements.json").write_text(
            json.dumps(
                {
                    "seed": args.seed,
                    "families": fams,
                    "descriptions": descs,
                    "dev": [s.to_dict() for s in dev],
                    "test": [s.to_dict() for s in test],
                },
                indent=1,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    gold_dev = [s.gold for s in dev]
    # arm name -> (test predictions, answers for latency/cost, router, style, dev micro-F1)
    arms: dict[str, tuple[list[tuple[str, ...]], list[Answer] | None, str, str, float | None]] = {}
    tuned: dict[str, Any] = {}
    extra: dict[str, Any] = {"jev_sweep": {}, "jev_noul_profile": {}}

    def dev_f1(pred: Sequence[Sequence[str]]) -> float:
        return float(multilabel_scores(gold_dev, pred, fams)["micro_f1"])

    for style in STYLES:
        desc = descs[style]
        if "jev" in args.arms:
            route = partial(jev.route_requirement, descriptions=desc, style=style)
            jdev = pmap(lambda s, r=route: r(s.sid, s.text), dev, args.workers)
            jtest = pmap(lambda s, r=route: r(s.sid, s.text), test, args.workers)
            print(
                f"[h2b] jev[{style}] dev {_progress({'jev': jdev})} test "
                f"{_progress({'jev': jtest})}",
                flush=True,
            )
            zero = dict.fromkeys(fams, 0.0)
            jd = [a.scores or zero for a in jdev]
            jt = [a.scores or zero for a in jtest]
            th, f1 = tune_threshold(jd, gold_dev, fams, SWEEP)
            per = tune_per_label_thresholds(jd, gold_dev, fams, SWEEP)
            tuned[f"jev[{style}]"] = {"global": th, "per_family": per}
            arms[f"jev[{style}]@0.5"] = (
                [threshold_predict(s, 0.5) for s in jt],
                jtest,
                "jev",
                style,
                dev_f1([threshold_predict(s, 0.5) for s in jd]),
            )
            arms[f"jev[{style}]@{th:g} (dev)"] = (
                [threshold_predict(s, th) for s in jt],
                jtest,
                "jev",
                style,
                f1,
            )
            arms[f"jev[{style}]@per-family (dev)"] = (
                [per_label_predict(s, per) for s in jt],
                jtest,
                "jev",
                style,
                dev_f1([per_label_predict(s, per) for s in jd]),
            )
            extra["jev_sweep"][style] = [
                {
                    "threshold": t,
                    **_pick(_eval_multi(test, [threshold_predict(s, t) for s in jt], fams, None)),
                }
                for t in SWEEP
            ]
            extra["jev_noul_profile"][style] = _noul_profile(test, jtest)
        if "deepseek" in args.arms:
            route = partial(ds.route_requirement, descriptions=desc, style=style)
            ddev = pmap(lambda s, r=route: r(s.sid, s.text), dev, args.workers)
            dtest = pmap(lambda s, r=route: r(s.sid, s.text), test, args.workers)
            print(
                f"[h2b] deepseek[{style}] dev {_progress({'ds': ddev})} test "
                f"{_progress({'ds': dtest})}",
                flush=True,
            )
            arms[f"deepseek[{style}]"] = (
                [a.labels for a in dtest],
                dtest,
                "deepseek",
                style,
                dev_f1([a.labels for a in ddev]),
            )

        lead = "" if style == "modelling" else "a requirement asking for "
        docs = [f"{f.replace('_', ' ')}: {lead}{d}" for f, d in desc.items()]
        sim_sets: dict[str, tuple[list[dict[str, float]], list[dict[str, float]], list[Answer]]]
        sim_sets = {}
        if "emb" in args.arms:
            try:
                sd = emb.similarities([EMB_INSTR_H2B + s.text for s in dev], docs)
                st = emb.similarities([EMB_INSTR_H2B + s.text for s in test], docs)
                sim_sets["emb"] = (
                    _sim_dicts(fams, [x for x, _ in sd]),
                    _sim_dicts(fams, [x for x, _ in st]),
                    [Answer(s.sid, latency_ms=ms) for s, (_, ms) in zip(test, st, strict=True)],
                )
            except EmbeddingUnavailable as exc:
                notes.append(f"embedding arm unavailable ({exc}); TF-IDF baseline only")
        sim_sets["tfidf"] = (
            _sim_dicts(fams, tfidf_similarities([s.text for s in dev], docs)),
            _sim_dicts(fams, tfidf_similarities([s.text for s in test], docs)),
            [Answer(s.sid) for s in test],
        )
        for name, (sdv, stt, lat) in sim_sets.items():
            for k in (1, 2, 3):
                arms[f"{name}[{style}]-top{k}"] = (
                    [topk_predict(s, k) for s in stt],
                    lat,
                    name,
                    style,
                    dev_f1([topk_predict(s, k) for s in sdv]),
                )
            th, f1 = tune_threshold(sdv, gold_dev, fams, GRID, min_k=1)
            tuned[f"{name}[{style}]"] = th
            arms[f"{name}[{style}]@{th:g} (dev, >=1)"] = (
                [threshold_predict(s, th, min_k=1) for s in stt],
                lat,
                name,
                style,
                f1,
            )
    arms["keyword"] = (
        [keyword_route(s.text) for s in test],
        None,
        "keyword",
        "-",
        dev_f1([keyword_route(s.text) for s in dev]),
    )

    # Dev-only selection: per router, the (style, rule) with the best dev micro-F1.
    best: dict[str, str] = {}
    for name, (_, _, router, _, f1) in arms.items():
        if f1 is not None and (router not in best or f1 > arms[best[router]][4] + 1e-12):
            best[router] = name
    results: dict[str, Any] = {}
    for name, (pred, ans, router, style, f1) in arms.items():
        results[name] = {
            "router": router,
            "style": style,
            "dev_micro_f1": f1,
            "dev_selected": best.get(router) == name,
            **_eval_multi(test, pred, fams, ans),
        }
    if "jev" in best and "deepseek" in best:
        dsel = arms[best["deepseek"]][0]
        extra["bootstrap_micro_f1"] = {
            f"{best['jev']} minus {best['deepseek']}": _bootstrap_f1(
                test, arms[best["jev"]][0], dsel, fams
            ),
        }
        j05 = f"jev[{arms[best['jev']][3]}]@0.5"
        extra["bootstrap_micro_f1"][f"{j05} minus {best['deepseek']}"] = _bootstrap_f1(
            test, arms[j05][0], dsel, fams
        )
    return {
        "families": fams,
        "dataset": _dataset_stats(test, dev),
        "dev_selected": best,
        "tuned_thresholds": tuned,
        "arms": results,
        **extra,
        "notes": notes,
    }


def _pick(d: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "micro_p",
        "micro_r",
        "micro_f1",
        "macro_f1",
        "exact_match",
        "flip_recall",
        "waiver_fp_rate",
    )
    return {k: d[k] for k in keys}


def _dataset_stats(test: Sequence[Statement], dev: Sequence[Statement]) -> dict[str, Any]:
    def stats(ss: Sequence[Statement]) -> dict[str, Any]:
        return {
            "n": len(ss),
            "labels_per_statement": dict(sorted(Counter(len(s.gold) for s in ss).items())),
            "family_counts": dict(sorted(Counter(f for s in ss for f in s.gold).items())),
            "negated_statements": sum(s.negated for s in ss),
            "prohibition_mentions": sum(len(s.flipped) for s in ss),
            "waiver_mentions": sum(len(s.waived) for s in ss),
            "with_distractors": sum(1 for s in ss if s.n_distractors),
        }

    return {"test": stats(test), "dev": stats(dev)}


# ---------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------


def _f(x: Any, nd: int = 3) -> str:
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(_f(c) for c in r) + " |" for r in rows]
    return out


def render_md(rep: dict[str, Any]) -> str:
    L = [
        "# P2-H2a / P2-H2b: query routing for an MILP assistant",
        "",
        "Generated by `experiments/h2/run_routing.py` from the response caches"
        " `runs/cache/h2-routing-*.jsonl`; do not edit by hand. Models: Jev "
        f"`{JEV_MODEL}`, DeepSeek `{DS_MODEL}` (non-thinking, JSON, T=0), embeddings "
        f"`{EMB_MODEL}` (ScaDS.AI). Costs: list prices at call time; ScaDS embeddings,"
        " TF-IDF and keywords are free. Latency: client wall-clock per decision; embedding"
        " latencies are batch-averaged (wall time of a batch of up to 16 texts divided by its"
        " size, embedding the class or family descriptions excluded), so they are not per-call"
        " like Jev's and DeepSeek's.",
        "",
        "Hypothesis labels (`docs/hypotheses.md` v0.1): **P2-H2a** = closed-taxonomy routing;"
        " **P2-H2b** = requirement -> constraint-family routing (this file) plus relevance"
        " filtering (`results/h2/filtering_summary.md`); **P2-H2c** (semantic pre-solve gate)"
        " is not yet tested.",
        "",
    ]
    h2a = rep.get("h2a")
    if h2a and not h2a.get("skipped"):
        L += [
            "## P2-H2a: closed-taxonomy routing (literature catalogue)",
            "",
            "Rows of the private catalogue are routed to one class of its taxonomy. Only",
            "aggregate metrics are reported. `emb-knn*-loo` sees the gold labels of all other",
            "rows (supervised reference); `majority` is the majority-class floor. Classes are",
            "named by opaque ids in `routing_summary.json` (C01.., V01.., P01.., O01.., numbered",
            "by sorted class name per kind); the class names, their descriptions and the",
            "id -> name mapping are private and not part of this repository.",
            "",
        ]
        blocks = [("all sets pooled", h2a["pooled"]), *h2a["sets"].items()]
        for title, s in blocks:
            L += [f"### {title} (n = {s['n']}, {s['n_classes']} classes)", ""]
            rows = [
                [
                    a,
                    v["accuracy"],
                    v["macro_f1"],
                    v["p50_ms"],
                    v["p95_ms"],
                    v["usd_per_1k"],
                    v["n_errors"],
                ]
                for a, v in s["arms"].items()
            ]
            L += _table(
                ["arm", "accuracy", "macro-F1", "p50 ms", "p95 ms", "USD / 1k", "errors"], rows
            )
            if "jev_minus_deepseek_acc" in s:
                b = s["jev_minus_deepseek_acc"]
                d = s["discordant"]
                L += [
                    "",
                    f"Jev minus DeepSeek accuracy: {b['diff']:+.3f} (paired bootstrap 95% CI"
                    f" {b['ci95_lo']:+.3f} to {b['ci95_hi']:+.3f}); discordant pairs: Jev only"
                    f" right {d['jev_only_right']}, DeepSeek only right"
                    f" {d['deepseek_only_right']}; agreement {d['agreement']:.3f}.",
                    "",
                ]
                cv = s.get("gate_cv")
                if cv:
                    chosen = ", ".join(f"{t}: {c}" for t, c in cv["theta_chosen"].items())
                    L += [
                        "**Gate, headline (held-out theta):** Jev answers if p_top >= theta,"
                        f" else DeepSeek; theta is chosen by {cv['k']}-fold cross-validation"
                        f" ({cv['n_splits']} seeded random partitions; each fold uses the grid"
                        " threshold with the best cascade accuracy on the other folds). Cascade"
                        f" accuracy {cv['cascade_acc_mean']:.3f} (2.5-97.5 % over partitions"
                        f" {cv['cascade_acc_lo']:.3f}-{cv['cascade_acc_hi']:.3f}),"
                        f" {cv['cascade_minus_slow_mean']:+.3f} vs DeepSeek alone"
                        f" ({cv['slow_acc']:.3f}) and {cv['cascade_minus_fast_mean']:+.3f} vs"
                        f" Jev alone ({cv['fast_acc']:.3f}); mean Jev coverage"
                        f" {cv['coverage_mean']:.3f}, {_f(cv.get('usd_per_1k_mean'))} USD / 1k."
                        f" Thresholds chosen per fold: {chosen}.",
                        "",
                        "Full threshold curve (in-sample: each row is scored on the same items"
                        " the threshold is read from, so the best row, theta ="
                        f" {cv['in_sample_best_theta']:g} with {cv['in_sample_best_acc']:.3f}, is"
                        " the in-sample best, not a held-out estimate). Random escalation of the"
                        " same number of items (expectation and 95% range) and oracle"
                        " escalation as controls.",
                        "",
                    ]
                else:
                    L += [
                        "Gate: Jev answers if p_top >= theta, else DeepSeek. Random escalation"
                        " of the same number of items (expectation and 95% range) and oracle"
                        " escalation as controls.",
                        "",
                    ]
                grows = [
                    [
                        g["threshold"],
                        g["coverage"],
                        g["fast_acc_on_kept"],
                        g["cascade_acc"],
                        g["cascade_macro_f1"],
                        g["random_escalation_acc"],
                        f"{g['random_escalation_lo']:.3f}-{g['random_escalation_hi']:.3f}",
                        g["oracle_escalation_acc"],
                        "{:+.3f} ({:+.3f} to {:+.3f})".format(
                            g["cascade_minus_deepseek"]["diff"],
                            g["cascade_minus_deepseek"]["ci95_lo"],
                            g["cascade_minus_deepseek"]["ci95_hi"],
                        ),
                        g.get("usd_per_1k"),
                        g.get("p50_ms"),
                    ]
                    for g in s["gate"]
                ]
                L += _table(
                    [
                        "theta",
                        "Jev coverage",
                        "Jev acc (kept)",
                        "cascade acc",
                        "cascade mF1",
                        "random-esc acc",
                        "random 95%",
                        "oracle-esc acc",
                        "cascade - DeepSeek (95% CI)",
                        "USD / 1k",
                        "p50 ms",
                    ],
                    grows,
                )
                cal = s["jev_calibration"]
                L += [
                    "",
                    "Jev p_top calibration: "
                    + "; ".join(f"{c['bin']}: n={c['n']}, acc={_f(c['acc'], 2)}" for c in cal)
                    + ".",
                ]
            L.append("")
        if h2a.get("notes"):
            L += ["Notes: " + "; ".join(h2a["notes"]), ""]
    elif h2a:
        L += ["## P2-H2a", "", "Skipped: " + "; ".join(h2a.get("notes", [])), ""]

    h2b = rep.get("h2b")
    if h2b:
        ds = h2b["dataset"]["test"]
        L += [
            "## P2-H2b, routing half: requirement -> constraint families (synthetic, multi-label)",
            "",
            f"{ds['n']} held-out test statements; labels = lp2graph `ConstraintDomainClass`"
            f" without `unclassified` ({len(h2b['families'])} families). Labels per statement:"
            f" {ds['labels_per_statement']}; statements with negation: {ds['negated_statements']}"
            f" ({ds['prohibition_mentions']} prohibitions that still need their family,"
            f" {ds['waiver_mentions']} waivers that must not trigger theirs); with distractor"
            f" sentences: {ds['with_distractors']}. The {h2b['dataset']['dev']['n']} dev"
            " statements (disjoint templates) choose, per router, the description style"
            " (`modelling` = what the constraint does, `requirement` = what a statement asks"
            " for), the threshold or top-k; `*` marks the dev-selected variant. Dataset:"
            " `results/h2/routing_h2b_statements.json`.",
            "",
        ]
        rows = [
            [
                ("* " if v["dev_selected"] else "") + a,
                v["dev_micro_f1"],
                v["micro_p"],
                v["micro_r"],
                v["micro_f1"],
                v["macro_f1"],
                v["exact_match"],
                v["flip_recall"],
                v["waiver_fp_rate"],
                v["exact_match_negated"],
                v["p50_ms"],
                v["p95_ms"],
                v["usd_per_1k"],
            ]
            for a, v in h2b["arms"].items()
        ]
        L += _table(
            [
                "arm",
                "dev micro-F1",
                "micro-P",
                "micro-R",
                "micro-F1",
                "macro-F1",
                "exact set",
                "prohibition recall",
                "waiver FP rate",
                "exact (negated)",
                "p50 ms",
                "p95 ms",
                "USD / 1k",
            ],
            rows,
        )
        for label, b in h2b.get("bootstrap_micro_f1", {}).items():
            L += [
                "",
                f"{label}, micro-F1: {b['diff']:+.3f} (paired bootstrap 95% CI"
                f" {b['ci95_lo']:+.3f} to {b['ci95_hi']:+.3f}, {b['n_boot']} resamples).",
            ]
        for style, sweep in h2b.get("jev_sweep", {}).items():
            L += ["", f"Jev[{style}] Noul threshold sweep (test, for reference only):", ""]
            L += _table(
                [
                    "theta",
                    "micro-P",
                    "micro-R",
                    "micro-F1",
                    "macro-F1",
                    "exact set",
                    "prohibition recall",
                    "waiver FP rate",
                ],
                [
                    [
                        r["threshold"],
                        r["micro_p"],
                        r["micro_r"],
                        r["micro_f1"],
                        r["macro_f1"],
                        r["exact_match"],
                        r["flip_recall"],
                        r["waiver_fp_rate"],
                    ]
                    for r in sweep
                ],
            )
            prof = h2b["jev_noul_profile"][style]
            L += [
                "",
                f"Mean Jev[{style}] Noul probability by the family's role in the statement: "
                + "; ".join(f"{k} {_f(v['mean'])} (n={v['n']})" for k, v in prof.items())
                + ".",
            ]
        sel = [a for a, v in h2b["arms"].items() if v["dev_selected"]]
        fam_rows = [[f, *(h2b["arms"][a]["per_label_f1"][f] for a in sel)] for f in h2b["families"]]
        L += ["", "Per-family F1 of the dev-selected arms (test):", ""]
        L += _table(["family", *sel], fam_rows)
        L += ["", f"Dev-tuned thresholds: {json.dumps(h2b['tuned_thresholds'])}."]
        if h2b.get("notes"):
            L += ["", "Notes: " + "; ".join(h2b["notes"])]
        L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--part", choices=("h2a", "h2b", "all"), default="all")
    ap.add_argument("--arms", default="jev,deepseek,emb", help="comma list of paid/remote arms")
    ap.add_argument("--offline", action="store_true", help="replay caches only")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N items per set")
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--n-dev", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--jev-budget", type=float, default=0.20)
    ap.add_argument("--deepseek-budget", type=float, default=1.00)
    args = ap.parse_args(argv)
    args.arms = {a.strip() for a in args.arms.split(",") if a.strip()}

    os.environ.setdefault("D4R_SECRETS_FILE", os.path.expanduser("~/.config/raiLP/secrets.env"))
    load_secrets()
    budget = Budget({"jev": args.jev_budget, "deepseek": args.deepseek_budget})
    suffix = "-smoke" if args.limit else ""
    jev = JevRouter(ResponseCache(CACHE_DIR / "h2-routing-jev.jsonl"), budget, offline=args.offline)
    ds = DeepSeekRouter(
        ResponseCache(CACHE_DIR / "h2-routing-deepseek.jsonl"),
        budget,
        offline=args.offline,
    )
    emb = ScadsEmbedder(ResponseCache(CACHE_DIR / "h2-routing-emb.jsonl"), offline=args.offline)

    out_json = RESULTS / f"routing_summary{suffix}.json"
    rep: dict[str, Any] = {}
    if out_json.exists():
        rep = json.loads(out_json.read_text(encoding="utf-8"))
    rep["config"] = {
        "jev_model": JEV_MODEL,
        "deepseek_model": DS_MODEL,
        "embedding_model": EMB_MODEL,
        "seed": args.seed,
        "n_test": args.limit or args.n_test,
        "n_dev": args.limit or args.n_dev,
        "gate_thresholds": list(GATE_THRESHOLDS),
    }
    if args.part in ("h2a", "all"):
        rep["h2a"] = run_h2a(args, jev, ds, emb)
    if args.part in ("h2b", "all"):
        rep["h2b"] = run_h2b(args, jev, ds, emb)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (RESULTS / f"routing_summary{suffix}.md").write_text(render_md(rep), encoding="utf-8")
    print(f"live spend this run (USD): { {k: round(v, 4) for k, v in budget.spent.items()} }")
    print(f"wrote {out_json.relative_to(ROOT)} and .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
