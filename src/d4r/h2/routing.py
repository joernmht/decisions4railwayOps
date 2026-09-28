"""Query routing for an MILP-modelling assistant: typed model vs LLM vs embeddings (P2-H2a/b).

Why: an MILP assistant must map free text onto closed label spaces before it can retrieve the
right templates. Two modes are tested (docs/hypotheses.md):

- **P2-H2a, closed-taxonomy routing** ("alternative to RAG" in the bounded sense): each element
  of a published railway MILP (a constraint, variable, parameter or objective) is routed to one
  class of a fixed taxonomy. Arms: Jev Choice, DeepSeek JSON, embedding nearest-description,
  TF-IDF nearest-description, and a leave-one-out embedding kNN that sees gold labels (a
  supervised reference, not a zero-shot arm). The confidence gate "Jev acts if p_top >= theta,
  otherwise escalate to the LLM" is compared with random escalation at the same rate.
- **P2-H2b, requirement -> constraint families** ("driver of RAG"): synthetic statements from
  :mod:`d4r.h2.requirements` get a multi-label set of lp2graph constraint families. Jev answers
  one Noul per family, all in one request (speculative fan-out).

The catalogue for P2-H2a lives in a *private* repository. It is read at runtime from
``D4R_CATALOG_DIR``; nothing but aggregate metrics leaves the process. Its class names and the
class descriptions written for them are private too: they are read at runtime from
``D4R_CLASS_DESCRIPTIONS`` (default ``~/.local/share/d4r/h2a_class_descriptions.json``, outside
every repository), public results name classes by opaque ids (``C01``, ``V01``, ...) and the
H2a response caches, which hold class labels, are gitignored.

Everything paid is cached (:class:`~d4r.engines.cache.ResponseCache`) and guarded by a per-provider
budget. Pure functions (metrics, parsers, prompt builders, TF-IDF) are tested offline.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from d4r.apikeys import load_secrets
from d4r.engines.cache import ResponseCache, request_key
from d4r.engines.jev import JEV_PRICE_PER_INPUT_TOKEN
from d4r.engines.llm import DeepSeekChat, deepseek_cost

__all__ = [
    "CATALOG_FIELDS",
    "CLASS_ID_PREFIX",
    "Answer",
    "Budget",
    "BudgetExceeded",
    "CacheMiss",
    "CatalogItem",
    "DeepSeekRouter",
    "EmbeddingUnavailable",
    "JevRouter",
    "ScadsEmbedder",
    "accuracy",
    "calibration_bins",
    "catalog_dir",
    "catalog_messages",
    "class_descriptions",
    "class_descriptions_path",
    "class_ids",
    "class_ids_path",
    "gate_curve",
    "gate_cv",
    "jev_catalog_question",
    "jev_family_questions",
    "load_catalog",
    "load_class_descriptions",
    "macro_f1",
    "multilabel_scores",
    "negation_scores",
    "paired_bootstrap",
    "parse_multi_label",
    "parse_single_label",
    "per_class_f1",
    "per_label_predict",
    "percentile",
    "requirement_messages",
    "tfidf_similarities",
    "threshold_predict",
    "topk_predict",
    "tune_per_label_thresholds",
    "tune_threshold",
]

JEV_MODEL = "jev-1.13.0"
DS_MODEL = "deepseek-flash"
EMB_MODEL = "Qwen/Qwen3-Embedding-4B"
SCADS_BASE_URL = "https://llm.scads.ai/v1"
DEFAULT_CATALOG_DIR = "~/LiteratureAssistant/optimization_model_catalog"
DEFAULT_CLASS_DESCRIPTIONS = "~/.local/share/d4r/h2a_class_descriptions.json"
CLASS_IDS_FILENAME = "h2a_class_ids.json"

# ---------------------------------------------------------------------------------------------
# P2-H2a: the closed catalogue taxonomy
# ---------------------------------------------------------------------------------------------

#: Text fields of each catalogue CSV that a router sees (never ``paper`` or the labels).
CATALOG_FIELDS: dict[str, tuple[str, ...]] = {
    "constraints": ("original_short_name", "verbatim_or_compact_form", "description"),
    "variables": ("variable_type", "description"),
    "parameters": ("kind", "description"),
    "objectives": ("sense", "expression", "description"),
}
_SINGULAR = {
    "constraints": "constraint",
    "variables": "variable",
    "parameters": "parameter",
    "objectives": "objective",
}

#: Kind -> id prefix of the opaque class ids that public results use instead of class names.
CLASS_ID_PREFIX = {"constraints": "C", "variables": "V", "parameters": "P", "objectives": "O"}


def class_descriptions_path() -> Path:
    """Private class-description file: ``D4R_CLASS_DESCRIPTIONS`` or the default outside any repo.

    The class names come from the private catalogue, so they (and the descriptions written for
    them) stay out of this public repository until the catalogue owner decides otherwise.
    """
    return Path(os.environ.get("D4R_CLASS_DESCRIPTIONS", DEFAULT_CLASS_DESCRIPTIONS)).expanduser()


def class_ids_path() -> Path:
    """Private id -> class-name mapping, written next to the class-description file."""
    return class_descriptions_path().with_name(CLASS_IDS_FILENAME)


def load_class_descriptions(path: Path | None = None) -> dict[str, dict[str, str]]:
    """Kind -> class -> description from the private JSON file; ``{}`` if the file is absent.

    The descriptions were written from the class *names* and textbook OR knowledge only (never
    from catalogue rows). Read lazily at call time, never at import.
    """
    p = path or class_descriptions_path()
    if not p.is_file():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {str(k): {str(c): str(d) for c, d in v.items()} for k, v in data.items()}


def class_ids(kind: str, classes: Sequence[str]) -> dict[str, str]:
    """Class name -> opaque id (``C01``, ``V01``, ...), numbered by sorted class name per kind."""
    prefix = CLASS_ID_PREFIX[kind]
    return {c: f"{prefix}{i:02d}" for i, c in enumerate(sorted(set(classes)), start=1)}


@dataclass(frozen=True)
class CatalogItem:
    """One catalogue element: an opaque id, the text a router sees, and its gold class."""

    item_id: str
    kind: str
    text: str
    gold: str


def catalog_dir() -> Path:
    """Catalogue directory: ``D4R_CATALOG_DIR`` or the default private checkout."""
    return Path(os.environ.get("D4R_CATALOG_DIR", DEFAULT_CATALOG_DIR)).expanduser()


def load_catalog(kind: str, directory: Path | None = None) -> list[CatalogItem] | None:
    """Items of ``<kind>_classified.csv`` with a gold ``ml_class``; ``None`` if the file is absent.

    Ids are row positions (``constraints-007``), so no paper key or row text leaves the process.
    """
    path = (directory or catalog_dir()) / f"{kind}_classified.csv"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "ml_class" not in rows[0]:
        return None
    out = []
    for i, r in enumerate(rows):
        gold = (r.get("ml_class") or "").strip()
        if not gold:
            continue
        text = " | ".join(
            f"{k}: {r[k].strip()}" for k in CATALOG_FIELDS[kind] if (r.get(k) or "").strip()
        )
        out.append(CatalogItem(f"{kind}-{i:03d}", kind, text, gold))
    return out


def class_descriptions(
    kind: str,
    classes: Sequence[str],
    known: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, str]:
    """Class -> description for the classes present at runtime, sorted by class name.

    ``known`` defaults to :func:`load_class_descriptions`; a class without an entry (or every
    class, when the private file is absent) is described by its name alone.
    """
    table = load_class_descriptions() if known is None else known
    kind_known = table.get(kind, {})
    return {c: kind_known.get(c, c) for c in sorted(classes)}


def jev_catalog_question(kind: str, descriptions: Mapping[str, str]) -> dict[str, Any]:
    """Jev Choice over the taxonomy classes (criteria = class descriptions)."""
    return {
        "type": "choice",
        "instructions": (
            f"Which class of the railway-optimisation {_SINGULAR[kind]} taxonomy does `element`"
            " belong to?"
        ),
        "criteria": dict(descriptions),
    }


def jev_catalog_state(item: CatalogItem) -> dict[str, str]:
    return {"model_element_kind": _SINGULAR[item.kind], "element": item.text}


def catalog_messages(kind: str, descriptions: Mapping[str, str], text: str) -> list[dict[str, str]]:
    """DeepSeek messages for one catalogue element (same class descriptions as Jev)."""
    classes = "\n".join(f"- {c}: {d}" for c, d in descriptions.items())
    system = (
        f"You classify one {_SINGULAR[kind]} of a railway-optimisation MILP into exactly one"
        " class of a closed taxonomy.\nClasses:\n"
        + classes
        + '\nAnswer with a JSON object only: {"class": "<class name exactly as listed>"}.'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{_SINGULAR[kind]}: {text}"},
    ]


# ---------------------------------------------------------------------------------------------
# P2-H2b: requirement -> constraint families
# ---------------------------------------------------------------------------------------------

_YES = (
    "The statement asks for this kind of constraint to be enforced. A requirement phrased as a"
    " prohibition (something must not happen) counts."
)
_NO = (
    "The statement does not ask for this kind of constraint, or says it need not be considered,"
    " does not matter or can be ignored. Background facts and numbers alone do not count."
)


def _family_line(desc: str, style: str) -> str:
    return desc if style == "modelling" else f"the statement asks for {desc}"


def jev_family_questions(
    descriptions: Mapping[str, str], style: str = "modelling"
) -> dict[str, dict[str, Any]]:
    """One Noul per family, all sent in one request (speculative fan-out).

    ``modelling``: the question names a constraint kind by its modelling description;
    ``requirement``: a literal question about what the statement asks for (the phrasing the Jev
    docs recommend: state the exact condition). The yes/no criteria are the same in both.
    """
    if style == "modelling":
        tpl = (
            "Does `statement` state a requirement that the railway rescheduling model must"
            " enforce with a constraint of this kind? {desc}"
        )
    else:
        tpl = "Does `statement` ask for {desc}?"
    return {
        fam: {
            "type": "noul",
            "instructions": tpl.format(desc=desc),
            "criteria": {"true": _YES, "false": _NO},
        }
        for fam, desc in descriptions.items()
    }


def requirement_messages(
    descriptions: Mapping[str, str], text: str, style: str = "modelling"
) -> list[dict[str, str]]:
    """DeepSeek multi-label messages (same descriptions and yes/no rules as the Jev Nouls)."""
    fams = "\n".join(f"- {f}: {_family_line(d, style)}" for f, d in descriptions.items())
    system = (
        "You read a railway rescheduling requirement statement and decide which constraint"
        " families an optimisation model needs to enforce the requirements it states.\n"
        "Families (identifier: meaning):\n"
        + fams
        + "\nRules: a statement can need one or several families. "
        + _YES
        + " "
        + _NO
        + '\nAnswer with a JSON object only: {"families": [<identifiers exactly as listed>]}.'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": text}]


# ---------------------------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------------------------


def _norm(s: str) -> str:
    return re.sub(r"[\s_\-]+", " ", s.strip().strip('"').lower())


def _json_obj(content: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.S)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None


def parse_single_label(content: str, labels: Sequence[str]) -> str | None:
    """The class named in ``{"class": ...}``; exact, then case/space-insensitive match."""
    obj = _json_obj(content)
    if obj is None:
        return None
    val = obj.get("class", obj.get("label"))
    if not isinstance(val, str):
        return None
    if val in labels:
        return val
    by_norm = {_norm(lab): lab for lab in labels}
    return by_norm.get(_norm(val))


def parse_multi_label(content: str, labels: Sequence[str]) -> tuple[tuple[str, ...], int]:
    """Families in ``{"families": [...]}`` (in label order) and the number of unknown entries."""
    obj = _json_obj(content)
    if obj is None:
        return (), 0
    vals = obj.get("families", [])
    if isinstance(vals, str):
        vals = [vals]
    if not isinstance(vals, list):
        return (), 0
    by_norm = {_norm(lab): lab for lab in labels}
    found, unknown = set(), 0
    for v in vals:
        lab = by_norm.get(_norm(str(v)))
        if lab is None:
            unknown += 1
        else:
            found.add(lab)
    return tuple(lab for lab in labels if lab in found), unknown


# ---------------------------------------------------------------------------------------------
# Metrics (pure)
# ---------------------------------------------------------------------------------------------


def accuracy(y: Sequence[str], p: Sequence[str | None]) -> float:
    return sum(a == b for a, b in zip(y, p, strict=True)) / len(y) if y else float("nan")


def per_class_f1(
    y: Sequence[str], p: Sequence[str | None], classes: Sequence[str]
) -> dict[str, float]:
    out = {}
    for c in classes:
        tp = sum(1 for a, b in zip(y, p, strict=True) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, p, strict=True) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, p, strict=True) if a == c and b != c)
        out[c] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    return out


def macro_f1(y: Sequence[str], p: Sequence[str | None], classes: Sequence[str]) -> float:
    f = per_class_f1(y, p, classes)
    return sum(f.values()) / len(f) if f else float("nan")


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolation percentile (numpy's default), ``q`` in [0, 100]."""
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def paired_bootstrap(
    stat: Callable[[Sequence[int]], float], n: int, n_boot: int = 10000, seed: int = 0
) -> dict[str, float]:
    """Point estimate and 95% percentile interval of ``stat`` over item resamples.

    ``stat(idx)`` evaluates a paired difference (e.g. arm A minus arm B) on the items ``idx``.
    """
    rng = random.Random(seed)
    point = stat(list(range(n)))
    draws = sorted(stat([rng.randrange(n) for _ in range(n)]) for _ in range(n_boot))
    return {
        "diff": point,
        "ci95_lo": draws[int(0.025 * n_boot)],
        "ci95_hi": draws[min(n_boot - 1, int(0.975 * n_boot))],
        "n": n,
        "n_boot": n_boot,
    }


def calibration_bins(
    p_top: Sequence[float | None], correct: Sequence[bool], edges: Sequence[float]
) -> list[dict[str, Any]]:
    """Accuracy per p_top bin ``[edges[i], edges[i+1])`` (last bin closed)."""
    out = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        last = i == len(edges) - 2
        sel = [
            c
            for p, c in zip(p_top, correct, strict=True)
            if p is not None and lo <= p and (p < hi or (last and p <= hi))
        ]
        out.append(
            {
                "bin": f"{lo:.2f}-{hi:.2f}",
                "n": len(sel),
                "acc": sum(sel) / len(sel) if sel else None,
            }
        )
    return out


def gate_curve(
    y: Sequence[str],
    fast: Sequence[str | None],
    fast_p: Sequence[float | None],
    slow: Sequence[str | None],
    classes: Sequence[str],
    thresholds: Sequence[float],
    fast_cost: Sequence[float] | None = None,
    slow_cost: Sequence[float] | None = None,
    fast_ms: Sequence[float | None] | None = None,
    slow_ms: Sequence[float | None] | None = None,
    n_random: int = 2000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Confidence-gated escalation: keep the fast answer if ``p >= theta``, else ask the slow arm.

    Controls at the same escalation count ``m``: random escalation (exact expectation plus the
    2.5-97.5 % range over ``n_random`` seeded draws) and oracle escalation (escalates the fast
    arm's errors that the slow arm gets right first).
    """
    n = len(y)
    fc = [a == b for a, b in zip(y, fast, strict=True)]
    sc = [a == b for a, b in zip(y, slow, strict=True)]
    rng = random.Random(seed)
    rows = []
    for th in thresholds:
        esc = [p is None or p < th for p in fast_p]
        m = sum(esc)
        kept = [c for c, e in zip(fc, esc, strict=True) if not e]
        pred = [s if e else f for f, s, e in zip(fast, slow, esc, strict=True)]
        rand_exp = sum((1 - m / n) * a + (m / n) * b for a, b in zip(fc, sc, strict=True)) / n
        draws = []
        for _ in range(n_random):
            pick = set(rng.sample(range(n), m))
            draws.append(sum(sc[i] if i in pick else fc[i] for i in range(n)) / n)
        draws.sort()
        # Oracle: escalating an item gains (sc - fc); take the m best gains.
        gains = sorted((b - a for a, b in zip(fc, sc, strict=True)), reverse=True)
        oracle = (sum(fc) + sum(gains[:m])) / n
        row: dict[str, Any] = {
            "threshold": th,
            "coverage": 1 - m / n,
            "escalated": m,
            "fast_acc_on_kept": sum(kept) / len(kept) if kept else None,
            "cascade_acc": accuracy(y, pred),
            "cascade_macro_f1": macro_f1(y, pred, classes),
            "random_escalation_acc": rand_exp,
            "random_escalation_lo": draws[int(0.025 * n_random)],
            "random_escalation_hi": draws[min(n_random - 1, int(0.975 * n_random))],
            "oracle_escalation_acc": oracle,
        }
        if fast_cost is not None and slow_cost is not None:
            cost = [
                f + (s if e else 0.0) for f, s, e in zip(fast_cost, slow_cost, esc, strict=True)
            ]
            row["usd_per_1k"] = 1000 * sum(cost) / n
        if fast_ms is not None and slow_ms is not None:
            lat = [
                (f or 0.0) + ((s or 0.0) if e else 0.0)
                for f, s, e in zip(fast_ms, slow_ms, esc, strict=True)
            ]
            row["p50_ms"] = percentile(lat, 50)
            row["p95_ms"] = percentile(lat, 95)
        rows.append(row)
    return rows


def gate_cv(
    y: Sequence[str],
    fast: Sequence[str | None],
    fast_p: Sequence[float | None],
    slow: Sequence[str | None],
    thresholds: Sequence[float],
    k: int = 5,
    n_splits: int = 200,
    seed: int = 0,
    fast_cost: Sequence[float] | None = None,
    slow_cost: Sequence[float] | None = None,
) -> dict[str, Any] | None:
    """Held-out accuracy of the confidence gate: theta is never chosen on the items it scores.

    For each of ``n_splits`` seeded random partitions into ``k`` folds, every fold is answered
    with the threshold that maximises cascade accuracy on the other ``k - 1`` folds (ties: first
    in grid order). Reports the mean cascade accuracy over partitions with its 2.5-97.5 % range
    and min/max, the mean coverage of the fast arm, and how often each threshold was chosen. The
    in-sample best threshold of :func:`gate_curve` is reported alongside for contrast.
    """
    n = len(y)
    k = min(k, n)
    if k < 2:
        return None
    fc = [a == b for a, b in zip(y, fast, strict=True)]
    sc = [a == b for a, b in zip(y, slow, strict=True)]
    esc = {th: [p is None or p < th for p in fast_p] for th in thresholds}
    right = {
        th: [s if e else f for f, s, e in zip(fc, sc, esc[th], strict=True)] for th in thresholds
    }
    rng = random.Random(seed)
    accs: list[float] = []
    covs: list[float] = []
    costs: list[float] = []
    chosen: Counter[float] = Counter()
    for _ in range(n_splits):
        idx = list(range(n))
        rng.shuffle(idx)
        folds = [idx[i::k] for i in range(k)]
        hit = kept = 0
        usd = 0.0
        for i, fold in enumerate(folds):
            train = [j for m, f in enumerate(folds) if m != i for j in f]
            train_hits = {th: sum(right[th][j] for j in train) for th in thresholds}
            best = max(train_hits.values())
            th = next(t for t in thresholds if train_hits[t] == best)
            chosen[th] += 1
            hit += sum(right[th][j] for j in fold)
            kept += sum(not esc[th][j] for j in fold)
            if fast_cost is not None and slow_cost is not None:
                usd += sum(fast_cost[j] + (slow_cost[j] if esc[th][j] else 0.0) for j in fold)
        accs.append(hit / n)
        covs.append(kept / n)
        costs.append(1000 * usd / n)
    in_sample = {th: sum(right[th]) / n for th in thresholds}
    best_in = max(in_sample.values())
    out: dict[str, Any] = {
        "k": k,
        "n_splits": n_splits,
        "seed": seed,
        "tie_break": "first threshold in grid order",
        "cascade_acc_mean": sum(accs) / len(accs),
        "cascade_acc_lo": percentile(accs, 2.5),
        "cascade_acc_hi": percentile(accs, 97.5),
        "cascade_acc_min": min(accs),
        "cascade_acc_max": max(accs),
        "coverage_mean": sum(covs) / len(covs),
        "fast_acc": sum(fc) / n,
        "slow_acc": sum(sc) / n,
        "cascade_minus_slow_mean": sum(accs) / len(accs) - sum(sc) / n,
        "cascade_minus_fast_mean": sum(accs) / len(accs) - sum(fc) / n,
        "theta_chosen": {f"{th:g}": chosen[th] for th in thresholds if chosen[th]},
        "in_sample_best_theta": next(t for t in thresholds if in_sample[t] == best_in),
        "in_sample_best_acc": best_in,
    }
    if fast_cost is not None and slow_cost is not None:
        out["usd_per_1k_mean"] = sum(costs) / len(costs)
    return out


def multilabel_scores(
    gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]], labels: Sequence[str]
) -> dict[str, Any]:
    """Micro P/R/F1, macro-F1 over ``labels``, exact-set match and per-label F1."""
    tp = fp = fn = 0
    per: dict[str, list[int]] = {lab: [0, 0, 0] for lab in labels}
    exact = 0
    for g, p in zip(gold, pred, strict=True):
        gs, ps = set(g), set(p)
        exact += gs == ps
        for lab in labels:
            if lab in gs and lab in ps:
                per[lab][0] += 1
            elif lab in ps:
                per[lab][1] += 1
            elif lab in gs:
                per[lab][2] += 1
    for a, b, c in per.values():
        tp, fp, fn = tp + a, fp + b, fn + c
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = {lab: (2 * a / (2 * a + b + c) if a + b + c else 0.0) for lab, (a, b, c) in per.items()}
    return {
        "n": len(gold),
        "micro_p": prec,
        "micro_r": rec,
        "micro_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "macro_f1": sum(f1.values()) / len(f1) if f1 else float("nan"),
        "exact_match": exact / len(gold) if gold else float("nan"),
        "per_label_f1": f1,
    }


def negation_scores(
    gold: Sequence[Sequence[str]],
    flipped: Sequence[Sequence[str]],
    waived: Sequence[Sequence[str]],
    pred: Sequence[Sequence[str]],
) -> dict[str, Any]:
    """Negation subset (NevIR-style).

    ``flip_recall``: share of families stated as prohibitions that were predicted (should be 1);
    ``waiver_fp_rate``: share of waived families that were predicted anyway (should be 0);
    exact-set match on statements with / without any negation.
    """
    n_flip = hit_flip = n_w = hit_w = 0
    ex_neg, n_neg, ex_pos, n_pos = 0, 0, 0, 0
    for g, f, w, p in zip(gold, flipped, waived, pred, strict=True):
        ps = set(p)
        n_flip += len(f)
        hit_flip += sum(1 for x in f if x in ps)
        n_w += len(w)
        hit_w += sum(1 for x in w if x in ps)
        if f or w:
            n_neg += 1
            ex_neg += set(g) == ps
        else:
            n_pos += 1
            ex_pos += set(g) == ps
    return {
        "n_flipped": n_flip,
        "flip_recall": hit_flip / n_flip if n_flip else None,
        "n_waived": n_w,
        "waiver_fp_rate": hit_w / n_w if n_w else None,
        "n_negated_statements": n_neg,
        "exact_match_negated": ex_neg / n_neg if n_neg else None,
        "exact_match_plain": ex_pos / n_pos if n_pos else None,
    }


def threshold_predict(
    scores: Mapping[str, float], threshold: float, min_k: int = 0
) -> tuple[str, ...]:
    """Labels with score >= threshold, topped up to ``min_k`` by rank; label order of ``scores``."""
    labs = list(scores)
    keep = {lab for lab in labs if scores[lab] >= threshold}
    if len(keep) < min_k:
        keep |= set(topk_predict(scores, min_k))
    return tuple(lab for lab in labs if lab in keep)


def topk_predict(scores: Mapping[str, float], k: int) -> tuple[str, ...]:
    """The ``k`` highest-scoring labels (ties broken by label order); label order of ``scores``."""
    labs = list(scores)
    ranked = sorted(range(len(labs)), key=lambda i: (-scores[labs[i]], i))[:k]
    keep = {labs[i] for i in ranked}
    return tuple(lab for lab in labs if lab in keep)


def tune_threshold(
    scores: Sequence[Mapping[str, float]],
    gold: Sequence[Sequence[str]],
    labels: Sequence[str],
    grid: Sequence[float],
    min_k: int = 0,
) -> tuple[float, float]:
    """Grid threshold with the best micro-F1 (first best in grid order) and that F1."""
    best_t, best_f = grid[0], -1.0
    for t in grid:
        pred = [threshold_predict(s, t, min_k) for s in scores]
        f = multilabel_scores(gold, pred, labels)["micro_f1"]
        if f > best_f + 1e-12:
            best_t, best_f = t, f
    return best_t, best_f


def tune_per_label_thresholds(
    scores: Sequence[Mapping[str, float]],
    gold: Sequence[Sequence[str]],
    labels: Sequence[str],
    grid: Sequence[float],
    fallback: float = 0.5,
) -> dict[str, float]:
    """Per-label threshold maximising that label's F1 (first best in grid order).

    Labels without a positive in ``gold`` keep ``fallback``.
    """
    out = {}
    for lab in labels:
        pos = [lab in g for g in gold]
        if not any(pos):
            out[lab] = fallback
            continue
        best_t, best_f = fallback, -1.0
        for t in grid:
            tp = sum(1 for s, p in zip(scores, pos, strict=True) if p and s[lab] >= t)
            fp = sum(1 for s, p in zip(scores, pos, strict=True) if not p and s[lab] >= t)
            fn = sum(pos) - tp
            f = 2 * tp / (2 * tp + fp + fn)
            if f > best_f + 1e-12:
                best_t, best_f = t, f
        out[lab] = best_t
    return out


def per_label_predict(
    scores: Mapping[str, float], thresholds: Mapping[str, float]
) -> tuple[str, ...]:
    """Labels whose score reaches their own threshold; label order of ``scores``."""
    return tuple(lab for lab in scores if scores[lab] >= thresholds.get(lab, 0.5))


# ---------------------------------------------------------------------------------------------
# TF-IDF baseline (pure Python, no scikit-learn)
# ---------------------------------------------------------------------------------------------

_STOP = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "them",
        "there",
        "these",
        "this",
        "to",
        "was",
        "were",
        "which",
        "with",
        "each",
        "every",
        "one",
        "any",
        "all",
        "may",
        "must",
        "should",
        "not",
        "no",
        "do",
        "does",
        "need",
    }
)


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens, stop words removed, crude suffix stripping."""
    out = []
    for t in re.findall(r"[a-z]+", text.lower()):
        if t in _STOP or len(t) < 2:
            continue
        for suf in ("ing", "ed", "es", "s"):
            if len(t) > len(suf) + 3 and t.endswith(suf):
                t = t[: -len(suf)]
                break
        out.append(t)
    return out


def _tfidf_vec(tokens: list[str], idf: Mapping[str, float]) -> dict[str, float]:
    tf = Counter(tokens)
    v = {t: (1 + math.log(c)) * idf[t] for t, c in tf.items() if t in idf}
    norm = math.sqrt(sum(x * x for x in v.values()))
    return {t: x / norm for t, x in v.items()} if norm else {}


def tfidf_similarities(queries: Sequence[str], docs: Sequence[str]) -> list[list[float]]:
    """Cosine similarity of each query to each doc, IDF fitted on ``docs`` (smoothed)."""
    dtoks = [tokenize(d) for d in docs]
    df = Counter(t for toks in dtoks for t in set(toks))
    n = len(docs)
    idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
    dvecs = [_tfidf_vec(toks, idf) for toks in dtoks]
    out = []
    for q in queries:
        qv = _tfidf_vec(tokenize(q), idf)
        out.append([sum(w * dv.get(t, 0.0) for t, w in qv.items()) for dv in dvecs])
    return out


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb) if na and nb else 0.0


# ---------------------------------------------------------------------------------------------
# Live arms (network; lazily imported clients; cached; budget-guarded)
# ---------------------------------------------------------------------------------------------


class BudgetExceeded(RuntimeError):
    """A provider's live-spend cap for this run is used up."""


class CacheMiss(RuntimeError):
    """Offline mode and the request is not in the cache."""


class EmbeddingUnavailable(RuntimeError):
    """The embedding endpoint could not be reached or kept failing."""


class Budget:
    """Per-provider cap on *live* spend in this process (cache replays are free)."""

    def __init__(self, caps: Mapping[str, float]) -> None:
        self.caps = dict(caps)
        self.spent = dict.fromkeys(self.caps, 0.0)
        self._lock = threading.Lock()

    def check(self, provider: str) -> None:
        with self._lock:
            if self.spent.get(provider, 0.0) >= self.caps.get(provider, math.inf):
                raise BudgetExceeded(
                    f"{provider} live budget {self.caps[provider]:.2f} USD used up"
                )

    def charge(self, provider: str, usd: float) -> None:
        with self._lock:
            self.spent[provider] = self.spent.get(provider, 0.0) + usd


@dataclass(frozen=True)
class Answer:
    """One arm's answer for one item.

    ``scores``: label -> probability (Jev) or similarity (embeddings); ``choice``: single-label
    answer; ``labels``: multi-label answer of arms that emit a set (DeepSeek, keywords).
    """

    item_id: str
    scores: dict[str, float] | None = None
    choice: str | None = None
    labels: tuple[str, ...] = ()
    latency_ms: float | None = None
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    error: str | None = None
    live: bool = False

    @property
    def p_top(self) -> float | None:
        if not self.scores or self.choice is None:
            return None
        return self.scores.get(self.choice)


def _retry(fn: Callable[[], Any], tries: int = 3, base_s: float = 2.0) -> Any:
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(base_s * 2**i)
    return None  # pragma: no cover


class JevRouter:
    """Jev (``jev-1.13.0``) routing calls with the repo's cache and cost conventions."""

    def __init__(
        self,
        cache: ResponseCache,
        budget: Budget | None = None,
        model: str = JEV_MODEL,
        timeout_s: float = 30.0,
        offline: bool = False,
    ) -> None:
        self.cache, self.budget, self.model = cache, budget, model
        self.timeout_s, self.offline = timeout_s, offline
        self._client: Any = None
        self._lock = threading.Lock()

    def _c(self) -> Any:
        with self._lock:
            if self._client is None:
                load_secrets()
                from typesafe_sdk import TypeSafeClient

                self._client = TypeSafeClient(model=self.model, timeout=self.timeout_s)
        return self._client

    def _call(
        self, engine_id: str, state: Any, questions: dict[str, Any]
    ) -> tuple[dict[str, Any], bool]:
        key = request_key(engine_id, self.model, {"state": state, "q": questions})
        hit = self.cache.get(key)
        if hit is not None:
            return hit, False
        if self.offline:
            raise CacheMiss(engine_id)
        if self.budget:
            self.budget.check("jev")
        client = self._c()
        t0 = time.perf_counter()
        r = _retry(lambda: client.system_one(state=state, questions=questions))
        latency = 1000 * (time.perf_counter() - t0)
        answers: dict[str, Any] = {}
        for name, a in r.answers.items():
            if a.type == "noul":
                answers[name] = {"type": "noul", "noul": float(a.noul)}
            elif a.type == "choice":
                answers[name] = {
                    "type": "choice",
                    "choice": a.choice,
                    "probabilities": {k: float(v) for k, v in a.probabilities.items()},
                    "confidence": float(a.confidence),
                }
        try:
            server_ms = r.raw_http_response.headers.get("x-envoy-upstream-service-time")
        except Exception:
            server_ms = None
        rec = {
            "answers": answers,
            "model": r.model,
            "input_tokens": int(r.usage.input_tokens or 0),
            "output_tokens": int(r.usage.output_tokens or 0),
            "latency_ms": round(latency, 1),
            "server_ms": server_ms,
            "utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        self.cache.put(key, rec)
        if self.budget:
            self.budget.charge("jev", rec["input_tokens"] * JEV_PRICE_PER_INPUT_TOKEN)
        return rec, True

    def _answer(self, item_id: str, engine_id: str, state: Any, qs: dict[str, Any]) -> Answer:
        try:
            rec, live = self._call(engine_id, state, qs)
        except CacheMiss:
            raise  # an offline replay must stop, not fail open
        except Exception as exc:
            return Answer(item_id, error=f"jev: {type(exc).__name__}: {exc}"[:300])
        a = rec["answers"]
        if len(a) == 1 and next(iter(a.values()))["type"] == "choice":
            ch = next(iter(a.values()))
            scores, choice = ch["probabilities"], ch["choice"]
        else:
            scores = {k: v["noul"] for k, v in a.items() if v["type"] == "noul"}
            scores = {k: scores[k] for k in qs if k in scores}
            choice = None
        return Answer(
            item_id,
            scores=scores,
            choice=choice,
            latency_ms=rec["latency_ms"],
            cost_usd=rec["input_tokens"] * JEV_PRICE_PER_INPUT_TOKEN,
            input_tokens=rec["input_tokens"],
            output_tokens=rec["output_tokens"],
            model=rec["model"],
            live=live,
        )

    def route_catalog(self, item: CatalogItem, descriptions: Mapping[str, str]) -> Answer:
        q = {"family": jev_catalog_question(item.kind, descriptions)}
        return self._answer(item.item_id, "h2a-jev-choice", jev_catalog_state(item), q)

    def route_requirement(
        self, item_id: str, text: str, descriptions: Mapping[str, str], style: str = "modelling"
    ) -> Answer:
        qs = jev_family_questions(descriptions, style)
        return self._answer(item_id, "h2b-jev-nouls", {"statement": text}, qs)


class DeepSeekRouter:
    """DeepSeek (non-thinking, JSON mode, temperature 0) as the LLM router."""

    def __init__(
        self,
        cache: ResponseCache,
        budget: Budget | None = None,
        model: str = DS_MODEL,
        offline: bool = False,
        chat: DeepSeekChat | None = None,
    ) -> None:
        self.cache, self.budget, self.model, self.offline = cache, budget, model, offline
        self.chat = chat or DeepSeekChat(timeout_s=60.0)

    def _call(self, engine_id: str, msgs: list[dict[str, str]]) -> tuple[dict[str, Any], bool]:
        key = request_key(engine_id, self.model, msgs)
        hit = self.cache.get(key)
        if hit is not None:
            return hit, False
        if self.offline:
            raise CacheMiss(engine_id)
        if self.budget:
            self.budget.check("deepseek")
        r = self.chat.chat(self.model, msgs, thinking=False, json_mode=True, max_tokens=300)
        rec = {
            # The answer is a short JSON label object; truncation guards against the model
            # echoing input text into the (committed) cache.
            "content": r.content[:400],
            "usage": r.usage,
            "latency_ms": r.latency_ms,
            "model": r.model,
            "utc": r.utc,
        }
        self.cache.put(key, rec)
        if self.budget:
            self.budget.charge("deepseek", self._cost(rec))
        return rec, True

    def _cost(self, rec: Mapping[str, Any]) -> float:
        return deepseek_cost(self.model, rec["usage"], datetime.fromisoformat(rec["utc"]))

    def _base(self, item_id: str, rec: Mapping[str, Any], live: bool, **kw: Any) -> Answer:
        u = rec["usage"]
        return Answer(
            item_id,
            latency_ms=rec["latency_ms"],
            cost_usd=self._cost(rec),
            input_tokens=u.get("prompt_tokens", 0),
            output_tokens=u.get("completion_tokens", 0),
            model=rec["model"],
            live=live,
            **kw,
        )

    def route_catalog(self, item: CatalogItem, descriptions: Mapping[str, str]) -> Answer:
        msgs = catalog_messages(item.kind, descriptions, item.text)
        try:
            rec, live = self._call("h2a-deepseek-json", msgs)
        except CacheMiss:
            raise  # an offline replay must stop, not fail open
        except Exception as exc:
            return Answer(item.item_id, error=f"deepseek: {type(exc).__name__}: {exc}"[:300])
        choice = parse_single_label(rec["content"], list(descriptions))
        err = None if choice else "unparseable or unknown class"
        return self._base(item.item_id, rec, live, choice=choice, error=err)

    def route_requirement(
        self, item_id: str, text: str, descriptions: Mapping[str, str], style: str = "modelling"
    ) -> Answer:
        msgs = requirement_messages(descriptions, text, style)
        try:
            rec, live = self._call("h2b-deepseek-json", msgs)
        except CacheMiss:
            raise  # an offline replay must stop, not fail open
        except Exception as exc:
            return Answer(item_id, error=f"deepseek: {type(exc).__name__}: {exc}"[:300])
        labels, unknown = parse_multi_label(rec["content"], list(descriptions))
        err = f"{unknown} unknown families" if unknown else None
        return self._base(item_id, rec, live, labels=labels, error=err)


class ScadsEmbedder:
    """Query-vs-document cosine similarities from the ScaDS.AI OpenAI-compatible endpoint.

    Only similarity rows are cached (key = query text + the full document list), never vectors,
    so the cache stays small and holds no text. Batches are paced and retried with back-off
    because the endpoint is tightly rate-limited. Latency per query is its batch time divided by
    the batch size; embedding the documents (the "index") is excluded.
    """

    def __init__(
        self,
        cache: ResponseCache,
        model: str = EMB_MODEL,
        base_url: str = SCADS_BASE_URL,
        batch_size: int = 16,
        pause_s: float = 1.5,
        max_retries: int = 6,
        offline: bool = False,
    ) -> None:
        self.cache, self.model, self.base_url = cache, model, base_url
        self.batch_size, self.pause_s, self.max_retries = batch_size, pause_s, max_retries
        self.offline = offline
        self._client: Any = None

    def _c(self) -> Any:
        if self._client is None:
            load_secrets()
            key = os.environ.get("SCADS_API_KEY")
            if not key:
                raise EmbeddingUnavailable("SCADS_API_KEY not set")
            from openai import OpenAI

            self._client = OpenAI(api_key=key, base_url=self.base_url, timeout=120, max_retries=0)
        return self._client

    def _embed(self, texts: Sequence[str]) -> tuple[list[list[float]], list[float]]:
        vecs: list[list[float]] = []
        per_ms: list[float] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            for attempt in range(self.max_retries):
                try:
                    t0 = time.perf_counter()
                    resp = self._c().embeddings.create(model=self.model, input=batch)
                    ms = 1000 * (time.perf_counter() - t0)
                    break
                except EmbeddingUnavailable:
                    raise
                except Exception as exc:
                    if attempt == self.max_retries - 1:
                        raise EmbeddingUnavailable(f"{type(exc).__name__}: {exc}"[:200]) from exc
                    time.sleep(min(60.0, 5.0 * 2**attempt))
            data = sorted(resp.data, key=lambda d: d.index)
            vecs += [list(d.embedding) for d in data]
            per_ms += [ms / len(batch)] * len(batch)
            time.sleep(self.pause_s)
        return vecs, per_ms

    def similarities(
        self, queries: Sequence[str], docs: Sequence[str]
    ) -> list[tuple[list[float], float | None]]:
        """For each query: (similarity to every doc, per-query latency in ms)."""
        keys = [
            request_key("scads-emb-sims", self.model, {"q": q, "docs": list(docs)}) for q in queries
        ]
        missing = [i for i, k in enumerate(keys) if self.cache.get(k) is None]
        if missing:
            if self.offline:
                raise CacheMiss("scads-emb-sims")
            uniq = list(dict.fromkeys([*docs, *(queries[i] for i in missing)]))
            vecs, ms = self._embed(uniq)
            index = {t: (v, m) for t, v, m in zip(uniq, vecs, ms, strict=True)}
            dvecs = [index[d][0] for d in docs]
            now = datetime.now(UTC).isoformat(timespec="seconds")
            for i in missing:
                qv, qms = index[queries[i]]
                sims = [round(cosine(qv, dv), 6) for dv in dvecs]
                self.cache.put(
                    keys[i],
                    {"sims": sims, "latency_ms": round(qms, 1), "model": self.model, "utc": now},
                )
        out = []
        for k in keys:
            rec = self.cache.get(k)
            assert rec is not None
            out.append((rec["sims"], rec.get("latency_ms")))
        return out
