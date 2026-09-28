"""Relevance filtering of noisy problem statements in front of an MILP generator.

Why: in a retrieval-augmented MILP assistant, the generator should see the problem data and not
the noise around it. Jev can *drive* that step without generating anything: one Noul per sentence
("is this sentence needed to formulate the optimisation model?"), all asked in one request over
the whole statement, and code keeps the sentences whose probability clears a threshold chosen on
held-out data. A filter is only safe if its recall on data-bearing sentences is (almost) perfect,
because a dropped release time silently yields a different model; precision only saves prompt
tokens. This module holds the arms, the metrics and the downstream generate-and-solve step used by
``experiments/h2/run_filtering.py`` (docs/hypotheses.md, P2-H2 filter/"driver of RAG" mode).

Arms (every paid call is cached with :class:`~d4r.engines.cache.ResponseCache`):

- ``jev``: TypeSafe Jev (pinned ``jev-1.13.0``), one Noul per sentence with explicit yes/no
  criteria, all in one request, the statement's sentences as state;
- ``jev-bare``: the same without criteria (ablation: how much do the criteria carry?);
- ``deepseek``: DeepSeek-flash, non-thinking, JSON mode, lists the needed sentence ids (given the
  same definition of "needed" as Jev's criteria). Log-probabilities are not requested: a list of
  ids has no per-sentence probability to read, so this arm has no threshold;
- ``embed``: cosine similarity of each sentence to a task query with an embedding model on the
  ScaDS.AI endpoint (``Qwen/Qwen3-Embedding-4B``), threshold chosen like Jev's;
- ``regex``: keep every sentence that contains a digit (the numeracy-trap baseline);
- ``keep-all``: no filtering.

Downstream: DeepSeek-flash writes the instance as a :class:`~d4r.engines.milp.MilpSpec` (data,
never code, as in ADR-0005) from the full or the filtered statement; HiGHS solves it and the
objective is compared with the exact optimum from :func:`d4r.h2.statements.reference_optimum`.
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from d4r.apikeys import load_secrets
from d4r.engines.cache import ResponseCache, request_key
from d4r.engines.jev import JEV_PRICE_PER_INPUT_TOKEN
from d4r.engines.llm import DeepSeekChat, deepseek_cost, is_peak
from d4r.engines.milp import MilpSpec, solve_spec
from d4r.h2.statements import Statement

__all__ = [
    "MILP_SYSTEM",
    "MILP_SYSTEM_V1",
    "NEEDED_FALSE",
    "NEEDED_TRUE",
    "Budget",
    "BudgetExceeded",
    "DeepSeekSentenceFilter",
    "EmbeddingSentenceFilter",
    "FilterScores",
    "JevSentenceFilter",
    "ReplayCache",
    "ReplayMiss",
    "SentenceMetrics",
    "aggregate",
    "choose_floor_threshold",
    "generate_and_solve",
    "keep_all_scores",
    "kept_ids",
    "mcnemar_exact_p",
    "milp_messages",
    "objective_matches",
    "parse_id_list",
    "regex_scores",
    "sentence_metrics",
    "sweep",
]

#: The shared definition of "needed" (Jev's Noul criteria and DeepSeek's instructions).
NEEDED_TRUE = (
    "The sentence describes today's dispatching problem on this line in a way the optimisation"
    " model must reflect: the line and its sections, how trains may use them, the objective, or a"
    " train's release time, running times, due time or priority weight, a headway, an overtaking"
    " rule, or a fault or blockage that affects today's trains."
)
NEEDED_FALSE = (
    "Leaving the sentence out would not change the model: it is background, concerns another line"
    " or another day, or gives details such as vehicles, liveries, passenger numbers, tickets,"
    " staff, platforms, history or weather, even if it contains numbers."
)
NEEDED_QUESTION = (
    "Is sentence `sentences.{sid}` needed to formulate the optimisation model for the dispatching"
    " problem stated in `sentences`?"
)

#: Embedding query (Qwen3-Embedding instruction format).
EMBED_QUERY = (
    "Instruct: Given a railway dispatching problem statement, retrieve the sentences that state"
    " data or rules needed to formulate its optimisation model\n"
    "Query: today's trains on this single-track line: sections, release times, running times,"
    " due times, priority weights, headway, overtaking rules, faults and blockages, and the"
    " objective to minimise weighted tardiness"
)

_DIGIT = re.compile(r"\d")


# --------------------------------------------------------------------------- budget
class BudgetExceeded(RuntimeError):
    """A provider's spending cap for this run is reached; no further fresh calls are made."""


@dataclass
class Budget:
    """Spending caps per provider for *fresh* (non-cached) calls in this process."""

    caps: dict[str, float]
    spent: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def check(self, provider: str) -> None:
        with self._lock:
            if self.spent.get(provider, 0.0) >= self.caps.get(provider, math.inf):
                raise BudgetExceeded(f"{provider} budget {self.caps[provider]:.2f} USD reached")

    def charge(self, provider: str, usd: float) -> None:
        with self._lock:
            self.spent[provider] = self.spent.get(provider, 0.0) + usd


# --------------------------------------------------------------------------- replay
class ReplayMiss(RuntimeError):
    """An offline replay met a request that is not in the cache."""


class ReplayCache(ResponseCache):
    """A response cache that only replays: a missing key raises before any network call.

    Every arm and :func:`generate_and_solve` look a request up before they call an API, so with
    this cache an offline replay either reproduces the cached answers or stops at the first miss.
    """

    def get(self, key: str) -> dict[str, Any] | None:
        hit = super().get(key)
        if hit is None:
            raise ReplayMiss(f"offline replay: request {key[:12]} not in {self.path}")
        return hit

    def put(self, key: str, value: dict[str, Any]) -> None:
        raise ReplayMiss(f"offline replay tried to store a fresh answer in {self.path}")


# --------------------------------------------------------------------------- scores
@dataclass(frozen=True)
class FilterScores:
    """One arm's per-sentence scores for one statement (1/0 for arms without probabilities)."""

    arm: str
    stmt_id: str
    scores: dict[str, float]
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    error: str | None = None
    fresh: bool = False


def kept_ids(fs: FilterScores, threshold: float = 0.5) -> frozenset[str]:
    """Sentence ids with score >= threshold. A failed call keeps everything (fail open)."""
    if fs.error:
        return frozenset(fs.scores)
    return frozenset(sid for sid, v in fs.scores.items() if v >= threshold)


def regex_scores(stmt: Statement) -> FilterScores:
    """Keep every sentence that contains a digit."""
    t0 = time.perf_counter()
    scores = {s.sid: float(bool(_DIGIT.search(s.text))) for s in stmt.sentences}
    return FilterScores("regex", stmt.stmt_id, scores, 1000 * (time.perf_counter() - t0))


def keep_all_scores(stmt: Statement) -> FilterScores:
    return FilterScores("keep-all", stmt.stmt_id, {s.sid: 1.0 for s in stmt.sentences})


def sentence_block(stmt: Statement) -> dict[str, str]:
    """Sentence id -> text, in reading order (the state both LLM arms see)."""
    return {s.sid: s.text for s in stmt.sentences}


# --------------------------------------------------------------------------- Jev
class JevSentenceFilter:
    """One Noul per sentence, all in one Jev request with the whole statement as state."""

    def __init__(
        self,
        model: str = "jev-1.13.0",
        criteria: bool = True,
        cache: ResponseCache | None = None,
        budget: Budget | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self.model = model
        self.criteria = criteria
        self.cache = cache if cache is not None else ResponseCache(None)
        self.budget = budget
        self.timeout_s = timeout_s
        self.arm = "jev" if criteria else "jev-bare"
        self._client: Any = None

    def state(self, stmt: Statement) -> dict[str, Any]:
        return {"sentences": sentence_block(stmt)}

    def questions(self, stmt: Statement) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for s in stmt.sentences:
            q: dict[str, Any] = {"type": "noul", "instructions": NEEDED_QUESTION.format(sid=s.sid)}
            if self.criteria:
                q["criteria"] = {"true": NEEDED_TRUE, "false": NEEDED_FALSE}
            out[s.sid] = q
        return out

    def _client_or_init(self) -> Any:
        if self._client is None:
            load_secrets()
            from typesafe_sdk import TypeSafeClient

            self._client = TypeSafeClient(model=self.model, timeout=self.timeout_s)
        return self._client

    def score(self, stmt: Statement) -> FilterScores:
        state, questions = self.state(stmt), self.questions(stmt)
        key = request_key(f"h2-filter-{self.arm}", self.model, {"state": state, "q": questions})
        hit = self.cache.get(key)
        fresh = False
        if hit is None:
            if self.budget:
                self.budget.check("jev")
            client = self._client_or_init()
            t0 = time.perf_counter()
            try:
                r = client.system_one(state=state, questions=questions)
            except Exception as exc:
                return FilterScores(
                    self.arm,
                    stmt.stmt_id,
                    {s.sid: 1.0 for s in stmt.sentences},
                    error=f"jev: {type(exc).__name__}: {exc}"[:300],
                )
            latency = 1000 * (time.perf_counter() - t0)
            hit = {
                "nouls": {sid: float(a.noul) for sid, a in r.nouls.items()},
                "model": r.model,
                "input_tokens": int(r.usage.input_tokens or 0),
                "output_tokens": int(r.usage.output_tokens or 0),
                "latency_ms": round(latency, 1),
                "request_id": getattr(r, "request_id", None),
                "utc": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            self.cache.put(key, hit)
            fresh = True
        cost = hit["input_tokens"] * JEV_PRICE_PER_INPUT_TOKEN
        if fresh and self.budget:
            self.budget.charge("jev", cost)
        missing = [s.sid for s in stmt.sentences if s.sid not in hit["nouls"]]
        return FilterScores(
            arm=self.arm,
            stmt_id=stmt.stmt_id,
            scores={s.sid: hit["nouls"].get(s.sid, 1.0) for s in stmt.sentences},
            latency_ms=hit["latency_ms"],
            input_tokens=hit["input_tokens"],
            output_tokens=hit["output_tokens"],
            cost_usd=cost,
            model=hit["model"],
            error=f"missing answers {missing}" if missing else None,
            fresh=fresh,
        )


# --------------------------------------------------------------------------- DeepSeek
FILTER_SYSTEM = (
    "You filter noisy railway dispatching problem statements before an optimisation model is"
    " written from them. A sentence is NEEDED if: " + NEEDED_TRUE + " A sentence is NOT needed"
    " if: " + NEEDED_FALSE + " Answer with a JSON object only, of the form"
    ' {"needed": ["<sentence id>", ...]}, listing every needed sentence id.'
)


def parse_id_list(content: str, known: Sequence[str]) -> list[str] | None:
    """Needed ids from the model's JSON answer (tolerant of stray text); None if unparseable."""
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
    ids = obj.get("needed") if isinstance(obj, dict) else None
    if not isinstance(ids, list):
        return None
    known_set = set(known)
    return sorted({str(i).strip() for i in ids} & known_set)


class DeepSeekSentenceFilter:
    """DeepSeek-flash (non-thinking, JSON mode) lists the needed sentence ids."""

    arm = "deepseek"

    def __init__(
        self,
        model: str = "deepseek-flash",
        cache: ResponseCache | None = None,
        chat: DeepSeekChat | None = None,
        budget: Budget | None = None,
    ) -> None:
        self.model = model
        self.cache = cache if cache is not None else ResponseCache(None)
        self.chat = chat or DeepSeekChat()
        self.budget = budget

    def messages(self, stmt: Statement) -> list[dict[str, str]]:
        lines = "\n".join(f"{sid}: {text}" for sid, text in sentence_block(stmt).items())
        user = (
            "Statement, one sentence per line:\n"
            + lines
            + '\n\nReply with the JSON object {"needed": [...]} only.'
        )
        return [{"role": "system", "content": FILTER_SYSTEM}, {"role": "user", "content": user}]

    def score(self, stmt: Statement) -> FilterScores:
        msgs = self.messages(stmt)
        key = request_key("h2-filter-deepseek[fast]", self.model, msgs)
        hit = self.cache.get(key)
        fresh = False
        if hit is None:
            if self.budget:
                self.budget.check("deepseek")
            try:
                r = self.chat.chat(self.model, msgs, thinking=False, max_tokens=1500)
            except Exception as exc:
                return FilterScores(
                    self.arm,
                    stmt.stmt_id,
                    {s.sid: 1.0 for s in stmt.sentences},
                    error=f"deepseek: {type(exc).__name__}: {exc}"[:300],
                )
            hit = {
                "content": r.content,
                "usage": r.usage,
                "latency_ms": r.latency_ms,
                "model": r.model,
                "utc": r.utc,
            }
            self.cache.put(key, hit)
            fresh = True
        cost = deepseek_cost(self.model, hit["usage"], datetime.fromisoformat(hit["utc"]))
        if fresh and self.budget:
            self.budget.charge("deepseek", cost)
        ids = parse_id_list(hit["content"], stmt.ids())
        err = None if ids is not None else f"unparseable answer: {hit['content'][:120]!r}"
        chosen = set(ids or [])
        scores = {s.sid: 1.0 if (ids is None or s.sid in chosen) else 0.0 for s in stmt.sentences}
        return FilterScores(
            arm=self.arm,
            stmt_id=stmt.stmt_id,
            scores=scores,
            latency_ms=hit["latency_ms"],
            input_tokens=hit["usage"].get("prompt_tokens", 0),
            output_tokens=hit["usage"].get("completion_tokens", 0),
            cost_usd=cost,
            model=hit["model"],
            error=err,
            fresh=fresh,
        )


# --------------------------------------------------------------------------- embeddings
def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class EmbeddingSentenceFilter:
    """Cosine similarity of each sentence to a task query (ScaDS.AI embedding endpoint).

    The endpoint is rate-limited, so calls are spaced by ``min_interval_s`` and retried with
    exponential back-off. Vectors are not cached (large); the cosine scores are.
    """

    arm = "embed"

    def __init__(
        self,
        model: str = "Qwen/Qwen3-Embedding-4B",
        base_url: str = "https://llm.scads.ai/v1",
        cache: ResponseCache | None = None,
        min_interval_s: float = 2.0,
        max_retries: int = 6,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.cache = cache if cache is not None else ResponseCache(None)
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self._client: Any = None
        self._last = 0.0
        self._query_vec: list[float] | None = None

    def _c(self) -> Any:
        if self._client is None:
            load_secrets()
            from openai import OpenAI

            self._client = OpenAI(
                api_key=os.environ.get("SCADS_API_KEY"),
                base_url=self.base_url,
                timeout=120.0,
                max_retries=0,
            )
        return self._client

    def _embed(self, texts: list[str]) -> tuple[list[list[float]], float]:
        delay = self.min_interval_s
        for attempt in range(self.max_retries):
            wait = self.min_interval_s - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            t0 = time.perf_counter()
            try:
                r = self._c().embeddings.create(model=self.model, input=texts)
                self._last = time.monotonic()
                return [list(d.embedding) for d in r.data], 1000 * (time.perf_counter() - t0)
            except Exception:
                self._last = time.monotonic()
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable")

    def score(self, stmt: Statement) -> FilterScores:
        texts = [s.text for s in stmt.sentences]
        key = request_key("h2-filter-embed", self.model, {"query": EMBED_QUERY, "texts": texts})
        hit = self.cache.get(key)
        fresh = False
        if hit is None:
            try:
                if self._query_vec is None:
                    self._query_vec = self._embed([EMBED_QUERY])[0][0]
                vecs, latency = self._embed(texts)
            except Exception as exc:
                return FilterScores(
                    self.arm,
                    stmt.stmt_id,
                    {s.sid: 1.0 for s in stmt.sentences},
                    error=f"embed: {type(exc).__name__}: {exc}"[:300],
                )
            hit = {
                "cosine": {
                    s.sid: round(_cosine(self._query_vec, v), 6)
                    for s, v in zip(stmt.sentences, vecs, strict=True)
                },
                "latency_ms": round(latency, 1),
                "model": self.model,
                "utc": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            self.cache.put(key, hit)
            fresh = True
        return FilterScores(
            arm=self.arm,
            stmt_id=stmt.stmt_id,
            scores={s.sid: float(hit["cosine"][s.sid]) for s in stmt.sentences},
            latency_ms=hit["latency_ms"],
            model=hit["model"],
            fresh=fresh,
        )


# --------------------------------------------------------------------------- metrics
@dataclass(frozen=True)
class SentenceMetrics:
    """Confusion counts of one filtered statement against the gold labels."""

    stmt_id: str
    n: int
    kept: int
    tp: int
    fp: int
    fn: int
    tn: int
    data_total: int
    data_kept: int
    frame_total: int
    frame_kept: int
    trap_total: int
    trap_kept: int
    near_total: int
    near_kept: int
    superseded_total: int
    superseded_kept: int
    chars_total: int
    chars_kept: int


def sentence_metrics(stmt: Statement, kept: Iterable[str]) -> SentenceMetrics:
    k = set(kept)
    tp = fp = fn = tn = 0
    data_t = data_k = frame_t = frame_k = trap_t = trap_k = near_t = near_k = sup_t = sup_k = 0
    chars_t = chars_k = 0
    for s in stmt.sentences:
        keep = s.sid in k
        chars_t += len(s.text)
        chars_k += len(s.text) if keep else 0
        if s.needed:
            tp += keep
            fn += not keep
        else:
            fp += keep
            tn += not keep
        if s.data_bearing:
            data_t += 1
            data_k += keep
        if s.kind == "frame":
            frame_t += 1
            frame_k += keep
        if not s.needed and s.has_number:
            trap_t += 1
            trap_k += keep
        if s.kind == "near_miss":
            near_t += 1
            near_k += keep
        if s.kind == "superseded":
            sup_t += 1
            sup_k += keep
    return SentenceMetrics(
        stmt.stmt_id,
        len(stmt.sentences),
        len(k & set(stmt.ids())),
        tp,
        fp,
        fn,
        tn,
        data_t,
        data_k,
        frame_t,
        frame_k,
        trap_t,
        trap_k,
        near_t,
        near_k,
        sup_t,
        sup_k,
        chars_t,
        chars_k,
    )


def _ratio(a: float, b: float) -> float | None:
    return round(a / b, 4) if b else None


def aggregate(ms: Sequence[SentenceMetrics]) -> dict[str, Any]:
    """Pooled sentence-level rates plus statement-level floors."""
    fields = [f for f in SentenceMetrics.__dataclass_fields__ if f != "stmt_id"]
    tot = {f: sum(getattr(m, f) for m in ms) for f in fields}
    return {
        "statements": len(ms),
        "sentences": tot["n"],
        "recall_needed": _ratio(tot["tp"], tot["tp"] + tot["fn"]),
        "recall_data": _ratio(tot["data_kept"], tot["data_total"]),
        "recall_frame": _ratio(tot["frame_kept"], tot["frame_total"]),
        "precision": _ratio(tot["tp"], tot["tp"] + tot["fp"]),
        "noise_removed": _ratio(tot["tn"], tot["tn"] + tot["fp"]),
        "trap_kept": _ratio(tot["trap_kept"], tot["trap_total"]),
        "near_miss_kept": _ratio(tot["near_kept"], tot["near_total"]),
        "superseded_kept": _ratio(tot["superseded_kept"], tot["superseded_total"]),
        "char_reduction": _ratio(tot["chars_total"] - tot["chars_kept"], tot["chars_total"]),
        "stmt_all_data_kept": _ratio(sum(m.data_kept == m.data_total for m in ms), len(ms)),
        "stmt_all_needed_kept": _ratio(sum(m.fn == 0 for m in ms), len(ms)),
        "dropped_needed": tot["fn"],
        "kept_noise": tot["fp"],
    }


def cost_latency(fs: Sequence[FilterScores]) -> dict[str, Any]:
    lat = [f.latency_ms for f in fs]
    return {
        "calls": len(fs),
        "errors": sum(f.error is not None for f in fs),
        "latency_ms_mean": round(statistics.fmean(lat), 1) if lat else None,
        "latency_ms_p50": round(statistics.median(lat), 1) if lat else None,
        "latency_ms_max": round(max(lat), 1) if lat else None,
        "input_tokens": sum(f.input_tokens for f in fs),
        "output_tokens": sum(f.output_tokens for f in fs),
        "cost_usd": round(sum(f.cost_usd for f in fs), 6),
        "cost_usd_per_statement": round(sum(f.cost_usd for f in fs) / len(fs), 7) if fs else None,
    }


def sweep(
    stmts: Sequence[Statement], scores: dict[str, FilterScores], thresholds: Sequence[float]
) -> list[dict[str, Any]]:
    """Aggregate metrics for each threshold (keep score >= threshold)."""
    rows = []
    for th in thresholds:
        ms = [sentence_metrics(s, kept_ids(scores[s.stmt_id], th)) for s in stmts]
        rows.append({"threshold": th, **aggregate(ms)})
    return rows


def choose_floor_threshold(rows: Sequence[dict[str, Any]]) -> float:
    """Largest threshold that keeps every needed sentence; else the one with the best recall."""
    perfect = [r["threshold"] for r in rows if (r["recall_needed"] or 0.0) >= 1.0]
    if perfect:
        return max(perfect)
    best = max(r["recall_needed"] or 0.0 for r in rows)
    return max(r["threshold"] for r in rows if (r["recall_needed"] or 0.0) == best)


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


# --------------------------------------------------------------------------- downstream
_MILP_JSON_FORMAT = (
    " Return a JSON object only, with this structure:"
    ' {"variables": [{"name": str, "type": "binary"|"integer"|"continuous", "lb": number|null,'
    ' "ub": number|null}], "constraints": [{"name": str, "terms": {"<var>": coefficient},'
    ' "sense": "<="|">="|"==", "rhs": number}], "objective": {"sense": "min"|"max",'
    ' "terms": {"<var>": coefficient}, "constant": number}}. Every constraint is linear: all'
    " variables on the left with their coefficients, one constant on the right; a variable appears"
    " at most once per constraint."
)

#: v1: the modelling approach in words only (the pilot showed frequent bookkeeping errors).
MILP_SYSTEM_V1 = (
    "You are an operations-research engineer supporting a railway traffic dispatcher. Read the"
    " dispatching problem statement and write a mixed-integer linear program (MILP) whose optimum"
    " solves it exactly. Model it with: a continuous start-time variable (minutes) for every train"
    " and section; a train enters its next section only after leaving the previous one (it may"
    " wait in between); for every pair of trains and every section a binary order variable with"
    " big-M constraints, so that sections are occupied by one train at a time with the stated"
    " headway between the exit of one train and the entry of the next; a continuous tardiness"
    " variable per train that is at least 0 and at least its exit time from the last section minus"
    " its due time; and the objective to minimise the sum of priority weight times tardiness. Pick"
    " big-M large enough (for example the largest due time plus the sum of all running times). Use"
    " only numbers stated in the problem and ignore information that does not belong to it."
    + _MILP_JSON_FORMAT
)

#: v2 (used): the same model as a fixed template with variable names, so that the generator's
#: job is to read the statement and fill the template; noise then acts on reading, not on algebra.
MILP_SYSTEM = (
    "You are an operations-research engineer supporting a railway traffic dispatcher. Read the"
    " dispatching problem statement and write a mixed-integer linear program (MILP) whose optimum"
    " solves it exactly. Use only numbers stated for today's problem on this line and ignore"
    " information that does not belong to it. Give each train a short key (T1, T2, ... in order of"
    " first mention) and number the sections 1..K in running order. Use this template:"
    " (1) continuous x_<T>_<k> >= 0: time the train enters section k;"
    " (2) x_<T>_1 >= its release time (earliest entry into the first section), and"
    " x_<T>_<k> >= the end of any blockage of section k;"
    " (3) x_<T>_<k+1> - x_<T>_<k> >= p_<T,k>, the running time of the train on section k including"
    " any extra minutes from faults (trains may wait at stations between sections);"
    " (4) continuous e_<T>: e_<T> - x_<T>_<K> == p_<T,K> (exit from the last section);"
    " (5) for every pair of trains A before B in key order and every section k a binary y_<A>_<B>_<k>"
    " (1 if A enters section k before B) with x_<B>_<k> - x_<A>_<k> - M*y_<A>_<B>_<k> >= p_<A,k> + h"
    " - M and x_<A>_<k> - x_<B>_<k> + M*y_<A>_<B>_<k> >= p_<B,k> + h, where h is the headway (0 if"
    " none is stated) and M a large constant such as 1000; where a rule fixes the order of A and B,"
    " fix y_<A>_<B>_<k> with lb = ub;"
    " (6) continuous tard_<T> >= 0 with tard_<T> - e_<T> >= -d_<T>, d_<T> the due time;"
    " (7) objective: minimise the sum of w_<T> * tard_<T>, w_<T> the priority weight."
    + _MILP_JSON_FORMAT
)


def milp_messages(text: str, system: str = MILP_SYSTEM) -> list[dict[str, str]]:
    user = "Problem statement:\n" + text + "\n\nReturn the MILP as the JSON object described."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def objective_matches(obj: float, ref: float, rel: float = 1e-4, abs_tol: float = 0.01) -> bool:
    return math.isfinite(obj) and abs(obj - ref) <= max(abs_tol, rel * abs(ref))


@dataclass(frozen=True)
class GenerationResult:
    """Outcome of generate -> validate -> solve (with error feedback) for one prompt text."""

    attempts: tuple[dict[str, Any], ...]
    first_valid: bool
    first_optimal: bool
    first_objective: float | None
    optimal: bool
    objective: float | None
    prompt_tokens_first: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_and_solve(
    text: str,
    chat: DeepSeekChat,
    cache: ResponseCache,
    model: str = "deepseek-flash",
    max_attempts: int = 3,
    budget: Budget | None = None,
    max_tokens: int | None = None,
    system: str = MILP_SYSTEM,
    thinking: bool = False,
) -> GenerationResult:
    """DeepSeek writes a MilpSpec for ``text``; invalid or non-optimal models are fed back.

    Thinking mode counts its reasoning against ``max_tokens``, so its default budget is larger.
    """
    max_tokens = max_tokens or (32000 if thinking else 12000)
    msgs = milp_messages(text, system)
    tag = f"h2-filter-milp[{'think' if thinking else 'fast'}]"
    attempts: list[dict[str, Any]] = []
    cost = latency = 0.0
    p_first = p_tot = c_tot = 0
    optimal = False
    objective: float | None = None
    error: str | None = None
    for i in range(max_attempts):
        key = request_key(tag, model, msgs)
        hit = cache.get(key)
        fresh = hit is None
        if hit is None:
            try:
                if budget:
                    budget.check("deepseek")
                r = chat.chat(model, msgs, thinking=thinking, max_tokens=max_tokens)
            except Exception as exc:
                error = f"api: {type(exc).__name__}: {exc}"[:300]
                break
            hit = {
                "content": r.content,
                "usage": r.usage,
                "latency_ms": r.latency_ms,
                "model": r.model,
                "utc": r.utc,
            }
            cache.put(key, hit)
        ts = datetime.fromisoformat(hit["utc"])
        c = deepseek_cost(model, hit["usage"], ts)
        if fresh and budget:
            budget.charge("deepseek", c)
        cost += c
        latency += hit["latency_ms"]
        pt = int(hit["usage"].get("prompt_tokens", 0))
        p_first = pt if i == 0 else p_first
        p_tot += pt
        c_tot += int(hit["usage"].get("completion_tokens", 0))
        problem: str | None = None
        status = None
        obj = float("nan")
        try:
            spec = MilpSpec.model_validate(json.loads(hit["content"]))
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            problem = f"invalid model: {str(exc)[:400]}"
        else:
            status, _, obj, solve_ms = solve_spec(spec, time_limit_s=10.0)
            latency += solve_ms
            if status != "Optimal":
                problem = f"solver status {status}"
        attempts.append(
            {
                "valid": problem is None or problem.startswith("solver"),
                "status": status,
                "objective": None if math.isnan(obj) else round(obj, 6),
                "problem": problem,
                "prompt_tokens": pt,
                "completion_tokens": int(hit["usage"].get("completion_tokens", 0)),
                "finish_reason_len": len(hit["content"]),
                "peak": is_peak(ts),
            }
        )
        if problem is None:
            optimal, objective = True, obj
            break
        msgs = [
            *msgs,
            {"role": "assistant", "content": hit["content"]},
            {
                "role": "user",
                "content": f"The model was rejected: {problem}. Return a corrected JSON model.",
            },
        ]
    first = attempts[0] if attempts else {}
    return GenerationResult(
        attempts=tuple(attempts),
        first_valid=bool(first.get("valid")),
        first_optimal=first.get("status") == "Optimal",
        first_objective=first.get("objective"),
        optimal=optimal,
        objective=objective,
        prompt_tokens_first=p_first,
        prompt_tokens=p_tot,
        completion_tokens=c_tot,
        cost_usd=cost,
        latency_ms=round(latency, 1),
        error=error,
    )
