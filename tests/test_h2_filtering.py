"""Offline tests for the P2-H2 filter arm: statement generator, reference optimum, metrics.

No network: the paid arms are exercised through pre-filled caches and a fake chat client.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from d4r.engines.cache import ResponseCache, request_key
from d4r.engines.llm import ChatResult
from d4r.engines.milp import solve_spec
from d4r.h2.filtering import (
    Budget,
    BudgetExceeded,
    FilterScores,
    JevSentenceFilter,
    ReplayCache,
    ReplayMiss,
    aggregate,
    choose_floor_threshold,
    generate_and_solve,
    keep_all_scores,
    kept_ids,
    mcnemar_exact_p,
    objective_matches,
    parse_id_list,
    regex_scores,
    sentence_metrics,
)
from d4r.h2.statements import (
    NOISE_KINDS,
    make_statement,
    make_statements,
    reference_milp,
    reference_optimum,
    schedule_for_orders,
    weighted_tardiness,
)


def _tiny_instance(**over: Any) -> dict[str, Any]:
    inst: dict[str, Any] = {
        "stations": ["A", "B", "C"],
        "sections": ["A–B", "B–C"],
        "trains": [
            {"name": "X 1", "release": 0, "run": [10, 10], "due": 20, "weight": 1},
            {"name": "Y 2", "release": 0, "run": [5, 5], "due": 10, "weight": 3},
        ],
        "headway": 2,
        "no_overtake": [],
        "faults": [],
        "blocks": [],
        "corrections": [],
    }
    inst.update(over)
    return inst


def test_statements_are_deterministic_and_well_formed() -> None:
    a, b = make_statements(12, seed=7), make_statements(12, seed=7)
    assert [s.text() for s in a] == [s.text() for s in b]
    assert make_statement(3, seed=7).text() != make_statement(3, seed=8).text()
    for s in a:
        assert 10 <= len(s.sentences) <= 25
        assert s.ids() == [f"s{i + 1:02d}" for i in range(len(s.sentences))]
        assert s.sentences[0].kind == "frame" and s.sentences[0].needed
        for x in s.sentences:
            assert x.needed == (x.kind not in NOISE_KINDS)
        kinds = [x.kind for x in s.sentences]
        assert kinds.count("superseded") == kinds.count("correction")
        if "superseded" in kinds:
            assert kinds.index("superseded") < kinds.index("correction")
        assert s.trap_ids() <= set(s.ids()) - s.needed_ids()
        assert s.data_ids() <= s.needed_ids()


def test_reference_optimum_matches_highs_on_the_reference_milp() -> None:
    for s in make_statements(8, seed=11):
        ref = reference_optimum(s.instance)
        status, _, obj, _ = solve_spec(reference_milp(s.instance))
        assert status == "Optimal"
        assert obj == pytest.approx(ref.objective, abs=1e-6)
        assert weighted_tardiness(s.instance, ref.schedule) == ref.objective


def test_schedule_respects_headway_blocks_faults_and_order_rules() -> None:
    inst = _tiny_instance()
    sched = schedule_for_orders(inst, [("X 1", "Y 2"), ("X 1", "Y 2")])
    assert sched["X 1"] == [(0, 10), (10, 20)]
    assert sched["Y 2"] == [(12, 17), (22, 27)]  # headway 2 after X leaves each section
    # best: Y first everywhere -> X exits at 7 + 10 + 10 = 27 -> tardiness 7 * 1
    assert reference_optimum(inst).objective == 7
    blocked = _tiny_instance(blocks=[{"section": "B–C", "until": 30}])
    assert reference_optimum(blocked).objective == (47 - 20) + 3 * (35 - 10)  # both wait
    faulty = _tiny_instance(faults=[{"train": "Y 2", "section": "A–B", "extra": 4}])
    assert reference_optimum(faulty).schedule["Y 2"][0] == (0, 9)
    fixed = _tiny_instance(no_overtake=[{"behind": "Y 2", "ahead": "X 1"}])
    ref = reference_optimum(fixed)
    assert all(order[0] == "X 1" for order in ref.orders)
    assert ref.objective == 3 * (27 - 10)


def test_metrics_gold_keep_all_and_fail_open() -> None:
    s = make_statement(0)
    gold = sentence_metrics(s, s.needed_ids())
    agg = aggregate([gold])
    assert agg["recall_needed"] == 1.0 and agg["precision"] == 1.0
    assert agg["trap_kept"] == 0.0 and agg["stmt_all_needed_kept"] == 1.0
    allm = aggregate([sentence_metrics(s, kept_ids(keep_all_scores(s)))])
    assert allm["precision"] == pytest.approx(len(s.needed_ids()) / len(s.sentences), abs=1e-4)
    assert allm["char_reduction"] == 0.0
    failed = FilterScores("jev", s.stmt_id, dict.fromkeys(s.ids(), 0.0), error="boom")
    assert kept_ids(failed, 0.5) == frozenset(s.ids())
    drop_one = set(s.data_ids()) - {sorted(s.data_ids())[0]}
    m = sentence_metrics(s, drop_one)
    assert m.fn >= 1 and aggregate([m])["stmt_all_data_kept"] == 0.0


def test_regex_baseline_keeps_every_trap_sentence() -> None:
    s = make_statement(4)
    kept = kept_ids(regex_scores(s))
    assert s.trap_ids() <= kept
    assert all(any(ch.isdigit() for ch in x.text) == (x.sid in kept) for x in s.sentences)


def test_parse_id_list_is_tolerant_and_filters_unknown_ids() -> None:
    known = ["s01", "s02", "s03"]
    assert parse_id_list('{"needed": ["s03", "s01", "s99"]}', known) == ["s01", "s03"]
    assert parse_id_list('Sure: {"needed": ["s02"]} done', known) == ["s02"]
    assert parse_id_list("no json here", known) is None
    assert parse_id_list('{"needed": "s01"}', known) is None


def test_threshold_choice_and_mcnemar() -> None:
    rows = [
        {"threshold": 0.1, "recall_needed": 1.0},
        {"threshold": 0.5, "recall_needed": 1.0},
        {"threshold": 0.7, "recall_needed": 0.98},
    ]
    assert choose_floor_threshold(rows) == 0.5
    assert choose_floor_threshold([{"threshold": 0.2, "recall_needed": 0.9}]) == 0.2
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(0, 5) == pytest.approx(0.0625)
    assert mcnemar_exact_p(3, 3) == 1.0
    assert objective_matches(51.0000001, 51) and not objective_matches(48.0, 51)


def test_jev_filter_questions_and_cached_scoring() -> None:
    s = make_statement(2)
    jev = JevSentenceFilter(cache=ResponseCache(None))
    qs = jev.questions(s)
    assert list(qs) == s.ids()
    assert all(f"`sentences.{sid}`" in q["instructions"] for sid, q in qs.items())
    assert all(q["type"] == "noul" and "criteria" in q for q in qs.values())
    assert "criteria" not in next(iter(JevSentenceFilter(criteria=False).questions(s).values()))
    key = request_key("h2-filter-jev", jev.model, {"state": jev.state(s), "q": qs})
    nouls = {x.sid: (0.9 if x.needed else 0.1) for x in s.sentences}
    jev.cache.put(
        key,
        {
            "nouls": nouls,
            "model": "jev-1.13.0",
            "input_tokens": 1000,
            "output_tokens": 10,
            "latency_ms": 250.0,
        },
    )
    fs = jev.score(s)  # cache hit, no network
    assert not fs.fresh and fs.cost_usd == pytest.approx(1000 * 0.042e-6)
    assert kept_ids(fs, 0.5) == s.needed_ids()


class _FakeChat:
    """Returns canned answers in order and records the conversations it was sent."""

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, model: str, messages: list[dict[str, str]], **_: Any) -> ChatResult:
        self.calls.append(messages)
        return ChatResult(
            content=self.answers.pop(0),
            reasoning="",
            usage={"prompt_tokens": 800, "completion_tokens": 1500},
            latency_ms=100.0,
            model=model,
            logprobs=None,
            utc=datetime(2026, 9, 28, 21, 0, tzinfo=UTC).isoformat(),
        )


def test_generate_and_solve_feeds_back_invalid_models() -> None:
    s = make_statement(1)
    good = reference_milp(s.instance).model_dump_json()
    chat = _FakeChat(["{not json", good])
    budget = Budget(caps={"deepseek": 1.0})
    g = generate_and_solve(s.text(), chat, ResponseCache(None), budget=budget)  # type: ignore[arg-type]
    assert len(g.attempts) == 2 and not g.first_valid and g.optimal
    assert objective_matches(g.objective or 0.0, reference_optimum(s.instance).objective)
    assert "rejected" in chat.calls[1][-1]["content"]
    assert g.prompt_tokens_first == 800 and g.prompt_tokens == 1600
    assert budget.spent["deepseek"] == pytest.approx(g.cost_usd)
    assert json.loads(good)["objective"]["sense"] == "min"


def test_budget_blocks_fresh_calls_once_the_cap_is_reached() -> None:
    b = Budget(caps={"jev": 0.01})
    b.check("jev")
    b.charge("jev", 0.02)
    with pytest.raises(BudgetExceeded):
        b.check("jev")
    b.check("deepseek")  # no cap configured


def test_replay_cache_stops_on_a_miss_instead_of_calling_an_api(tmp_path) -> None:
    path = tmp_path / "cache.jsonl"
    ResponseCache(path).put("k", {"v": 1})
    replay = ReplayCache(path)
    assert replay.get("k") == {"v": 1}
    with pytest.raises(ReplayMiss):
        replay.get("missing")
    with pytest.raises(ReplayMiss):
        replay.put("new", {"v": 2})
    # a paid arm on a replay cache raises at the lookup, before any client is created
    jev = JevSentenceFilter(cache=ReplayCache(tmp_path / "empty.jsonl"))
    with pytest.raises(ReplayMiss):
        jev.score(make_statements(1, 5)[0])
    assert jev._client is None
