"""Offline tests for P2-H2a/b routing: generator, parsers, prompt builders and metrics (no network)."""

from __future__ import annotations

import csv
import json
import re
from typing import get_args

import pytest

from d4r.h2.requirements import (
    DISTRACTORS,
    FAMILY_SPECS,
    constraint_families,
    family_descriptions,
    generate,
    keyword_route,
    template_texts,
)
from d4r.h2.routing import (
    Answer,
    Budget,
    BudgetExceeded,
    accuracy,
    calibration_bins,
    catalog_messages,
    class_descriptions,
    class_descriptions_path,
    class_ids,
    class_ids_path,
    gate_curve,
    gate_cv,
    jev_catalog_question,
    jev_family_questions,
    load_catalog,
    load_class_descriptions,
    macro_f1,
    multilabel_scores,
    negation_scores,
    paired_bootstrap,
    parse_multi_label,
    parse_single_label,
    per_class_f1,
    per_label_predict,
    percentile,
    requirement_messages,
    tfidf_similarities,
    threshold_predict,
    tokenize,
    topk_predict,
    tune_per_label_thresholds,
    tune_threshold,
)

# --------------------------------------------------------------------------- requirements


def test_families_come_from_lp2graph_enum() -> None:
    from lp2graph.core.model import ConstraintDomainClass

    enum = get_args(ConstraintDomainClass)
    fams = constraint_families()
    assert fams == tuple(f for f in enum if f != "unclassified")
    assert set(FAMILY_SPECS) == set(fams)
    assert list(family_descriptions()) == list(fams)


def test_every_family_has_templates_in_both_splits() -> None:
    for fam, spec in FAMILY_SPECS.items():
        for kind in ("pos", "flip", "waiver"):
            assert spec.pool(kind, "dev"), (fam, kind)
            assert spec.pool(kind, "test"), (fam, kind)
        dev_idx = {i for i, _ in spec.pool("pos", "dev")}
        test_idx = {i for i, _ in spec.pool("pos", "test")}
        assert not dev_idx & test_idx


def test_generator_is_seeded_and_well_formed() -> None:
    a = generate(80, seed=7)
    b = generate(80, seed=7)
    c = generate(80, seed=8)
    assert a == b
    assert [s.text for s in a] != [s.text for s in c]
    fams = set(constraint_families())
    for s in a:
        assert 1 <= len(s.gold) <= 3
        assert set(s.gold) <= fams
        assert set(s.flipped) <= set(s.gold)
        assert not set(s.waived) & set(s.gold)
        assert "{" not in s.text and "}" not in s.text
        assert s.negated == bool(s.flipped or s.waived)
    assert any(s.flipped for s in a) and any(s.waived for s in a)
    assert any(s.n_distractors for s in a)
    # a prefix of a longer run is the shorter run (resumable, --limit consistent)
    assert generate(10, seed=7) == a[:10]


def test_test_split_uses_only_held_out_templates() -> None:
    dev_n = {"pos": 2, "flip": 1, "waiver": 1}
    for s in generate(150, seed=3, split="test"):
        for tid in s.template_ids:
            if tid.startswith("distractor:"):
                continue
            _, kind, idx = tid.split(":")
            assert int(idx) >= dev_n[kind], tid
    for s in generate(60, seed=3, split="dev"):
        for tid in s.template_ids:
            if tid.startswith("distractor:"):
                continue
            _, kind, idx = tid.split(":")
            assert int(idx) < dev_n[kind], tid


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    w = re.findall(r"[a-z]+", text.lower())
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


@pytest.mark.parametrize("style", ["modelling", "requirement"])
def test_descriptions_do_not_quote_test_templates(style) -> None:
    desc = " ".join(family_descriptions(style).values())
    dgrams = _ngrams(desc, 5)
    for tpl in template_texts("test"):
        assert not _ngrams(tpl, 5) & dgrams, tpl


def test_keyword_router_is_negation_blind() -> None:
    assert "headway_separation" in keyword_route("A minimum headway of 3 minutes applies.")
    # the waiver mentions the same keyword and is (wrongly, by design) routed too
    assert "headway_separation" in keyword_route("Headways do not need to be modelled.")
    assert keyword_route("Station Riesa was opened in 1901.") == ()


def test_distractors_fill_cleanly() -> None:
    s = generate(200, seed=11)
    used = {t for x in s for t in x.template_ids if t.startswith("distractor")}
    assert len(used) == len(DISTRACTORS)


# --------------------------------------------------------------------------- catalogue


def test_load_catalog_hides_labels_and_paper(tmp_path) -> None:
    path = tmp_path / "variables_classified.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "paper",
                "symbol",
                "variable_type",
                "description",
                "canonical_category",
                "canonical_name",
                "ml_class",
            ]
        )
        w.writerow(["Secret2020", "x", "binary", "train is cancelled", "C", "N", "Class A"])
        w.writerow(["Secret2020", "y", "", "no label", "C", "N", ""])
    items = load_catalog("variables", tmp_path)
    assert items is not None and len(items) == 1
    it = items[0]
    assert it.item_id == "variables-000" and it.gold == "Class A"
    assert "Secret2020" not in it.text and "Class A" not in it.text
    assert it.text == "variable_type: binary | description: train is cancelled"
    assert load_catalog("constraints", tmp_path) is None


def _private_descriptions(tmp_path, monkeypatch):
    """A stand-in for the private class-description file, with neutral dummy classes."""
    path = tmp_path / "private" / "descriptions.json"
    path.parent.mkdir()
    table = {"variables": {"Class B": "second dummy class", "Class A": "first dummy class"}}
    path.write_text(json.dumps(table), encoding="utf-8")
    monkeypatch.setenv("D4R_CLASS_DESCRIPTIONS", str(path))
    return path, table


def test_class_descriptions_load_lazily_from_private_file(tmp_path, monkeypatch) -> None:
    path, table = _private_descriptions(tmp_path, monkeypatch)
    assert class_descriptions_path() == path
    assert class_ids_path() == path.with_name("h2a_class_ids.json")
    assert load_class_descriptions() == table
    d = class_descriptions("variables", ["Class B", "Class C", "Class A"])
    assert d == {
        "Class A": "first dummy class",
        "Class B": "second dummy class",
        "Class C": "Class C",
    }
    assert list(d) == sorted(d)
    # the same result with an explicit table (no file access)
    assert class_descriptions("variables", ["Class A"], known=table) == {
        "Class A": "first dummy class"
    }


def test_class_descriptions_fall_back_to_name_without_private_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("D4R_CLASS_DESCRIPTIONS", str(tmp_path / "absent.json"))
    assert load_class_descriptions() == {}
    d = class_descriptions("constraints", ["Class B", "Class A"])
    assert d == {"Class A": "Class A", "Class B": "Class B"}


def test_class_ids_are_opaque_and_sorted_per_kind() -> None:
    ids = class_ids("constraints", ["Class B", "Class A", "Class C", "Class A"])
    assert ids == {"Class A": "C01", "Class B": "C02", "Class C": "C03"}
    assert class_ids("variables", ["x"]) == {"x": "V01"}
    assert class_ids("parameters", ["x"]) == {"x": "P01"}
    assert class_ids("objectives", ["y", "x"]) == {"x": "O01", "y": "O02"}


def test_prompt_builders() -> None:
    desc = {"A": "first", "B": "second"}
    q = jev_catalog_question("constraints", desc)
    assert q["type"] == "choice" and q["criteria"] == desc
    msgs = catalog_messages("constraints", desc, "x <= y")
    assert "JSON" in msgs[0]["content"] and msgs[1]["content"] == "constraint: x <= y"
    fq = jev_family_questions(family_descriptions())
    assert list(fq) == list(constraint_families())
    assert all(v["type"] == "noul" and set(v["criteria"]) == {"true", "false"} for v in fq.values())
    rm = requirement_messages(family_descriptions(), "No overtaking at Riesa.")
    assert all(f in rm[0]["content"] for f in constraint_families())
    assert "json" in rm[0]["content"].lower()
    cues = family_descriptions("requirement")
    lit = jev_family_questions(cues, "requirement")
    assert lit["headway_separation"]["instructions"] == (
        f"Does `statement` ask for {cues['headway_separation']}?"
    )
    rr = requirement_messages(cues, "x", "requirement")[0]["content"]
    assert "- timing_window: the statement asks for " in rr


# --------------------------------------------------------------------------- parsers


def test_parse_single_label() -> None:
    labels = ["Class A one", "Class B/two"]
    assert parse_single_label('{"class": "Class A one"}', labels) == "Class A one"
    assert parse_single_label('{"class": "class  a one"}', labels) == "Class A one"
    assert parse_single_label('Sure: {"class": "Class B/two"}', labels) == "Class B/two"
    assert parse_single_label('{"class": "Other"}', labels) is None
    assert parse_single_label("not json", labels) is None


def test_parse_multi_label() -> None:
    labels = list(constraint_families())
    got, unknown = parse_multi_label(
        json.dumps({"families": ["timing_window", "precedence ordering", "made_up"]}), labels
    )
    assert got == ("precedence_ordering", "timing_window") and unknown == 1
    assert parse_multi_label('{"families": []}', labels) == ((), 0)
    assert parse_multi_label('{"families": "timing_window"}', labels) == (("timing_window",), 0)
    assert parse_multi_label("garbage", labels) == ((), 0)


# --------------------------------------------------------------------------- metrics


def test_single_label_metrics() -> None:
    y = ["a", "a", "b", "c"]
    p = ["a", "b", "b", None]
    assert accuracy(y, p) == 0.5
    f = per_class_f1(y, p, ["a", "b", "c"])
    assert f["a"] == pytest.approx(2 / 3) and f["b"] == pytest.approx(2 / 3) and f["c"] == 0.0
    assert macro_f1(y, p, ["a", "b", "c"]) == pytest.approx(4 / 9)


def test_percentile_matches_linear_interpolation() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([10.0], 95) == 10.0
    assert percentile([], 50) is None
    assert percentile(list(range(101)), 95) == pytest.approx(95.0)


def test_paired_bootstrap_is_seeded_and_brackets_point() -> None:
    a = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1] * 3
    b = [1, 0, 0, 1, 0, 0, 1, 1, 0, 0] * 3

    def diff(idx):
        return (sum(a[i] for i in idx) - sum(b[i] for i in idx)) / len(idx)

    r1 = paired_bootstrap(diff, len(a), n_boot=500, seed=1)
    r2 = paired_bootstrap(diff, len(a), n_boot=500, seed=1)
    assert r1 == r2
    assert r1["ci95_lo"] <= r1["diff"] <= r1["ci95_hi"]
    assert r1["diff"] == pytest.approx(0.3)


def test_gate_curve_controls() -> None:
    y = ["a", "b", "a", "b"]
    fast = ["a", "a", "a", "b"]  # wrong on item 1
    p = [0.9, 0.4, 0.95, 0.8]
    slow = ["b", "b", "a", "a"]  # right on items 1 and 2
    rows = gate_curve(y, fast, p, slow, ["a", "b"], [0.0, 0.5, 1.01], n_random=200, seed=0)
    none, gated, allslow = rows
    assert none["coverage"] == 1.0 and none["cascade_acc"] == 0.75
    assert gated["escalated"] == 1 and gated["cascade_acc"] == 1.0
    assert gated["fast_acc_on_kept"] == 1.0
    # random escalation of one item: E = 3/4 * 0.75 + 1/4 * 0.5
    assert gated["random_escalation_acc"] == pytest.approx(0.6875)
    assert gated["oracle_escalation_acc"] == 1.0
    assert allslow["coverage"] == 0.0 and allslow["cascade_acc"] == 0.5
    assert allslow["random_escalation_acc"] == pytest.approx(0.5)


def test_gate_cv_chooses_theta_on_other_folds() -> None:
    # Jev is right exactly when p >= 0.7 and DeepSeek is always right: theta 0.7 and 0.9 tie on
    # every training split and the first in grid order (0.7) must be chosen for every fold.
    p = [0.95, 0.6, 0.8, 0.6, 0.99, 0.75, 0.6, 0.9, 0.85, 0.6]
    y = ["a"] * 10
    fast = ["a" if x >= 0.7 else "b" for x in p]
    slow = ["a"] * 10
    r = gate_cv(y, fast, p, slow, [0.5, 0.7, 0.9], k=5, n_splits=20, seed=3)
    assert r is not None
    assert r["cascade_acc_mean"] == 1.0 and r["cascade_acc_lo"] == 1.0
    assert r["theta_chosen"] == {"0.7": 100}
    assert r["coverage_mean"] == pytest.approx(0.6)
    assert r["in_sample_best_theta"] == 0.7 and r["in_sample_best_acc"] == 1.0
    assert r["fast_acc"] == pytest.approx(0.6) and r["slow_acc"] == 1.0
    assert r["cascade_minus_slow_mean"] == 0.0
    # seeded and reproducible; degenerate input returns None
    assert gate_cv(y, fast, p, slow, [0.5, 0.7, 0.9], n_splits=20, seed=3) == r
    assert gate_cv(["a"], ["a"], [0.9], ["a"], [0.5]) is None


def test_gate_cv_is_held_out() -> None:
    # One item per fold (k = n): its theta is chosen on the other items only. Item 0 is the only
    # one where a high theta pays off, so leaving it out picks the low theta for it.
    y = ["a", "a", "a", "a"]
    fast = ["b", "a", "a", "a"]
    p = [0.8, 0.8, 0.8, 0.8]
    slow = ["a", "b", "b", "a"]
    r = gate_cv(y, fast, p, slow, [0.5, 0.9], k=4, n_splits=5, seed=0)
    assert r is not None
    # in-sample: theta 0.5 keeps Jev everywhere (3/4), theta 0.9 escalates all (2/4)
    assert r["in_sample_best_theta"] == 0.5 and r["in_sample_best_acc"] == 0.75
    assert r["cascade_acc_mean"] == 0.75


def test_calibration_bins() -> None:
    bins = calibration_bins(
        [0.1, 0.5, 0.95, 1.0, None], [False, True, True, True, False], (0.0, 0.5, 1.0)
    )
    assert bins[0] == {"bin": "0.00-0.50", "n": 1, "acc": 0.0}
    assert bins[1]["n"] == 3 and bins[1]["acc"] == 1.0


def test_multilabel_and_negation_scores() -> None:
    labels = ["x", "y", "z"]
    gold = [("x",), ("x", "y"), ("z",)]
    pred = [("x",), ("x",), ("y", "z")]
    m = multilabel_scores(gold, pred, labels)
    assert m["micro_p"] == pytest.approx(3 / 4) and m["micro_r"] == pytest.approx(3 / 4)
    assert m["exact_match"] == pytest.approx(1 / 3)
    assert m["per_label_f1"]["x"] == 1.0 and m["per_label_f1"]["y"] == 0.0
    n = negation_scores(gold, [("x",), (), ()], [(), (), ("y",)], pred)
    assert n["flip_recall"] == 1.0 and n["waiver_fp_rate"] == 1.0
    assert n["n_negated_statements"] == 2
    assert n["exact_match_negated"] == 0.5 and n["exact_match_plain"] == 0.0


def test_threshold_topk_and_tuning() -> None:
    s = {"x": 0.9, "y": 0.2, "z": 0.6}
    assert threshold_predict(s, 0.5) == ("x", "z")
    assert threshold_predict(s, 0.95) == ()
    assert threshold_predict(s, 0.95, min_k=1) == ("x",)
    assert topk_predict(s, 2) == ("x", "z")
    assert topk_predict({"a": 0.5, "b": 0.5}, 1) == ("a",)
    t, f = tune_threshold(
        [s, {"x": 0.1, "y": 0.7, "z": 0.3}],
        [("x",), ("y",)],
        ["x", "y", "z"],
        [0.1, 0.5, 0.65, 0.8],
    )
    assert t == 0.65 and f == 1.0


def test_tfidf_prefers_matching_document() -> None:
    docs = ["minimum headway between consecutive trains", "periodic cyclic timetable period"]
    sims = tfidf_similarities(["keep a headway between trains", "a cyclic period"], docs)
    assert sims[0][0] > sims[0][1] and sims[1][1] > sims[1][0]
    assert "the" not in tokenize("the trains")
    assert tfidf_similarities(["zzz"], docs) == [[0.0, 0.0]]


def test_budget_and_answer() -> None:
    b = Budget({"jev": 0.01})
    b.check("jev")
    b.charge("jev", 0.02)
    with pytest.raises(BudgetExceeded):
        b.check("jev")
    b.check("deepseek")  # no cap -> never exceeded
    a = Answer("i", scores={"A": 0.7, "B": 0.3}, choice="A")
    assert a.p_top == 0.7
    assert Answer("j").p_top is None


def test_per_label_thresholds() -> None:
    scores = [{"x": 0.9, "y": 0.6}, {"x": 0.4, "y": 0.8}, {"x": 0.7, "y": 0.3}]
    gold = [("x",), ("y",), ("x",)]
    th = tune_per_label_thresholds(scores, gold, ["x", "y", "z"], [0.3, 0.5, 0.7])
    assert th == {"x": 0.5, "y": 0.7, "z": 0.5}
    assert per_label_predict({"x": 0.55, "y": 0.65, "z": 0.9}, th) == ("x", "z")
