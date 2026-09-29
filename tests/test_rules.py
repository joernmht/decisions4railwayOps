"""Ril 420-derived rulebook: formal priority order, doc sync, service assignment."""

from __future__ import annotations

from pathlib import Path

from d4r.rules import outranks, rulebook, rulebook_markdown


def test_priority_order_p1_to_p6() -> None:
    assert outranks("urgent_relief", 1.0, "passenger_express", 1.0) is True
    assert outranks("passenger_express", 1.0, "freight_express", 0.5) is True
    assert outranks("freight_express", 0.5, "freight_fast", 0.5) is True
    assert outranks("freight_fast", 0.5, "freight_other", 0.33) is True
    # P4 ranks 'Fast' freight above other freight only; vs regional passenger P5/P6 decide
    assert outranks("freight_fast", 0.5, "passenger_other", 0.5) is None
    assert outranks("passenger_other", 0.5, "freight_other", 0.33) is True  # P6: faster first
    assert outranks("passenger_other", 0.5, "passenger_other", 0.5) is None


def test_every_rule_cites_module_and_date() -> None:
    rb = rulebook()
    for group in ("semantics", "objectives", "priority_rules", "constraints", "parameters"):
        for r in rb[group]:
            assert r["cite"]["module"] and r["cite"]["valid_from"], r["id"]


def test_rulebook_doc_is_generated_from_json() -> None:
    doc = Path(__file__).parents[1] / "docs" / "research" / "ril420-derived-rules.md"
    assert doc.read_text(encoding="utf-8") == rulebook_markdown()


def test_services_follow_speed() -> None:
    from d4r.rules import SERVICE_RANK
    from d4r.sim.scenario import PRESETS, assign_services, make_env

    env = make_env(PRESETS["M"], 3)
    sv = assign_services(env, 3)
    assert set(sv.values()) <= set(SERVICE_RANK)
    for a in env.agents:
        if float(a.speed_counter.max_speed) >= 0.99:
            assert sv[a.handle] == "passenger_express"
