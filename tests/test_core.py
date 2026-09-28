"""Offline tests: no network, small scenarios, fast."""

from __future__ import annotations

import json

import pytest

from d4r.dispatch.cards import DecisionCard, Option, OtherTrain, TrainFacts, card_json, render_state
from d4r.dispatch.controller import Controller, option_values, run_episode
from d4r.engines.base import DefaultEngine, RandomEngine
from d4r.engines.cache import ResponseCache, request_key
from d4r.engines.llm import _letter_probs, parse_choice
from d4r.engines.milp import FixedMilpEngine, MilpSpec, check_selectors, fixed_model, solve_spec
from d4r.engines.rules import SlackPriorityEngine
from d4r.sim.deadlock import deadlocked
from d4r.sim.scenario import PRESETS


def _meet_card(slack_self: int = 20, slack_other: int = 0, repair: int = 0) -> DecisionCard:
    return DecisionCard(
        card_id="c1",
        scenario="T",
        seed=1,
        step=10,
        max_steps=100,
        kind="meet",
        train=TrainFacts(
            id="T0",
            status="moving",
            slack_steps=slack_self,
            remaining_cells=20,
            trains_queued_behind=0,
        ),
        others=(
            OtherTrain(
                id="T1",
                relation="oncoming on this train's route",
                distance_cells=8,
                status="moving",
                slack_steps=slack_other,
                remaining_cells=25,
                repair_steps_left=repair,
                shared_track_cells=5,
                trains_queued_behind=1,
            ),
        ),
        options=(Option(id="PROCEED", description="go"), Option(id="HOLD", description="wait")),
        default_option="PROCEED",
    )


def test_render_bucketed_and_raw_differ_only_in_numbers() -> None:
    c = _meet_card()
    b, r = render_state(c, "bucketed"), render_state(c, "raw")
    assert b.keys() == r.keys()
    assert isinstance(r["oncoming_trains"][0]["distance_ahead_on_route"], int)
    assert "near" in b["oncoming_trains"][0]["distance_ahead_on_route"]
    assert json.loads(card_json(c)) == b


def test_deadlock_cycle_and_propagation() -> None:
    occupant = {(0, 0): 1, (0, 1): 2, (0, 2): 3}
    intended = {1: (0, 1), 2: (0, 0), 3: (0, 1)}  # 1<->2 cycle, 3 waits on 2
    assert deadlocked(occupant, intended) == {1, 2, 3}
    assert deadlocked(occupant, {1: (0, 1)}) == set()


def test_episode_is_deterministic_and_default_matches_itself() -> None:
    r1, _ = run_episode(PRESETS["T"], 3, DefaultEngine())
    r2, _ = run_episode(PRESETS["T"], 3, DefaultEngine())
    assert r1.to_dict() | {"wall_s": 0} == r2.to_dict() | {"wall_s": 0}
    assert r1.n_trains == 3


def test_random_engine_is_seeded_by_card() -> None:
    c = _meet_card()
    assert RandomEngine().decide([c])[0].option_id == RandomEngine().decide([c])[0].option_id


def test_slack_rule() -> None:
    eng = SlackPriorityEngine()
    assert eng.choose(_meet_card(slack_self=20, slack_other=0)) == "HOLD"
    assert eng.choose(_meet_card(slack_self=0, slack_other=20)) == "PROCEED"
    assert eng.choose(_meet_card(slack_self=20, slack_other=0, repair=30)) == "PROCEED"


def test_fixed_milp_prefers_holding_the_train_with_reserve() -> None:
    assert FixedMilpEngine().decide([_meet_card(20, -5)])[0].option_id == "HOLD"
    assert FixedMilpEngine().decide([_meet_card(-5, 20)])[0].option_id == "PROCEED"
    spec = fixed_model(_meet_card())
    assert check_selectors(spec, ["PROCEED", "HOLD"]) is None
    status, values, _, _ = solve_spec(spec)
    assert status == "Optimal" and round(values["choose_PROCEED"] + values["choose_HOLD"]) == 1


def test_milp_spec_rejects_undeclared_variables() -> None:
    with pytest.raises(ValueError):
        MilpSpec.model_validate(
            {
                "variables": [{"name": "x"}],
                "constraints": [{"terms": {"y": 1}, "sense": "<=", "rhs": 1}],
                "objective": {"terms": {"x": 1}},
            }
        )


def test_parse_choice_and_letter_probs() -> None:
    assert parse_choice('{"choice": "B", "confidence": 70}', ["A", "B"]) == ("B", 0.7)
    assert parse_choice("junk", ["A", "B"]) == (None, None)
    lp = [
        {"token": '{"', "logprob": 0.0, "top": []},
        {"token": "choice", "logprob": 0.0, "top": []},
        {"token": '":"', "logprob": 0.0, "top": []},
        {
            "token": "A",
            "logprob": -0.1,
            "top": [{"token": "A", "logprob": -0.1}, {"token": "B", "logprob": -2.4}],
        },
    ]
    p = _letter_probs(lp, ["A", "B"])
    assert p is not None and p["A"] > 0.85 and abs(sum(p.values()) - 1) < 1e-9


def test_cache_roundtrip(tmp_path) -> None:
    c = ResponseCache(tmp_path / "c.jsonl")
    k = request_key("e", "m", {"a": 1})
    c.put(k, {"x": 1})
    assert ResponseCache(tmp_path / "c.jsonl").get(k) == {"x": 1}


def test_rollout_values_are_reproducible() -> None:
    """Forking mid-step and rolling out the default equals the real continuation."""
    ctl = Controller(PRESETS["B"], 4)
    while not ctl.done:
        points = ctl.prepare()
        if points:
            v1 = option_values(ctl, points[0])
            v2 = option_values(ctl, points[0])
            assert v1 == v2
            base = ctl.total_reward
            ctl.commit(ctl.decide(points))
            ctl.run_to_end()
            assert round(ctl.total_reward - base, 3) == v1[points[0].card.default_option]
            return
        ctl.commit([])
    pytest.skip("no decision point in this episode")


def test_engines_keep_an_empty_file_backed_cache(tmp_path) -> None:
    """Regression: an empty ResponseCache is falsy (``__len__``); engines must not replace it."""
    from d4r.engines.jev import JevEngine
    from d4r.engines.llm import DeepSeekEngine
    from d4r.engines.milp import DeepSeekLPEngine

    c = ResponseCache(tmp_path / "x.jsonl")
    assert len(c) == 0
    for eng in (JevEngine(cache=c), DeepSeekEngine(cache=c), DeepSeekLPEngine(cache=c)):
        assert eng.cache is c
