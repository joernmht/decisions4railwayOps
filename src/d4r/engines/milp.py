"""Small MILPs for dispatching decisions: a validated JSON model spec, a HiGHS solve, two engines.

- :class:`FixedMilpEngine` (**OR baseline, no LLM**): code builds a delay/lateness model of the
  decision from the raw card and HiGHS picks the option.
- :class:`DeepSeekLPEngine` (**LLM + LP execution**): DeepSeek writes the MILP itself, as a JSON
  model spec, from the raw card; we validate it, solve it with HiGHS and apply the option whose
  selector variable ``choose_<OPTION_ID>`` is 1. The LLM never emits code: the spec is data,
  validated against :class:`MilpSpec` before anything is solved (same stance as the
  raiLParchitect loop, ADR-0001 there).

Both arms receive the **raw** card (exact integers), because a model needs coefficients; the
direct-choice arms can be run on the raw card too (ablation P2-H6) to separate information from
method.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from d4r.dispatch.cards import DecisionCard, render_state
from d4r.engines.base import Decision
from d4r.engines.cache import ResponseCache, request_key
from d4r.engines.llm import DeepSeekChat, deepseek_cost, is_peak

__all__ = ["DeepSeekLPEngine", "FixedMilpEngine", "MilpSpec", "fixed_model", "solve_spec"]


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Var(_M):
    name: str = Field(min_length=1, max_length=64)
    type: Literal["binary", "integer", "continuous"] = "continuous"
    lb: float | None = 0.0
    ub: float | None = None


class Row(_M):
    name: str = ""
    terms: dict[str, float]
    sense: Literal["<=", ">=", "=="]
    rhs: float


class Objective(_M):
    sense: Literal["min", "max"] = "min"
    terms: dict[str, float]
    constant: float = 0.0


class MilpSpec(_M):
    """A small MILP as data. Selector variables are named ``choose_<OPTION_ID>``."""

    variables: list[Var] = Field(min_length=1, max_length=500)
    constraints: list[Row] = Field(default_factory=list, max_length=2000)
    objective: Objective

    @model_validator(mode="after")
    def _names_resolve(self) -> MilpSpec:
        names = [v.name for v in self.variables]
        if len(set(names)) != len(names):
            raise ValueError("duplicate variable names")
        known = set(names)
        for row in self.constraints:
            unknown = set(row.terms) - known
            if unknown:
                raise ValueError(
                    f"constraint {row.name!r} uses undeclared variables {sorted(unknown)}"
                )
        unknown = set(self.objective.terms) - known
        if unknown:
            raise ValueError(f"objective uses undeclared variables {sorted(unknown)}")
        for x in [*self.constraints, self.objective]:
            coefs = x.terms.values()
            if any(not math.isfinite(c) for c in coefs):
                raise ValueError("non-finite coefficient")
        return self


def check_selectors(spec: MilpSpec, option_ids: Sequence[str]) -> str | None:
    """The spec must declare one binary selector per option and force exactly one to 1."""
    by_name = {v.name: v for v in spec.variables}
    sel = [f"choose_{o}" for o in option_ids]
    missing = [s for s in sel if s not in by_name]
    if missing:
        return f"missing selector variables {missing}"
    if any(by_name[s].type != "binary" for s in sel):
        return "selector variables must be binary"
    for row in spec.constraints:
        if (
            row.sense == "=="
            and set(row.terms) == set(sel)
            and all(abs(c - 1) < 1e-9 for c in row.terms.values())
            and abs(row.rhs - 1) < 1e-9
        ):
            return None
    return "missing constraint: sum of selector variables == 1"


def solve_spec(
    spec: MilpSpec, time_limit_s: float = 5.0
) -> tuple[str, dict[str, float], float, float]:
    """Solve with HiGHS. Returns (status, values, objective, solve_ms)."""
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit_s))
    inf = highspy.kHighsInf
    xs: dict[str, Any] = {}
    for v in spec.variables:
        lb = -inf if v.lb is None else float(v.lb)
        ub = inf if v.ub is None else float(v.ub)
        if v.type == "binary":
            lb, ub = max(lb, 0.0), min(ub, 1.0)
        kind = (
            highspy.HighsVarType.kContinuous
            if v.type == "continuous"
            else highspy.HighsVarType.kInteger
        )
        xs[v.name] = h.addVariable(lb=lb, ub=ub, type=kind, name=v.name)

    def expr(terms: dict[str, float]) -> Any:
        e = 0
        for name, c in sorted(terms.items()):
            e = e + float(c) * xs[name]
        return e

    for row in spec.constraints:
        e = expr(row.terms)
        if row.sense == "<=":
            h.addConstr(e <= row.rhs)
        elif row.sense == ">=":
            h.addConstr(e >= row.rhs)
        else:
            h.addConstr(e == row.rhs)
    t0 = time.perf_counter()
    obj = expr(spec.objective.terms)
    if spec.objective.sense == "min":
        h.minimize(obj)
    else:
        h.maximize(obj)
    ms = 1000 * (time.perf_counter() - t0)
    status = h.modelStatusToString(h.getModelStatus())
    if status != "Optimal":
        return status, {}, float("nan"), ms
    values = {n: float(h.val(x)) for n, x in xs.items()}
    return status, values, float(h.getInfo().objective_function_value) + spec.objective.constant, ms


# --------------------------------------------------------------------------- fixed model
def fixed_model(card: DecisionCard, depart_wait: int = 5) -> MilpSpec:
    """Code-built lateness model of a decision (the OR baseline; no LLM).

    One step = one cell of travel. For a *meet*, whichever train waits loses roughly the distance
    between the trains (the other must travel it to pass); a broken-down oncoming train adds its
    repair time to the waiting. Lateness is delay beyond the schedule reserve, weighted by one plus
    the number of trains queued behind (they wait too). For a *depart*, waiting costs the train
    ``depart_wait`` steps; departing makes the nearest oncoming train yield instead.
    """
    t = card.train
    others = list(card.others)
    vars_: list[Var] = [Var(name=f"choose_{o}", type="binary") for o in card.option_ids()]
    rows: list[Row] = [
        Row(
            name="one_option",
            terms={f"choose_{o}": 1.0 for o in card.option_ids()},
            sense="==",
            rhs=1.0,
        )
    ]
    obj: dict[str, float] = {}

    def lateness(name: str, delay_terms: dict[str, float], slack: int, weight: float) -> None:
        # L >= sum(delay_terms) - slack, L >= 0
        vars_.append(Var(name=name, type="continuous", lb=0.0))
        terms = {name: 1.0} | {k: -v for k, v in delay_terms.items()}
        rows.append(Row(name=f"{name}_def", terms=terms, sense=">=", rhs=-float(slack)))
        obj[name] = obj.get(name, 0.0) + weight
        # small tie-breaker on raw delay so that zero-lateness cases still prefer less waiting
        for k, v in delay_terms.items():
            obj[k] = obj.get(k, 0.0) + 0.01 * weight * v

    if card.kind == "meet":
        wait_self = max((o.distance_cells + o.repair_steps_left for o in others), default=0)
        lateness(
            "late_self",
            {"choose_HOLD": float(wait_self)},
            t.slack_steps,
            1.0 + t.trains_queued_behind,
        )
        for i, o in enumerate(others):
            lateness(
                f"late_o{i}",
                {"choose_PROCEED": float(o.distance_cells)},
                o.slack_steps,
                1.0 + o.trains_queued_behind,
            )
    elif card.kind == "depart":
        lateness("late_self", {"choose_WAIT": float(depart_wait)}, t.slack_steps, 1.0)
        if others:
            o = min(others, key=lambda x: x.distance_cells)
            lateness(
                "late_o",
                {"choose_DEPART_NOW": float(o.distance_cells)},
                o.slack_steps,
                1.0 + o.trains_queued_behind,
            )
    return MilpSpec(variables=vars_, constraints=rows, objective=Objective(sense="min", terms=obj))


def _chosen(values: dict[str, float], option_ids: Sequence[str]) -> str | None:
    on = [o for o in option_ids if values.get(f"choose_{o}", 0.0) > 0.5]
    return on[0] if len(on) == 1 else None


class FixedMilpEngine:
    """OR baseline: code-built lateness MILP, solved with HiGHS."""

    name = "fixed-milp"

    def decide(self, cards: Sequence[DecisionCard], ctx: Any = None) -> list[Decision]:
        out = []
        for c in cards:
            t0 = time.perf_counter()
            spec = fixed_model(c)
            status, values, objective, _ = solve_spec(spec)
            choice = _chosen(values, c.option_ids()) if status == "Optimal" else None
            out.append(
                Decision(
                    c.card_id,
                    choice or c.default_option,
                    probabilities={
                        o: float(o == (choice or c.default_option)) for o in c.option_ids()
                    },
                    latency_ms=round(1000 * (time.perf_counter() - t0), 2),
                    error=None if choice else f"solve: {status}",
                    meta={"objective": objective},
                )
            )
        return out


# --------------------------------------------------------------------------- LLM + LP
LP_SYSTEM = (
    "You are an operations-research engineer supporting a railway traffic dispatcher. For the"
    " decision below, write a small mixed-integer linear program (MILP) that models the delay"
    " consequences of the options, so that its optimal solution selects the best option. Rules:"
    " (1) declare one binary variable named choose_<OPTION_ID> per option (for example"
    " choose_PROCEED) and a constraint that their sum equals 1; (2) you may add further variables"
    " (entry times, precedence binaries, delays, lateness) and linear constraints; (3) one step of"
    " time equals one cell of travel for every train; (4) use only numbers given in the situation."
    " Return a JSON object only, with this structure:"
    ' {"variables": [{"name": str, "type": "binary"|"integer"|"continuous", "lb": number|null,'
    ' "ub": number|null}], "constraints": [{"name": str, "terms": {"<var>": coefficient},'
    ' "sense": "<="|">="|"==", "rhs": number}], "objective": {"sense": "min"|"max",'
    ' "terms": {"<var>": coefficient}, "constant": number}}.'
)


class DeepSeekLPEngine:
    """LLM + LP execution: DeepSeek writes the MILP, HiGHS solves it, code applies the option."""

    def __init__(
        self,
        model: str = "deepseek-flash",
        thinking: bool = False,
        max_attempts: int = 3,
        cache: ResponseCache | None = None,
        chat: DeepSeekChat | None = None,
    ) -> None:
        self.model = model
        self.thinking = thinking
        self.max_attempts = max_attempts
        self.cache = cache or ResponseCache(None)
        self.chat = chat or DeepSeekChat()
        self.name = f"{model}+lp[{'think' if thinking else 'fast'}]"

    def _first_messages(self, card: DecisionCard) -> list[dict[str, str]]:
        state = render_state(card, "raw")
        opts = "\n".join(f"- {o.id}: {o.description}" for o in card.options)
        user = (
            "Situation (JSON, exact integers; distances and reserves in cells = steps):\n"
            + json.dumps(state, ensure_ascii=False, indent=1)
            + "\n\nOptions:\n"
            + opts
            + "\n\nReturn the MILP as the JSON object described."
        )
        return [{"role": "system", "content": LP_SYSTEM}, {"role": "user", "content": user}]

    def _call(self, msgs: list[dict[str, str]]) -> dict[str, Any]:
        key = request_key(self.name, self.model, msgs)
        hit = self.cache.get(key)
        if hit is None:
            r = self.chat.chat(self.model, msgs, thinking=self.thinking, max_tokens=16000)
            hit = {
                "content": r.content,
                "usage": r.usage,
                "latency_ms": r.latency_ms,
                "model": r.model,
                "utc": r.utc,
            }
            self.cache.put(key, hit)
        return hit

    def decide_one(self, card: DecisionCard) -> Decision:
        from datetime import datetime

        msgs = self._first_messages(card)
        attempts: list[dict[str, Any]] = []
        latency = cost = 0.0
        tin = tout = 0
        model = ""
        choice: str | None = None
        for _ in range(self.max_attempts):
            try:
                hit = self._call(msgs)
            except Exception as exc:
                attempts.append({"error": f"api: {type(exc).__name__}: {exc}"})
                break
            ts = datetime.fromisoformat(hit["utc"])
            latency += hit["latency_ms"]
            cost += deepseek_cost(self.model, hit["usage"], ts)
            tin += hit["usage"].get("prompt_tokens", 0)
            tout += hit["usage"].get("completion_tokens", 0)
            model = hit["model"]
            problem: str | None = None
            try:
                spec = MilpSpec.model_validate(json.loads(hit["content"]))
                problem = check_selectors(spec, card.option_ids())
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                problem = f"invalid model: {str(exc)[:400]}"
            if problem is None:
                status, values, objective, solve_ms = solve_spec(spec)
                latency += solve_ms
                if status == "Optimal":
                    choice = _chosen(values, card.option_ids())
                    problem = None if choice else "solution does not select exactly one option"
                else:
                    problem = f"solver status {status}"
                attempts.append(
                    {
                        "problem": problem,
                        "status": status,
                        "objective": objective,
                        "solve_ms": round(solve_ms, 2),
                        "peak": is_peak(ts),
                    }
                )
            else:
                attempts.append({"problem": problem, "peak": is_peak(ts)})
            if choice:
                break
            msgs = [
                *msgs,
                {"role": "assistant", "content": hit["content"]},
                {
                    "role": "user",
                    "content": f"The model was rejected: {problem}. Return a corrected JSON model.",
                },
            ]
        return Decision(
            card_id=card.card_id,
            option_id=choice or card.default_option,
            probabilities={
                o: float(o == (choice or card.default_option)) for o in card.option_ids()
            },
            latency_ms=round(latency, 1),
            input_tokens=tin,
            output_tokens=tout,
            cost_usd=cost,
            model=model,
            error=None if choice else f"no valid model after {len(attempts)} attempt(s)",
            meta={"attempts": attempts},
        )

    def decide(self, cards: Sequence[DecisionCard], ctx: Any = None) -> list[Decision]:
        return [self.decide_one(c) for c in cards]
