"""Composite engines built on a probability-returning engine: cost-sensitive choice and escalation.

- :class:`CostSensitiveEngine` keeps the fast engine's *probabilities* but moves the decision rule
  into code: choose the non-default option when its probability is at least θ. With calibrated
  probabilities the Bayes-optimal θ is c_FP / (c_FP + c_FN), the ratio of the average loss of a
  needless intervention to the average loss of a missed one; we estimate both from a development
  game set (labels only, no engine outputs), which is the TypeSafe pattern "keep policy explicit".
- :class:`GatedEngine` acts on the fast engine's choice when its top probability is at least θ and
  otherwise escalates the card to a slow engine (P2-H4). Latency and cost add up on escalation,
  because the fast call has already been made.

Both thresholds used in the experiments are fixed on scenario A before testing on B and C
(docs/lab-notebook.md, pilot2 pre-registration).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from d4r.dispatch.cards import DecisionCard
from d4r.engines.base import Decision, Engine

__all__ = ["CostSensitiveEngine", "GatedEngine"]


class CostSensitiveEngine:
    """Non-default option iff P(non-default) ≥ θ (two-option cards); otherwise the default."""

    def __init__(self, base: Engine, theta: float, name: str | None = None) -> None:
        self.base = base
        self.theta = theta
        self.name = name or f"{base.name}+cost[{theta}]"

    def decide(self, cards: Sequence[DecisionCard], ctx: Any = None) -> list[Decision]:
        out = []
        for card, d in zip(cards, self.base.decide(cards, ctx), strict=True):
            if d.error or not d.probabilities:
                out.append(d)
                continue
            others = [o for o in card.option_ids() if o != card.default_option]
            p_nd = sum(d.probabilities.get(o, 0.0) for o in others)
            best_nd = max(others, key=lambda o: d.probabilities.get(o, 0.0)) if others else None
            choice = best_nd if (best_nd and p_nd >= self.theta) else card.default_option
            out.append(
                replace(
                    d,
                    option_id=choice,
                    meta={**d.meta, "base_choice": d.option_id, "p_nondefault": p_nd},
                )
            )
        return out


class GatedEngine:
    """Fast engine if confident (p_top ≥ θ), else escalate to the slow engine."""

    def __init__(self, fast: Engine, slow: Engine, theta: float, name: str | None = None) -> None:
        self.fast = fast
        self.slow = slow
        self.theta = theta
        self.name = name or f"{fast.name}>{slow.name}[{theta}]"

    def decide(self, cards: Sequence[DecisionCard], ctx: Any = None) -> list[Decision]:
        fast = self.fast.decide(cards, ctx)
        esc_idx = [
            i for i, d in enumerate(fast) if d.error or d.p_top is None or d.p_top < self.theta
        ]
        slow = (
            dict(zip(esc_idx, self.slow.decide([cards[i] for i in esc_idx], ctx), strict=True))
            if esc_idx
            else {}
        )
        out = []
        for i, d in enumerate(fast):
            if i not in slow:
                out.append(replace(d, meta={**d.meta, "escalated": False}))
                continue
            s = slow[i]
            out.append(
                replace(
                    s,
                    latency_ms=d.latency_ms + s.latency_ms,
                    input_tokens=d.input_tokens + s.input_tokens,
                    output_tokens=d.output_tokens + s.output_tokens,
                    cost_usd=d.cost_usd + s.cost_usd,
                    meta={
                        **s.meta,
                        "escalated": True,
                        "fast_choice": d.option_id,
                        "fast_probabilities": d.probabilities,
                    },
                )
            )
        return out
