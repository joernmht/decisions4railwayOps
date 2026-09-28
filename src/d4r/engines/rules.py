"""Rule-based dispatchers: codified priority rules of the kind used in dispatching practice.

- ``slack-priority``: the train with more schedule reserve waits. At a meeting, hold this train only
  if it has clearly more reserve than every oncoming train and none of them is broken down for a
  long time; before departure, wait only if the train can afford it and an oncoming train is close.
  This is the delay-based priority rule that the ARI family of rail rules and dispatcher practice
  share (D'Ariano et al. 2008); first-come-first-served is the ``dla-default`` engine.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from d4r.dispatch.cards import DecisionCard
from d4r.engines.base import Decision

__all__ = ["SlackPriorityEngine"]


class SlackPriorityEngine:
    """The train with more schedule reserve yields."""

    name = "rule-slack-priority"

    def __init__(self, margin: int = 3, depart_wait: int = 5, near: int = 6) -> None:
        self.margin = margin
        self.depart_wait = depart_wait
        self.near = near

    def choose(self, c: DecisionCard) -> str:
        t = c.train
        if c.kind == "meet":
            if not c.others:
                return "PROCEED"
            if any(o.repair_steps_left > 10 for o in c.others):
                return "PROCEED"
            if all(t.slack_steps >= o.slack_steps + self.margin for o in c.others):
                return "HOLD"
            return "PROCEED"
        if c.kind == "depart":
            close = [
                o for o in c.others if o.distance_cells <= self.near and o.repair_steps_left == 0
            ]
            if close and t.slack_steps >= self.depart_wait + self.margin:
                return "WAIT"
            return "DEPART_NOW"
        return c.default_option

    def decide(self, cards: Sequence[DecisionCard], ctx: Any = None) -> list[Decision]:
        out = []
        for c in cards:
            ch = self.choose(c)
            out.append(Decision(c.card_id, ch, {o: float(o == ch) for o in c.option_ids()}))
        return out
