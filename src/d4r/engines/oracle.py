"""Rollout oracle: label a decision by simulating every option to the end of the episode.

For each option the oracle forks the full simulator state, applies the option (all other
decisions in the same step take their default), lets the DLA default run the rest of the episode,
and scores the total Flatland reward from now on. This is one step of policy improvement over DLA
(a rollout algorithm in the sense of Bertsekas): the label says which option is best *given that
the plain interlocking continues afterwards*. It is the ground truth for decision-level accuracy,
regret and calibration, and, used as an engine, a compute-heavy upper reference.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any

from d4r.dispatch.cards import DecisionCard
from d4r.engines.base import Decision

__all__ = ["RolloutOracle", "softmax_values"]


def softmax_values(values: dict[str, float], temperature: float = 5.0) -> dict[str, float]:
    """Turn rollout values (total reward, higher is better) into a reference distribution."""
    m = max(values.values())
    ex = {k: math.exp((v - m) / temperature) for k, v in values.items()}
    z = sum(ex.values())
    return {k: round(v / z, 6) for k, v in ex.items()}


class RolloutOracle:
    """Chooses the option with the best rollout value; ties go to the card's default option."""

    name = "rollout-oracle"

    def decide(self, cards: Sequence[DecisionCard], ctx: Any = None) -> list[Decision]:
        from d4r.dispatch.controller import option_values

        if ctx is None:
            raise ValueError("RolloutOracle needs the live controller as ctx")
        points = {p.card.card_id: p for p in ctx._points}
        out = []
        for card in cards:
            t0 = time.perf_counter()
            values = option_values(ctx, points[card.card_id])
            best = max(values.values())
            winners = [o for o in card.option_ids() if values[o] == best]
            choice = card.default_option if card.default_option in winners else winners[0]
            out.append(
                Decision(
                    card.card_id,
                    choice,
                    probabilities=softmax_values(values),
                    latency_ms=round(1000 * (time.perf_counter() - t0), 1),
                    meta={"values": values},
                )
            )
        return out
