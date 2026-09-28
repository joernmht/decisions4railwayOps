"""The engine contract every contestant implements, plus trivial reference engines.

An engine receives decision cards and returns one :class:`Decision` per card: the chosen option id
and, where the engine has one, a probability for every option. Engines never touch the simulator;
the controller applies their choices. Invalid or failed answers fall back to the card's default
option and are counted as fallbacks.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from d4r.dispatch.cards import DecisionCard

__all__ = ["Decision", "DefaultEngine", "Engine", "RandomEngine"]


@dataclass(frozen=True)
class Decision:
    """An engine's answer for one card."""

    card_id: str
    option_id: str
    probabilities: dict[str, float] | None = None
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def p_top(self) -> float | None:
        """Probability of the chosen option, if the engine reports probabilities."""
        if not self.probabilities:
            return None
        return self.probabilities.get(self.option_id)


class Engine(Protocol):
    """A dispatching decision engine."""

    name: str

    def decide(self, cards: Sequence[DecisionCard], ctx: Any = None) -> list[Decision]:
        """Answer every card. ``ctx`` is the live controller (only the rollout oracle uses it)."""
        ...


class DefaultEngine:
    """Never intervenes: always the card's default option (what the DLA interlocking would do)."""

    name = "dla-default"

    def decide(self, cards: Sequence[DecisionCard], ctx: Any = None) -> list[Decision]:
        return [
            Decision(
                c.card_id,
                c.default_option,
                {o: float(o == c.default_option) for o in c.option_ids()},
            )
            for c in cards
        ]


class RandomEngine:
    """Uniformly random option, seeded by the card id so reruns are identical."""

    name = "random"

    def __init__(self, salt: str = "") -> None:
        self.salt = salt

    def decide(self, cards: Sequence[DecisionCard], ctx: Any = None) -> list[Decision]:
        out = []
        for c in cards:
            ids = c.option_ids()
            h = int(hashlib.sha256((c.card_id + self.salt).encode()).hexdigest(), 16)
            out.append(Decision(c.card_id, ids[h % len(ids)], {o: 1.0 / len(ids) for o in ids}))
        return out
