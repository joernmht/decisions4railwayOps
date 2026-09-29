"""Jev (TypeSafe AI) as a dispatching engine: one Choice question per decision card.

The card's rendered state is Jev's ``state``; the card's options become the Choice ``criteria``
(option id -> description), so Jev can only answer with a safe option. Jev returns a probability
for every option; we keep the whole distribution for the calibration analysis. Arithmetic stays in
code: the default card is bucketed, following Jev's documented weakness with raw numbers.

The model is pinned (``jev-1.13.0``) rather than ``jev-latest``, and every call is cached.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Literal

from d4r.apikeys import load_secrets
from d4r.dispatch.cards import DecisionCard, render_state
from d4r.engines.base import Decision
from d4r.engines.cache import ResponseCache, request_key

__all__ = ["JEV_PRICE_PER_INPUT_TOKEN", "JevEngine", "jev_question"]

#: USD per input token (docs.typesafe.ai/models, 2026-09-28: $0.042 / 1M input, output free).
JEV_PRICE_PER_INPUT_TOKEN = 0.042e-6


def jev_question(card: DecisionCard, guided: bool = False) -> dict[str, Any]:
    """The Choice question for a card, in the SDK's dict form."""
    q = {
        "type": "choice",
        "instructions": {
            "question": (
                "You are the train dispatcher. Which option should be taken for train `train.id`"
                " right now?"
            ),
            "goal": "Follow `objective`: the lowest total delay of all trains at their destinations.",
            "evidence": (
                "Weigh `train` (its schedule reserve, remaining distance, trains queued behind it)"
                " against the trains in `oncoming_trains`, if any. A train that is late or has many"
                " trains behind it should usually not be the one that waits; waiting for a train"
                " that is broken down for a long time wastes time."
            ),
        },
        "criteria": {o.id: o.description for o in card.options},
    }
    if guided:
        q["instructions"]["rules"] = (
            "Apply `dispatching_rules`: where they decide which train goes first, follow them unless"
            " a justified exception applies; where they do not decide, follow `objective`."
        )
    return q


class JevEngine:
    """Dispatching engine backed by TypeSafe's Jev."""

    def __init__(
        self,
        model: str = "jev-1.13.0",
        variant: Literal["bucketed", "raw"] = "bucketed",
        cache: ResponseCache | None = None,
        timeout_s: float = 10.0,
        guidance: str | None = None,
    ) -> None:
        self.model = model
        self.guidance = guidance
        self.variant = variant
        self.cache = cache if cache is not None else ResponseCache(None)
        self.timeout_s = timeout_s
        self.name = (
            f"jev[{model}{'' if variant == 'bucketed' else ',raw'}{',guided' if guidance else ''}]"
        )
        self._client: Any = None

    def _client_or_init(self) -> Any:
        if self._client is None:
            load_secrets()
            from typesafe_sdk import TypeSafeClient

            self._client = TypeSafeClient(model=self.model, timeout=self.timeout_s)
        return self._client

    def decide_one(self, card: DecisionCard) -> Decision:
        state = render_state(card, self.variant)
        question = jev_question(card, guided=self.guidance is not None)
        if self.guidance:
            state = {**state, "dispatching_rules": self.guidance}
        key = request_key("jev", self.model, {"state": state, "q": question})
        hit = self.cache.get(key)
        if hit is None:
            client = self._client_or_init()
            t0 = time.perf_counter()
            try:
                r = client.system_one(state=state, questions={"action": question})
            except Exception as exc:
                return Decision(
                    card.card_id, card.default_option, error=f"jev: {type(exc).__name__}: {exc}"
                )
            latency = 1000 * (time.perf_counter() - t0)
            ans = r.choices["action"]
            server_ms = None
            try:
                server_ms = r.raw_http_response.headers.get("x-envoy-upstream-service-time")
            except Exception:
                server_ms = None
            hit = {
                "choice": ans.choice,
                "probabilities": {k: float(v) for k, v in ans.probabilities.items()},
                "confidence": float(ans.confidence),
                "model": r.model,
                "input_tokens": int(r.usage.input_tokens or 0),
                "output_tokens": int(r.usage.output_tokens or 0),
                "latency_ms": round(latency, 1),
                "server_ms": server_ms,
                "request_id": getattr(r, "request_id", None),
                "utc": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            self.cache.put(key, hit)
        return Decision(
            card_id=card.card_id,
            option_id=hit["choice"],
            probabilities=hit["probabilities"],
            latency_ms=hit["latency_ms"],
            input_tokens=hit["input_tokens"],
            output_tokens=hit["output_tokens"],
            cost_usd=hit["input_tokens"] * JEV_PRICE_PER_INPUT_TOKEN,
            model=hit["model"],
            meta={
                "confidence": hit["confidence"],
                "server_ms": hit.get("server_ms"),
                "cache_key": key,
            },
        )

    def decide(self, cards: Sequence[DecisionCard], ctx: Any = None) -> list[Decision]:
        return [self.decide_one(c) for c in cards]
