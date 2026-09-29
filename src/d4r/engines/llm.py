"""DeepSeek as a dispatching engine (generative LLM baseline), plus the shared chat client.

The LLM receives exactly the same rendered card as Jev, with the options listed under single-letter
labels, and must answer JSON ``{"choice": "<letter>", "confidence": <0-100>}``. Two probability
readings are recorded for the calibration comparison (P2-H3):

- ``probabilities``: from token log-probabilities of the answer letter (non-thinking mode only;
  thinking mode returns no usable log-probabilities, so its distribution is one-hot);
- ``meta.verbal_confidence``: the model's self-reported confidence (known to be overconfident).

Costs follow DeepSeek's published prices (checked 2026-09-28): peak hours (Mon–Fri 01–04 and 06–10
UTC) cost twice the off-peak rate; cache hits are billed at the cache-hit input price.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from d4r.apikeys import load_secrets
from d4r.dispatch.cards import DecisionCard, render_state
from d4r.engines.base import Decision
from d4r.engines.cache import ResponseCache, request_key

__all__ = ["DeepSeekChat", "DeepSeekEngine", "deepseek_cost", "letter_options"]

#: USD per 1M tokens, off-peak (cache-hit input, cache-miss input, output). Peak = 2x.
_PRICES = {
    "deepseek-flash": (0.003, 0.15, 0.60),
    "deepseek-v4-pro": (0.022, 0.66, 1.98),
}
LETTERS = "ABCDEFGH"

SYSTEM_PROMPT = (
    "You are an experienced railway traffic dispatcher. You receive the current situation of one"
    " train as JSON and a closed list of safe options prepared by the interlocking. Choose the"
    " option that best serves the stated objective. Answer with a JSON object only, of the form"
    ' {"choice": "<option letter>", "confidence": <integer 0-100>}.'
)


def is_peak(ts: datetime) -> bool:
    """DeepSeek peak pricing window (UTC)."""
    return ts.weekday() < 5 and (1 <= ts.hour < 4 or 6 <= ts.hour < 10)


def deepseek_cost(model: str, usage: dict[str, int], ts: datetime) -> float:
    """USD cost of one call from its usage record."""
    hit, miss, out = _PRICES.get(model, _PRICES["deepseek-flash"])
    k = 2.0 if is_peak(ts) else 1.0
    cost = (
        usage.get("prompt_cache_hit_tokens", 0) * hit
        + usage.get("prompt_cache_miss_tokens", usage.get("prompt_tokens", 0)) * miss
        + usage.get("completion_tokens", 0) * out
    )
    return k * cost / 1e6


def letter_options(card: DecisionCard) -> list[tuple[str, str, str]]:
    """(letter, option id, description) in card order."""
    return [(LETTERS[i], o.id, o.description) for i, o in enumerate(card.options)]


@dataclass
class ChatResult:
    content: str
    reasoning: str
    usage: dict[str, int]
    latency_ms: float
    model: str
    logprobs: list[dict[str, Any]] | None
    utc: str


class DeepSeekChat:
    """Thin wrapper around the OpenAI-compatible DeepSeek endpoint."""

    def __init__(
        self, base_url: str = "https://api.deepseek.com", timeout_s: float = 180.0
    ) -> None:
        self.base_url = base_url
        self.timeout_s = timeout_s
        self._client: Any = None

    def _c(self) -> Any:
        if self._client is None:
            load_secrets()
            from openai import OpenAI

            self._client = OpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
                base_url=self.base_url,
                timeout=self.timeout_s,
                max_retries=3,
            )
        return self._client

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        thinking: bool,
        json_mode: bool = True,
        logprobs: bool = False,
        max_tokens: int = 4000,
        temperature: float = 0.0,
    ) -> ChatResult:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "extra_body": {"thinking": {"type": "enabled" if thinking else "disabled"}},
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if not thinking:
            kwargs["temperature"] = temperature
            if logprobs:
                kwargs["logprobs"] = True
                kwargs["top_logprobs"] = 5
        t0 = time.perf_counter()
        r = self._c().chat.completions.create(**kwargs)
        latency = 1000 * (time.perf_counter() - t0)
        msg = r.choices[0].message
        usage = r.usage.model_dump() if r.usage is not None else {}
        usage = {k: int(v) for k, v in usage.items() if isinstance(v, int)}
        lp = None
        if logprobs and r.choices[0].logprobs is not None and r.choices[0].logprobs.content:
            lp = [
                {
                    "token": t.token,
                    "logprob": t.logprob,
                    "top": [
                        {"token": a.token, "logprob": a.logprob} for a in (t.top_logprobs or [])
                    ],
                }
                for t in r.choices[0].logprobs.content
            ]
        return ChatResult(
            content=msg.content or "",
            reasoning=getattr(msg, "reasoning_content", None) or "",
            usage=usage,
            latency_ms=round(latency, 1),
            model=r.model,
            logprobs=lp,
            utc=datetime.now(UTC).isoformat(timespec="seconds"),
        )


def _letter_probs(
    logprobs: list[dict[str, Any]] | None, letters: list[str]
) -> dict[str, float] | None:
    """Distribution over option letters at the position where the answer letter was emitted."""
    if not logprobs:
        return None
    text = ""
    for tok in logprobs:
        clean = tok["token"].strip().strip('"').strip()
        if clean in letters and "choice" in text:
            mass: dict[str, float] = dict.fromkeys(letters, 0.0)
            for alt in tok["top"]:
                a = alt["token"].strip().strip('"').strip()
                if a in mass:
                    mass[a] += math.exp(alt["logprob"])
            if sum(mass.values()) == 0:
                mass[clean] = 1.0
            z = sum(mass.values())
            return {k: v / z for k, v in mass.items()}
        text += tok["token"]
    return None


def parse_choice(content: str, letters: list[str]) -> tuple[str | None, float | None]:
    """Letter and verbal confidence from the model's JSON answer (tolerant of stray text)."""
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.S)
        if not m:
            return None, None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None, None
    letter = str(obj.get("choice", "")).strip().strip('"').upper()[:1]
    conf = obj.get("confidence")
    try:
        conf_f = float(conf) / 100.0 if conf is not None else None
    except (TypeError, ValueError):
        conf_f = None
    return (letter if letter in letters else None), conf_f


class DeepSeekEngine:
    """Direct-choice LLM engine (DeepSeek, thinking or non-thinking)."""

    def __init__(
        self,
        model: str = "deepseek-flash",
        thinking: bool = False,
        variant: Literal["bucketed", "raw"] = "bucketed",
        cache: ResponseCache | None = None,
        chat: DeepSeekChat | None = None,
        guidance: str | None = None,
    ) -> None:
        self.model = model
        self.thinking = thinking
        self.variant = variant
        self.guidance = guidance
        self.cache = cache if cache is not None else ResponseCache(None)
        self.chat = chat if chat is not None else DeepSeekChat()
        mode = "think" if thinking else "fast"
        self.name = f"{model}[{mode}{'' if variant == 'bucketed' else ',raw'}]"

    def messages(self, card: DecisionCard) -> list[dict[str, str]]:
        state = render_state(card, self.variant)
        opts = "\n".join(f"{L}: {oid} - {desc}" for L, oid, desc in letter_options(card))
        user = (
            "Situation (JSON):\n"
            + json.dumps(state, ensure_ascii=False, indent=1)
            + "\n\nOptions:\n"
            + opts
            + '\n\nReply with the JSON object {"choice": ..., "confidence": ...} only.'
        )
        system = SYSTEM_PROMPT
        if self.guidance:
            system += (
                "\n\n" + self.guidance + "\nWhere these rules decide which train goes first, follow"
                " them unless a justified exception applies; otherwise follow the objective."
            )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def decide_one(self, card: DecisionCard) -> Decision:
        msgs = self.messages(card)
        key = request_key(self.name, self.model, msgs)
        hit = self.cache.get(key)
        if hit is None:
            try:
                r = self.chat.chat(
                    self.model,
                    msgs,
                    thinking=self.thinking,
                    logprobs=not self.thinking,
                    max_tokens=16000,
                )
            except Exception as exc:
                return Decision(
                    card.card_id,
                    card.default_option,
                    error=f"deepseek: {type(exc).__name__}: {exc}",
                )
            hit = {
                "content": r.content,
                "reasoning_chars": len(r.reasoning),
                "usage": r.usage,
                "latency_ms": r.latency_ms,
                "model": r.model,
                "logprobs": r.logprobs,
                "utc": r.utc,
            }
            self.cache.put(key, hit)
        lopts = letter_options(card)
        letters = [L for L, _, _ in lopts]
        to_id = {L: oid for L, oid, _ in lopts}
        letter, verbal = parse_choice(hit["content"], letters)
        err = None if letter else f"unparseable answer: {hit['content'][:120]!r}"
        option = to_id[letter] if letter else card.default_option
        lp = _letter_probs(hit.get("logprobs"), letters)
        probs = {to_id[k]: round(v, 6) for k, v in lp.items()} if lp else None
        if probs is None and letter:
            probs = {oid: float(oid == option) for _, oid, _ in lopts}
        usage = hit["usage"]
        ts = datetime.fromisoformat(hit["utc"])
        return Decision(
            card_id=card.card_id,
            option_id=option,
            probabilities=probs,
            latency_ms=hit["latency_ms"],
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            cost_usd=deepseek_cost(self.model, usage, ts),
            model=hit["model"],
            error=err,
            meta={
                "verbal_confidence": verbal,
                "reasoning_chars": hit.get("reasoning_chars", 0),
                "peak": is_peak(ts),
                "cache_key": key,
            },
        )

    def decide(self, cards: Sequence[DecisionCard], ctx: Any = None) -> list[Decision]:
        return [self.decide_one(c) for c in cards]
