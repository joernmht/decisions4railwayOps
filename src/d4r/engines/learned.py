"""Learned dispatcher: cost-weighted logistic regression on decision-card facts.

This is the *information ceiling* reference of the contest: how well can the raw facts on a
decision card predict the rollout-best option when a model is fitted to labelled cards? It is
trained on game-set cards of seeds disjoint from the benchmark (seeds 51–200 vs 1–50), with each
card weighted by its spread (the regret at stake), so the fitted rule minimises expected regret
rather than error rate. If this simple model beats the zero-shot engines, the card carries signal
they fail to use; if it does not, the card itself is the bottleneck.

Implemented in plain numpy (deterministic full-batch gradient descent, L2 penalty) to keep the lab
dependency-light.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from d4r.dispatch.cards import DecisionCard
from d4r.engines.base import Decision

__all__ = ["FEATURES", "LearnedEngine", "card_features", "fit_logistic"]

FEATURES = (
    "bias",
    "is_meet",
    "slack",
    "slack_neg",
    "remaining",
    "queued",
    "status_moving",
    "status_stopped",
    "n_others",
    "min_dist",
    "min_other_slack",
    "min_other_slack_neg",
    "max_other_repair",
    "sum_other_queued",
    "max_shared",
    "min_other_remaining",
    "slack_minus_other",
    "time_share",
    "horizon",
)


def card_features(card: DecisionCard) -> list[float]:
    """Numeric features of a card (raw facts, lightly scaled)."""
    t = card.train
    o = list(card.others)
    min_dist = min((x.distance_cells for x in o), default=30)
    min_os = min((x.slack_steps for x in o), default=0)
    f = {
        "bias": 1.0,
        "is_meet": float(card.kind == "meet"),
        "slack": t.slack_steps / 20.0,
        "slack_neg": float(t.slack_steps < 0),
        "remaining": t.remaining_cells / 30.0,
        "queued": float(t.trains_queued_behind),
        "status_moving": float(t.status == "moving"),
        "status_stopped": float(t.status.startswith("stopped")),
        "n_others": float(len(o)),
        "min_dist": min_dist / 15.0,
        "min_other_slack": min_os / 20.0,
        "min_other_slack_neg": float(min_os < 0),
        "max_other_repair": max((x.repair_steps_left for x in o), default=0) / 20.0,
        "sum_other_queued": float(sum(x.trains_queued_behind for x in o)),
        "max_shared": max((x.shared_track_cells for x in o), default=0) / 15.0,
        "min_other_remaining": min((x.remaining_cells for x in o), default=0) / 30.0,
        "slack_minus_other": (t.slack_steps - min_os) / 20.0,
        "time_share": card.step / max(1, card.max_steps),
        "horizon": card.max_steps / 100.0,
    }
    return [f[k] for k in FEATURES]


def fit_logistic(
    records: Iterable[dict[str, Any]], l2: float = 1e-2, steps: int = 4000, lr: float = 0.5
) -> dict[str, Any]:
    """Fit P(non-default option is best) on labelled game-set records, weighted by spread."""
    xs, ys, ws = [], [], []
    for r in records:
        if r["spread"] <= 0:
            continue
        card = DecisionCard.model_validate(r["card"])
        xs.append(card_features(card))
        ys.append(float(card.default_option not in r["best"]))
        ws.append(float(r["spread"]))
    x = np.asarray(xs)
    y = np.asarray(ys)
    w = np.asarray(ws)
    w = w / w.mean()
    beta = np.zeros(x.shape[1])
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(x @ beta)))
        grad = x.T @ (w * (p - y)) / len(y) + l2 * np.r_[0.0, beta[1:]]
        beta -= lr * grad
    return {
        "features": list(FEATURES),
        "beta": [round(float(b), 6) for b in beta],
        "n": len(y),
        "positive_share": float(y.mean()),
        "l2": l2,
        "weighting": "spread",
    }


class LearnedEngine:
    """Choose the non-default option iff the fitted model says it is more likely best."""

    name = "learned-lr"

    def __init__(self, weights: dict[str, Any] | str | Path) -> None:
        if not isinstance(weights, dict):
            weights = json.loads(Path(weights).read_text(encoding="utf-8"))
        assert weights["features"] == list(FEATURES), "feature set changed; refit"
        self.beta = np.asarray(weights["beta"])

    def p_nondefault(self, card: DecisionCard) -> float:
        z = float(np.asarray(card_features(card)) @ self.beta)
        return 1.0 / (1.0 + np.exp(-z))

    def decide(self, cards: Sequence[DecisionCard], ctx: Any = None) -> list[Decision]:
        out = []
        for c in cards:
            p = self.p_nondefault(c)
            nd = [o for o in c.option_ids() if o != c.default_option]
            choice = nd[0] if (nd and p >= 0.5) else c.default_option
            probs = {o: (p if o in nd else 1.0 - p) for o in c.option_ids()}
            out.append(Decision(c.card_id, choice, probs))
        return out
