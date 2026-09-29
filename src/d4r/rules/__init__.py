"""Dispatching rules derived from DB InfraGO's Richtlinie 420.02 (paraphrased, cited).

The regulation itself is not redistributed. ``ril420.json`` holds our own paraphrases with exact
citations (module, section, paragraph, item, validity date), see
``docs/research/ril420-derived-rules.md``. This module turns the priority rules P1–P6 into a
formal order on service classes, renders the rules as plain text for guided engines, and assigns
service classes to Flatland trains in mixed-traffic scenarios.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

__all__ = [
    "SERVICE_LABELS",
    "SERVICE_RANK",
    "SERVICE_SPEED",
    "SERVICE_WEIGHT",
    "outranks",
    "priority_key",
    "rulebook",
    "rules_text",
]

#: Service classes, their Ril 420.0201 rank (P1..P5; lower = higher priority) and label.
SERVICE_RANK: dict[str, int] = {
    "urgent_relief": 1,
    "passenger_express": 2,
    "freight_express": 3,
    "freight_fast": 4,
    "passenger_other": 5,
    "freight_other": 5,
}
SERVICE_LABELS: dict[str, str] = {
    "urgent_relief": "urgent relief train",
    "passenger_express": "long-distance passenger train, very high priority (Express)",
    "freight_express": "freight train, very high priority (Express)",
    "freight_fast": "freight train, high priority (Fast)",
    "passenger_other": "regional passenger train",
    "freight_other": "freight train, standard priority",
}
#: Flatland speed (cells per step) of each service in the mixed-traffic scenarios.
SERVICE_SPEED: dict[str, float] = {
    "urgent_relief": 1.0,
    "passenger_express": 1.0,
    "freight_express": 0.5,
    "freight_fast": 0.5,
    "passenger_other": 0.5,
    "freight_other": 1.0 / 3.0,
}


#: Weights for the *priority-weighted delay* sensitivity objective. An assumption of this study,
#: not taken from Ril 420 (whose overarching objective, O0, weights all trains equally).
SERVICE_WEIGHT: dict[str, float] = {
    "urgent_relief": 4.0,
    "passenger_express": 3.0,
    "freight_express": 2.0,
    "freight_fast": 1.5,
    "passenger_other": 1.0,
    "freight_other": 1.0,
}


@lru_cache(maxsize=1)
def rulebook() -> dict[str, Any]:
    """The derived rulebook (``ril420.json``)."""
    text = resources.files(__package__).joinpath("ril420.json").read_text(encoding="utf-8")
    data: dict[str, Any] = json.loads(text)
    return data


def _rank_between(a: str, b: str) -> tuple[int, int]:
    """Effective ranks of two services for a pairwise comparison.

    Rule P4 (freight 'Fast') only ranks above *other freight*; against a regional passenger train
    both count as equal-standing trains of rule P5.
    """
    ra, rb = SERVICE_RANK[a], SERVICE_RANK[b]
    if a == "freight_fast" and b.startswith("passenger"):
        ra = 5
    if b == "freight_fast" and a.startswith("passenger"):
        rb = 5
    return ra, rb


def outranks(a: str, speed_a: float, b: str, speed_b: float) -> bool | None:
    """Does a train of service ``a`` go before one of service ``b`` under Ril 420.0201 P1–P6?

    Returns ``None`` when the rules give no order (equal rank and equal travel speed).
    """
    ra, rb = _rank_between(a, b)
    if ra != rb:
        return ra < rb
    if abs(speed_a - speed_b) > 1e-9:  # P6: the faster train goes first, in principle
        return speed_a > speed_b
    return None


def priority_key(service: str, speed: float) -> tuple[int, float]:
    """Sort key (ascending = higher priority); pairwise use :func:`outranks` for P4's scope."""
    return SERVICE_RANK[service], -speed


def rules_text() -> str:
    """The priority rules and objectives in plain English, for rule-guided engines."""
    rb = rulebook()
    lines = [
        "Dispatching rules of the infrastructure manager (paraphrased from DB InfraGO Ril 420.0201):"
    ]
    for o in rb["objectives"][:3]:
        lines.append(f"- Objective: {o['paraphrase']}")
    for p in rb["priority_rules"]:
        if p.get("modelled", True):
            lines.append(f"- Rule {p['id']}: {p['paraphrase']}")
    for c in rb["constraints"][:1]:
        lines.append(f"- {c['paraphrase']}")
    lines.append(f"- {rb['semantics'][0]['paraphrase']}")
    return "\n".join(lines)
