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
    "rulebook_markdown",
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


def rulebook_markdown() -> str:
    """Render the derived rulebook as Markdown (docs/research/ril420-derived-rules.md)."""
    rb = rulebook()

    def cite(c: dict[str, Any]) -> str:
        parts = [f"Ril {c['module']}"]
        for k in ("section", "paragraph", "item", "entry"):
            if c.get(k):
                parts.append(f"{k} {c[k]}")
        return ", ".join(parts) + f" (valid from {c['valid_from']})"

    src = rb["source"]
    out = [
        "# Dispatching rules derived from DB InfraGO Ril 420.02",
        "",
        "Generated from `src/d4r/rules/ril420.json` by `d4r.rules.rulebook_markdown()`; do not edit",
        "by hand.",
        "",
        f"**Source.** {src['document']}, {src['edition']}. Cite as `{src['citation_key']}`.",
        "",
        f"**Status.** {src['note']} The regulation itself is not part of this repository.",
        "",
        "## Semantics",
        "",
    ]
    out += [f"- **{s['id']}** {s['paraphrase']} — {cite(s['cite'])}" for s in rb["semantics"]]
    out += [
        "",
        "## Objectives",
        "",
        "| id | regime | paraphrase | formal reading | source |",
        "|---|---|---|---|---|",
    ]
    out += [
        f"| {o['id']} | {o.get('regime', 'overall')} | {o['paraphrase']} | {o.get('formal', '')} | {cite(o['cite'])} |"
        for o in rb["objectives"]
    ]
    out += [
        "",
        "## Priority rules (order of trains)",
        "",
        "| id | paraphrase | classes | modelled | source |",
        "|---|---|---|---|---|",
    ]
    out += [
        f"| {p['id']} | {p['paraphrase']} | {', '.join(p.get('classes', [])) or p.get('tie_break', '')} | {'yes' if p.get('modelled', True) else 'no: ' + p.get('why_not', '')} | {cite(p['cite'])} |"
        for p in rb["priority_rules"]
    ]
    out += [
        "",
        "## Constraints and process rules",
        "",
        "| id | paraphrase | formal reading | source |",
        "|---|---|---|---|",
    ]
    out += [
        f"| {c['id']} | {c['paraphrase']} | {c.get('formal', 'not modelled')} | {cite(c['cite'])} |"
        for c in rb["constraints"]
    ]
    out += [
        "",
        "## Parameters",
        "",
        "| id | name | value | paraphrase | source |",
        "|---|---|---:|---|---|",
    ]
    out += [
        f"| {k['id']} | `{k['name']}` | {k['value']} {k['unit']} | {k['paraphrase']} | {cite(k['cite'])} |"
        for k in rb["parameters"]
    ]
    m = rb["measures"]
    out += [
        "",
        "## Catalogue of dispatching measures",
        "",
        f"{m['paraphrase']} — {cite(m['cite'])}",
        "",
    ]
    for group in ("general", "with_railway_undertaking", "freight"):
        out.append(f"- *{group.replace('_', ' ')}*: " + "; ".join(m[group]))
    out += ["", "Coverage in the Flatland contest:", ""]
    out += [f"- {k.replace('_', ' ')}: {v}" for k, v in m["contest_coverage"].items()]
    out += [
        "",
        "## How the lab uses these rules",
        "",
        "- `rule-ril420` applies P1–P6 to the decision card (hold/wait for an oncoming train that goes",
        "  first; ignore a train broken down longer than the hold cap, the kind of justified exception",
        "  that S1 admits). P4 is read as ranking 'Fast' freight above other *freight* only; against a",
        "  regional passenger train both are equal-standing and P6 (speed) decides.",
        "- `jev-ril`, `deepseek-*-ril` receive the paraphrased rules (`rules_text()`) with the card.",
        "- The rollout oracle scores O0 (unweighted delay of all trains). Mixed-traffic game sets also",
        "  score a priority-weighted delay (weights in `SERVICE_WEIGHT`, an assumption of this study,",
        "  not part of Ril 420) to measure the price of rule adherence under both views.",
        "- C4/C5 (dispatchers decide, signallers execute; no access to route setting) are the",
        "  regulatory counterpart of the lab's interlocking/dispatcher split (ADR-0003).",
        "",
    ]
    return "\n".join(out)
