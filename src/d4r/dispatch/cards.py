"""Decision cards: the one shared description of a dispatching decision.

Every engine — Jev, a generative LLM, an LLM that writes an LP, a rule — sees the *same* card for
the same decision. The card holds code-computed facts (distances, slack, who is oncoming) and the
closed set of safe options. Numbers are computed here, never by the engine; engines only select.

Two renderings exist (see :func:`render_state`):

- ``bucketed`` (default): numbers are mapped to named buckets ("close", "tight", "long repair").
  Jev's documentation says it is unreliable with raw numbers, counting and time comparison, so
  the default card gives every engine the same named buckets.
- ``raw``: exact integers instead of buckets, for the numeracy / serialisation ablation (P2-H6).
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict

__all__ = [
    "DecisionCard",
    "Option",
    "OtherTrain",
    "TrainFacts",
    "bucket_distance",
    "bucket_remaining",
    "bucket_repair",
    "bucket_slack",
    "render_state",
]

DecisionKind = Literal["depart", "meet", "route"]


def bucket_distance(cells: int) -> str:
    """Distance along the route, in cells (one cell ≈ one block section)."""
    if cells <= 2:
        return "adjacent (1-2 cells)"
    if cells <= 6:
        return "close (3-6 cells)"
    if cells <= 15:
        return "near (7-15 cells)"
    return "far (more than 15 cells)"


def bucket_slack(steps: int) -> str:
    """Schedule slack: latest arrival minus (now + remaining running time)."""
    if steps < 0:
        return "late (will miss its latest arrival even without further waiting)"
    if steps <= 4:
        return "tight (0-4 steps of reserve)"
    if steps <= 14:
        return "on time (5-14 steps of reserve)"
    return "ahead of schedule (15 or more steps of reserve)"


def bucket_remaining(cells: int) -> str:
    """Remaining running distance to the destination."""
    if cells < 10:
        return "short (under 10 cells)"
    if cells < 30:
        return "medium (10-29 cells)"
    return "long (30 or more cells)"


def bucket_repair(steps: int) -> str:
    """Remaining breakdown duration."""
    if steps <= 0:
        return "not broken down"
    if steps <= 10:
        return "broken down, short repair left (1-10 steps)"
    return "broken down, long repair left (more than 10 steps)"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Option(_Frozen):
    """One safe action the dispatcher may choose."""

    id: str
    description: str


class TrainFacts(_Frozen):
    """Facts about the train the decision is for."""

    id: str
    status: str
    slack_steps: int
    remaining_cells: int
    trains_queued_behind: int
    repair_steps_left: int = 0


class OtherTrain(_Frozen):
    """A train relevant to the decision (currently: oncoming on the route)."""

    id: str
    relation: str
    distance_cells: int
    status: str
    slack_steps: int
    remaining_cells: int
    repair_steps_left: int
    shared_track_cells: int
    trains_queued_behind: int


class DecisionCard(_Frozen):
    """A single dispatching decision, as presented to every engine."""

    card_id: str
    scenario: str
    seed: int
    step: int
    max_steps: int
    kind: DecisionKind
    train: TrainFacts
    others: tuple[OtherTrain, ...] = ()
    options: tuple[Option, ...]
    default_option: str
    context: str = ""

    @staticmethod
    def make_id(scenario: str, seed: int, step: int, handle: int, kind: str, salt: str = "") -> str:
        """Stable id: the same situation in the same run always gets the same id."""
        raw = f"{scenario}|{seed}|{step}|{handle}|{kind}|{salt}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def option_ids(self) -> list[str]:
        """Option ids in presentation order."""
        return [o.id for o in self.options]


_KIND_TEXT = {
    "depart": (
        "The train is ready to depart from its origin station. Other trains are already on, or"
        " about to use, its route in the opposite direction."
    ),
    "meet": (
        "The train is about to meet one or more oncoming trains on its route. Track between them is"
        " partly single track, so one of the trains must wait for the other at a place where they"
        " can pass. The interlocking guarantees safety either way; the question is who goes first."
    ),
    "route": "The train is at a facing switch and can take more than one route to its destination.",
}


def render_state(card: DecisionCard, variant: Literal["bucketed", "raw"] = "bucketed") -> dict:
    """The JSON state an engine receives for ``card``.

    Both variants carry the same information; ``bucketed`` replaces exact numbers by named ranges.
    """
    bucketed = variant == "bucketed"

    def dist(c: int) -> object:
        return bucket_distance(c) if bucketed else c

    def slack(s: int) -> object:
        return bucket_slack(s) if bucketed else s

    def rem(c: int) -> object:
        return bucket_remaining(c) if bucketed else c

    def rep(s: int) -> object:
        return bucket_repair(s) if bucketed else s

    t = card.train
    state: dict = {
        "situation": _KIND_TEXT[card.kind],
        "objective": (
            "Minimise the total delay of all trains at their destinations. A train that never"
            " departs or does not arrive before the end of operations counts as heavily delayed."
        ),
        "train": {
            "id": t.id,
            "status": t.status,
            "schedule_reserve": slack(t.slack_steps),
            "remaining_distance": rem(t.remaining_cells),
            "trains_queued_behind_it": t.trains_queued_behind,
        },
    }
    if t.repair_steps_left:
        state["train"]["breakdown"] = rep(t.repair_steps_left)
    if card.others:
        state["oncoming_trains"] = [
            {
                "id": o.id,
                "distance_ahead_on_route": dist(o.distance_cells),
                "status": o.status,
                "schedule_reserve": slack(o.slack_steps),
                "remaining_distance": rem(o.remaining_cells),
                "breakdown": rep(o.repair_steps_left),
                "shared_track_ahead": dist(o.shared_track_cells),
                "trains_queued_behind_it": o.trains_queued_behind,
            }
            for o in card.others
        ]
    if card.context:
        state["context"] = card.context
    state["time"] = {
        "now_step": card.step if not bucketed else None,
        "share_of_operating_period_elapsed": _share_bucket(card.step, card.max_steps)
        if bucketed
        else round(card.step / max(1, card.max_steps), 2),
    }
    if bucketed:
        state["time"].pop("now_step")
    return state


def _share_bucket(step: int, max_steps: int) -> str:
    share = step / max(1, max_steps)
    if share < 0.33:
        return "early in the operating period"
    if share < 0.66:
        return "middle of the operating period"
    return "late in the operating period (little time left)"


def card_json(card: DecisionCard, variant: Literal["bucketed", "raw"] = "bucketed") -> str:
    """Canonical JSON of the rendered state (sorted keys) — used for caching and prompts."""
    return json.dumps(render_state(card, variant), sort_keys=True, ensure_ascii=False)
