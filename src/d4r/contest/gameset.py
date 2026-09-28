"""Game set: decision cards collected from DLA episodes and labelled by rollout.

The behaviour policy is the plain interlocking (``dla-default``), so the cards show the situations
that arise in ordinary operation. At each decision point the rollout oracle values every option
(total Flatland reward from that step to the end of the episode under DLA). The labelled set is the
offline benchmark: every engine answers the same cards, and we score accuracy, regret and
calibration without the compounding of closed-loop dynamics.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from d4r.dispatch.cards import DecisionCard
from d4r.dispatch.controller import Controller, option_values
from d4r.engines.base import Decision
from d4r.sim.scenario import Scenario

__all__ = ["LabelledCard", "build_gameset", "load_gameset"]


class _Labeller:
    """Behaves like ``dla-default`` but records the rollout value of every option."""

    name = "dla-default+labels"

    def __init__(self) -> None:
        self.labels: dict[str, dict[str, float]] = {}

    def decide(self, cards: Sequence[DecisionCard], ctx: Any = None) -> list[Decision]:
        points = {p.card.card_id: p for p in ctx._points}
        out = []
        for c in cards:
            self.labels[c.card_id] = option_values(ctx, points[c.card_id])
            out.append(Decision(c.card_id, c.default_option))
        return out


class LabelledCard(dict):  # type: ignore[type-arg]
    """A JSON record: ``card`` (DecisionCard dict), ``values``, ``best``, ``spread``."""

    @property
    def card(self) -> DecisionCard:
        return DecisionCard.model_validate(self["card"])


def build_gameset(
    scenario: Scenario, seeds: Iterable[int], out: Path, **controller_kwargs: Any
) -> list[LabelledCard]:
    """Run DLA episodes for ``seeds`` and write one labelled card per line to ``out``."""
    out.parent.mkdir(parents=True, exist_ok=True)
    records: list[LabelledCard] = []
    with out.open("w", encoding="utf-8") as f:
        for seed in seeds:
            lab = _Labeller()
            ctl = Controller(scenario, seed, lab, **controller_kwargs)  # type: ignore[arg-type]
            ctl.run_to_end()
            episode = ctl.result().to_dict()
            for rec in ctl.records:
                values = lab.labels[rec.card.card_id]
                best_v = max(values.values())
                best = sorted(o for o, v in values.items() if v == best_v)
                row = LabelledCard(
                    card=rec.card.model_dump(mode="json"),
                    values=values,
                    best=best,
                    spread=round(best_v - min(values.values()), 3),
                    episode={
                        k: episode[k]
                        for k in ("arrival_share", "total_reward", "deadlocked_trains")
                    },
                )
                records.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return records


def load_gameset(path: Path) -> list[LabelledCard]:
    """Read a game set written by :func:`build_gameset`."""
    with path.open(encoding="utf-8") as f:
        return [LabelledCard(json.loads(line)) for line in f if line.strip()]
