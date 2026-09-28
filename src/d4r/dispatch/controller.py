"""Dispatch controller: the DLA interlocking moves trains, an engine decides at decision points.

Division of labour, mirroring a real control centre:

- **Interlocking (code, DLA).** Every step the vendored deadlock-avoidance policy computes, for
  each train, whether moving along its route is safe and which action does it. Safety is never
  delegated to an engine.
- **Dispatcher (engine).** At *decision points* the engine chooses among safe options:

  - ``depart`` — a train is ready to enter the network while oncoming trains use its route:
    ``DEPART_NOW`` (default) or ``WAIT`` a few steps (retiming).
  - ``meet`` — a moving train is about to meet oncoming trains on partly single track and is at a
    place where it could let them pass: ``PROCEED`` (default) or ``HOLD`` until they have passed
    (reordering; in railway terms, choosing the crossing point).

  The default option is exactly what DLA alone would do, so the ``dla-default`` engine reproduces
  the plain DLA baseline.

A step is split into :meth:`Controller.prepare` (interlocking + detection) and
:meth:`Controller.commit` (apply decisions, advance Flatland). Between the two, :meth:`snapshot`
can fork the whole state; the rollout oracle uses that to label decisions by simulation.
"""

from __future__ import annotations

import copy
import time
import warnings
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from d4r.dispatch.cards import DecisionCard, Option, OtherTrain, TrainFacts
from d4r.engines.base import Decision, DefaultEngine, Engine
from d4r.sim.deadlock import deadlocked
from d4r.sim.scenario import Scenario, make_env

__all__ = ["Controller", "DecisionPoint", "DecisionRecord", "EpisodeResult", "run_episode"]

Cell = tuple[int, int]

#: Look-ahead (cells) for the hold-safety check and for counting queued followers.
_SAFETY_LOOKAHEAD = 20
_QUEUE_LOOKAHEAD = 6


@dataclass(frozen=True)
class DecisionPoint:
    """A detected decision: the card plus what the controller needs to apply it."""

    card: DecisionCard
    handle: int
    awaited: frozenset[int] = frozenset()


@dataclass
class Hold:
    """A train held by a dispatcher decision."""

    until_step: int
    kind: str
    awaited: frozenset[int] = frozenset()


@dataclass
class DecisionRecord:
    """One decision as it happened in an episode."""

    card: DecisionCard
    decision: Decision
    applied_option: str
    fallback: bool

    def to_dict(self) -> dict[str, Any]:
        d = self.decision
        return {
            "card": self.card.model_dump(mode="json"),
            "option": d.option_id,
            "applied_option": self.applied_option,
            "fallback": self.fallback,
            "probabilities": d.probabilities,
            "latency_ms": d.latency_ms,
            "input_tokens": d.input_tokens,
            "output_tokens": d.output_tokens,
            "cost_usd": d.cost_usd,
            "model": d.model,
            "error": d.error,
            "meta": d.meta,
        }


@dataclass
class EpisodeResult:
    """Outcome of one episode under one engine."""

    scenario: str
    seed: int
    engine: str
    steps: int
    max_steps: int
    n_trains: int
    arrived: int
    total_reward: float
    normalized_reward: float
    total_arrival_delay: int
    late_arrivals: int
    deadlocked_trains: int
    first_deadlock_step: int | None
    malfunctions: int
    decisions: dict[str, int] = field(default_factory=dict)
    options_chosen: dict[str, int] = field(default_factory=dict)
    fallbacks: int = 0
    engine_latency_ms_total: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    wall_s: float = 0.0

    @property
    def arrival_share(self) -> float:
        return self.arrived / max(1, self.n_trains)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["arrival_share"] = round(self.arrival_share, 4)
        return d


def _as_int(action: Any) -> int:
    """Flatland 4.2 ``RailEnvActions`` is a plain Enum; normalise to its int value."""
    return int(getattr(action, "value", action))


class Controller:
    """Runs one Flatland episode with DLA as interlocking and ``engine`` as dispatcher."""

    def __init__(
        self,
        scenario: Scenario,
        seed: int,
        engine: Engine | None = None,
        *,
        meet_hold_cap: int = 12,
        depart_wait: int = 5,
        max_depart_asks: int = 3,
        kinds: tuple[str, ...] = ("depart", "meet"),
        meet_trigger_cells: int = 15,
        env: Any = None,
    ) -> None:
        from d4r.sim._vendor.dla import DeadLockAvoidancePolicy

        self.scenario = scenario
        self.seed = seed
        self.engine: Engine = engine or DefaultEngine()
        self.meet_hold_cap = meet_hold_cap
        self.depart_wait = depart_wait
        self.max_depart_asks = max_depart_asks
        self.kinds = kinds
        self.meet_trigger_cells = meet_trigger_cells
        self.env = env if env is not None else make_env(scenario, seed)
        self.dla = DeadLockAvoidancePolicy()
        self.holds: dict[int, Hold] = {}
        self.asked: set[tuple] = set()
        self.depart_asks: Counter[int] = Counter()
        self.records: list[DecisionRecord] = []
        self.total_reward = 0.0
        self.first_dead: dict[int, int] = {}
        self.malfunctions = 0
        self._was_malfunctioning: dict[int, bool] = {a.handle: False for a in self.env.agents}
        self.done = False
        # set by prepare(), consumed by commit()
        self._acts: dict[int, int] | None = None
        self._points: list[DecisionPoint] = []

    # ------------------------------------------------------------------ helpers
    @property
    def step_no(self) -> int:
        return int(self.env._elapsed_steps)

    @property
    def max_steps(self) -> int:
        return int(self.env._max_episode_steps)

    def _path(self, h: int) -> tuple:
        p = self.dla._set_paths.get(h)
        return tuple(p) if p else ()

    def _remaining(self, h: int) -> int:
        a = self.env.agents[h]
        p = self._path(h)
        if not p:
            return 0
        return len(p) - 1 if a.position is not None else len(p)

    def _slack(self, h: int) -> int:
        a = self.env.agents[h]
        return int(a.latest_arrival) - self.step_no - self._remaining(h)

    def _repair(self, h: int) -> int:
        a = self.env.agents[h]
        return int(a.malfunction_handler.malfunction_down_counter)

    def _status(self, h: int) -> str:
        from flatland.envs.step_utils.states import TrainState

        a = self.env.agents[h]
        if a.malfunction_handler.in_malfunction:
            return "broken down"
        if a.state in (TrainState.READY_TO_DEPART, TrainState.WAITING):
            return "not yet on the network"
        if a.state == TrainState.STOPPED:
            return "stopped (waiting at a signal)"
        if a.state == TrainState.MOVING:
            return "moving"
        return a.state.name.lower()

    def _queued_behind(self, h: int) -> int:
        """Other on-map trains that will pass this train's current cell soon, in its direction."""
        a = self.env.agents[h]
        if a.position is None:
            return 0
        pos, heading = tuple(a.position), int(a.direction)
        n = 0
        for x in self.env.agents:
            if x.handle == h or x.position is None:
                continue
            for wp in self._path(x.handle)[1 : _QUEUE_LOOKAHEAD + 1]:
                if tuple(wp.position) == pos and int(wp.direction) == heading:
                    n += 1
                    break
        return n

    def _hold_is_safe(self, h: int) -> bool:
        """Holding ``h`` in place must not stand in the way of any train in another direction."""
        from flatland.envs.step_utils.states import TrainState

        a = self.env.agents[h]
        if a.position is None:
            return True  # off-map trains occupy nothing
        pos, heading = tuple(a.position), int(a.direction)
        for x in self.env.agents:
            if x.handle == h or x.state == TrainState.DONE:
                continue
            for wp in self._path(x.handle)[1 : _SAFETY_LOOKAHEAD + 1]:
                if tuple(wp.position) == pos and int(wp.direction) != heading:
                    return False
        return True

    def _nearest_oncoming(self, h: int, opp: frozenset[int]) -> int:
        """Cells along ``h``'s route to the nearest oncoming train."""
        cells = [tuple(wp.position) for wp in self._path(h)]
        best = len(cells)
        for o in opp:
            pos = self.env.agents[o].position
            if pos is not None and tuple(pos) in cells[1:]:
                best = min(best, cells.index(tuple(pos), 1))
        return best

    def _oncoming_facts(self, h: int, opp: frozenset[int]) -> tuple[OtherTrain, ...]:
        path = self._path(h)
        cells = [tuple(wp.position) for wp in path]
        out = []
        for o in sorted(opp):
            ao = self.env.agents[o]
            opos = tuple(ao.position) if ao.position is not None else None
            try:
                idx = cells.index(opos, 1)
            except ValueError:
                idx = len(cells)
            o_cells = {tuple(wp.position) for wp in self._path(o)}
            shared = sum(1 for c in cells[1:idx] if c in o_cells)
            out.append(
                OtherTrain(
                    id=f"T{o}",
                    relation="oncoming on this train's route",
                    distance_cells=idx,
                    status=self._status(o),
                    slack_steps=self._slack(o),
                    remaining_cells=self._remaining(o),
                    repair_steps_left=self._repair(o),
                    shared_track_cells=shared,
                    trains_queued_behind=self._queued_behind(o),
                )
            )
        return tuple(out)

    def _train_facts(self, h: int) -> TrainFacts:
        return TrainFacts(
            id=f"T{h}",
            status=self._status(h),
            slack_steps=self._slack(h),
            remaining_cells=self._remaining(h),
            trains_queued_behind=self._queued_behind(h),
            repair_steps_left=self._repair(h),
        )

    # ------------------------------------------------------------------ detection
    def _detect(self) -> list[DecisionPoint]:
        from flatland.envs.step_utils.states import TrainState

        points: list[DecisionPoint] = []
        can_move = self.dla.agent_can_move
        for a in self.env.agents:
            h = a.handle
            if h in self.holds or h not in can_move or a.state == TrainState.DONE:
                continue
            opp = frozenset(int(x) for x in self.dla.opp_agent_map.get(h, set()))
            if not opp:
                continue
            if a.malfunction_handler.in_malfunction:
                continue
            if (
                "depart" in self.kinds
                and a.position is None
                and a.state == TrainState.READY_TO_DEPART
            ):
                if self.depart_asks[h] >= self.max_depart_asks:
                    continue
                key = ("depart", h, self.depart_asks[h])
                if key in self.asked:
                    continue
                self.asked.add(key)
                self.depart_asks[h] += 1
                points.append(self._depart_point(h, opp))
            elif "meet" in self.kinds and a.position is not None:
                key = ("meet", h, opp)
                if key in self.asked or not self._hold_is_safe(h):
                    continue
                if self._nearest_oncoming(h, opp) > self.meet_trigger_cells:
                    continue
                self.asked.add(key)
                points.append(self._meet_point(h, opp))
        return points

    def _depart_point(self, h: int, opp: frozenset[int]) -> DecisionPoint:
        n = self.depart_asks[h]
        card = DecisionCard(
            card_id=DecisionCard.make_id(
                self.scenario.name, self.seed, self.step_no, h, "depart", str(n)
            ),
            scenario=self.scenario.name,
            seed=self.seed,
            step=self.step_no,
            max_steps=self.max_steps,
            kind="depart",
            train=self._train_facts(h),
            others=self._oncoming_facts(h, opp),
            options=(
                Option(id="DEPART_NOW", description=f"Release train T{h} into the network now."),
                Option(
                    id="WAIT",
                    description=(
                        f"Keep train T{h} at its origin for {self.depart_wait} more steps so the"
                        " oncoming traffic can clear, then decide again."
                    ),
                ),
            ),
            default_option="DEPART_NOW",
        )
        return DecisionPoint(card=card, handle=h, awaited=opp)

    def _meet_point(self, h: int, opp: frozenset[int]) -> DecisionPoint:
        names = ", ".join(f"T{o}" for o in sorted(opp))
        card = DecisionCard(
            card_id=DecisionCard.make_id(self.scenario.name, self.seed, self.step_no, h, "meet"),
            scenario=self.scenario.name,
            seed=self.seed,
            step=self.step_no,
            max_steps=self.max_steps,
            kind="meet",
            train=self._train_facts(h),
            others=self._oncoming_facts(h, opp),
            options=(
                Option(
                    id="PROCEED",
                    description=(
                        f"Let train T{h} continue now. The oncoming train(s) {names} will be stopped"
                        f" by the interlocking wherever they can let T{h} pass, if T{h} reaches the"
                        " shared single track first."
                    ),
                ),
                Option(
                    id="HOLD",
                    description=(
                        f"Hold train T{h} where it is (a place where {names} can pass it) until"
                        f" {names} have passed, for at most {self.meet_hold_cap} steps."
                    ),
                ),
            ),
            default_option="PROCEED",
        )
        return DecisionPoint(card=card, handle=h, awaited=opp)

    # ------------------------------------------------------------------ step
    def prepare(self) -> list[DecisionPoint]:
        """Run the interlocking for this step and detect the decision points."""
        n = self.env.get_num_agents()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            acts = self.dla.act_many(list(range(n)), [self.env] * n)
        self._acts = {h: _as_int(v) for h, v in acts.items()}
        self._points = self._detect()
        return self._points

    def decide(self, points: list[DecisionPoint]) -> list[Decision]:
        """Ask the engine; invalid or missing answers become the default option."""
        if not points:
            return []
        cards = [p.card for p in points]
        try:
            decisions = self.engine.decide(cards, ctx=self)
        except Exception as exc:  # an engine failure must not stop the railway
            decisions = [
                Decision(c.card_id, c.default_option, error=f"engine: {exc!r}") for c in cards
            ]
        by_id = {d.card_id: d for d in decisions}
        return [
            by_id.get(c.card_id, Decision(c.card_id, c.default_option, error="missing"))
            for c in cards
        ]

    def commit(self, decisions: list[Decision], record: bool = True) -> None:
        """Apply the decisions for the prepared step, enforce holds, advance Flatland."""
        from flatland.envs.rail_env_action import RailEnvActions
        from flatland.envs.step_utils.states import TrainState

        assert self._acts is not None, "prepare() must be called before commit()"
        acts = self._acts
        now = self.step_no
        for point, dec in zip(self._points, decisions, strict=True):
            valid = dec.option_id in point.card.option_ids()
            applied = dec.option_id if valid else point.card.default_option
            fallback = (not valid) or dec.error is not None
            if applied == "WAIT":
                self.holds[point.handle] = Hold(now + self.depart_wait, "depart")
            elif applied == "HOLD":
                self.holds[point.handle] = Hold(now + self.meet_hold_cap, "meet", point.awaited)
            if record:
                self.records.append(DecisionRecord(point.card, dec, applied, fallback))

        # enforce and release holds
        for h in sorted(self.holds):
            hold = self.holds[h]
            a = self.env.agents[h]
            release = now >= hold.until_step or a.state == TrainState.DONE
            if hold.kind == "meet" and not release:
                still = hold.awaited & {int(x) for x in self.dla.opp_agent_map.get(h, set())}
                release = not still or not self._hold_is_safe(h)
            if release:
                del self.holds[h]
                continue
            acts[h] = _as_int(
                RailEnvActions.STOP_MOVING if a.position is not None else RailEnvActions.DO_NOTHING
            )

        # deadlock census on route intentions (before moving)
        occupant = {tuple(a.position): a.handle for a in self.env.agents if a.position is not None}
        intended: dict[int, Cell] = {}
        for a in self.env.agents:
            if a.position is None or a.state == TrainState.DONE:
                continue
            p = self._path(a.handle)
            if len(p) > 1:
                intended[a.handle] = tuple(p[1].position)
        for h in deadlocked(occupant, intended):
            self.first_dead.setdefault(h, now)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, rewards, dones, _ = self.env.step(acts)
        self.total_reward += float(sum(rewards.values()))
        for a in self.env.agents:
            m = bool(a.malfunction_handler.in_malfunction)
            if m and not self._was_malfunctioning.get(a.handle, False):
                self.malfunctions += 1
            self._was_malfunctioning[a.handle] = m
        self._acts = None
        self._points = []
        self.done = bool(dones["__all__"])

    def step(self) -> None:
        """prepare → decide → commit."""
        points = self.prepare()
        self.commit(self.decide(points))

    # ------------------------------------------------------------------ forking
    def snapshot(self, engine: Engine | None = None) -> Controller:
        """Deep copy of the full state mid-step (between prepare and commit)."""
        from flatland.core.env_observation_builder import DummyObservationBuilder
        from flatland.envs.rail_env import RailEnv

        env2 = RailEnv(
            width=self.env.width,
            height=self.env.height,
            obs_builder_object=DummyObservationBuilder(),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            env2.clone_from(self.env)
        env_ref, self.dla.rail_env = self.dla.rail_env, None
        engine_ref = self.engine
        records_ref = self.records
        self.engine = None  # type: ignore[assignment]
        self.records = []
        try:
            env_self = self.env
            self.env = None
            clone = copy.deepcopy(self)
        finally:
            self.env = env_self
            self.dla.rail_env = env_ref
            self.engine = engine_ref
            self.records = records_ref
        clone.env = env2
        clone.dla.rail_env = env2
        clone.engine = engine or DefaultEngine()
        return clone

    def run_to_end(self) -> None:
        """Continue the episode with the current engine until Flatland reports done."""
        while not self.done:
            self.step()

    # ------------------------------------------------------------------ results
    def result(self, wall_s: float = 0.0) -> EpisodeResult:
        from flatland.envs.step_utils.states import TrainState

        done = [a for a in self.env.agents if a.state == TrainState.DONE]
        delays = [max(0, int(a.arrival_time) - int(a.latest_arrival)) for a in done]
        n = self.env.get_num_agents()
        norm = self.total_reward / (self.max_steps * n) + 1.0
        kinds: Counter[str] = Counter(r.card.kind for r in self.records)
        chosen: Counter[str] = Counter(f"{r.card.kind}:{r.applied_option}" for r in self.records)
        return EpisodeResult(
            scenario=self.scenario.name,
            seed=self.seed,
            engine=getattr(self.engine, "name", "?"),
            steps=self.step_no,
            max_steps=self.max_steps,
            n_trains=n,
            arrived=len(done),
            total_reward=round(self.total_reward, 3),
            normalized_reward=round(norm, 4),
            total_arrival_delay=int(sum(delays)),
            late_arrivals=sum(1 for d in delays if d > 0),
            deadlocked_trains=len(self.first_dead),
            first_deadlock_step=min(self.first_dead.values()) if self.first_dead else None,
            malfunctions=self.malfunctions,
            decisions=dict(kinds),
            options_chosen=dict(chosen),
            fallbacks=sum(1 for r in self.records if r.fallback),
            engine_latency_ms_total=round(sum(r.decision.latency_ms for r in self.records), 1),
            input_tokens=sum(r.decision.input_tokens for r in self.records),
            output_tokens=sum(r.decision.output_tokens for r in self.records),
            cost_usd=round(sum(r.decision.cost_usd for r in self.records), 6),
            wall_s=round(wall_s, 2),
        )


def run_episode(
    scenario: Scenario, seed: int, engine: Engine | None = None, **kwargs: Any
) -> tuple[EpisodeResult, list[DecisionRecord]]:
    """Run one full episode and return its result and decision records."""
    t0 = time.perf_counter()
    ctl = Controller(scenario, seed, engine, **kwargs)
    ctl.run_to_end()
    return ctl.result(time.perf_counter() - t0), ctl.records


def option_values(
    ctl: Controller, point: DecisionPoint, others: Mapping[str, str] | None = None
) -> dict[str, float]:
    """Rollout value of each option of ``point``: total reward from now to the end of the episode.

    Every other decision in this step takes its default option (or ``others[card_id]``), and the
    rest of the episode runs under the DLA default. Deterministic: Flatland breakdowns depend only
    on the seeded environment RNG, which the clone copies.
    """
    values: dict[str, float] = {}
    for opt in point.card.option_ids():
        fork = ctl.snapshot()
        base = fork.total_reward
        decisions = []
        for p in ctl._points:
            if p.card.card_id == point.card.card_id:
                decisions.append(Decision(p.card.card_id, opt))
            else:
                choice = (others or {}).get(p.card.card_id, p.card.default_option)
                decisions.append(Decision(p.card.card_id, choice))
        fork._points = list(ctl._points)
        fork._acts = dict(ctl._acts or {})
        fork.commit(decisions, record=False)
        fork.run_to_end()
        values[opt] = round(fork.total_reward - base, 3)
    return values
