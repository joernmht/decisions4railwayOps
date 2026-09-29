"""Scenario specification and a deterministic Flatland environment factory.

A scenario is fully determined by a :class:`Scenario` (map size, trains, cities, breakdown
calibration) plus an integer seed. The same (scenario, seed) always produces the same rail network,
timetable and breakdown schedule, whatever the dispatching engine does, because Flatland draws
breakdowns per (train, step) from the seeded environment RNG. That makes paired comparisons across
engines possible.

Defaults follow the Flatland-3/ECML calibration found in the research census (2026-09-28):
breakdown interval 540 steps (rate ≈ 0.00185), duration 20–50 steps, speed 1.0, three cities.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

__all__ = ["PRESETS", "Scenario", "assign_services", "make_env"]


@dataclass(frozen=True)
class Scenario:
    """A family of Flatland instances; combine with a seed to get one instance."""

    name: str
    width: int = 30
    height: int = 30
    n_trains: int = 10
    n_cities: int = 3
    max_rails_between_cities: int = 2
    max_rail_pairs_in_city: int = 2
    malfunction_rate: float = 1.0 / 540.0
    malfunction_min: int = 20
    malfunction_max: int = 50
    #: Mixed traffic: trains get Flatland speeds and Ril 420 service classes (see d4r.rules).
    mixed: bool = False

    def to_dict(self) -> dict[str, object]:
        """Plain-dict form for run records."""
        return asdict(self)


#: Scenario presets. A = development, B = main contest, C = stretch (see docs/design/contest.md).
PRESETS: dict[str, Scenario] = {
    "A": Scenario(name="A", n_trains=7),
    "B": Scenario(name="B", n_trains=10),
    "C": Scenario(name="C", width=40, height=40, n_trains=10, n_cities=4),
    # Mixed traffic (service classes + speeds) for the Ril 420 rule study (P2-H9).
    "M": Scenario(name="M", n_trains=10, mixed=True),
    # Tiny scenario for the offline test suite.
    "T": Scenario(name="T", width=25, height=25, n_trains=3, n_cities=2, malfunction_rate=0.0),
}


#: Share of trains per Flatland speed in mixed-traffic scenarios (1 = one cell per step).
MIXED_SPEEDS: dict[float, float] = {1.0: 0.25, 0.5: 0.55, 1.0 / 3.0: 0.20}
#: Service classes drawn for speed-0.5 trains (weights), see d4r.rules.SERVICE_SPEED.
_HALF_SPEED_SERVICES = (
    ("passenger_other", 0.35),
    ("freight_express", 0.10),
    ("freight_fast", 0.10),
)


def assign_services(env, seed: int) -> dict[int, str]:  # env: flatland RailEnv
    """Deterministic service class per train, consistent with the train's Flatland speed.

    Speed 1 -> long-distance passenger Express; speed 1/3 -> standard freight; speed 1/2 -> regional
    passenger, freight Express or freight Fast (seeded draw per train). Urgent relief trains are not
    generated (they would be injected by a disruption scenario).
    """
    import numpy as np

    rng = np.random.default_rng(10_000 + seed)
    names = [n for n, _ in _HALF_SPEED_SERVICES]
    w = np.array([p for _, p in _HALF_SPEED_SERVICES])
    out: dict[int, str] = {}
    for a in sorted(env.agents, key=lambda x: x.handle):
        speed = float(a.speed_counter.max_speed)
        draw = int(rng.choice(len(names), p=w / w.sum()))  # always draw: stable across speeds
        if speed >= 0.99:
            out[a.handle] = "passenger_express"
        elif speed <= 0.34:
            out[a.handle] = "freight_other"
        else:
            out[a.handle] = names[draw]
    return out


def make_env(scenario: Scenario, seed: int):  # -> flatland RailEnv (untyped upstream)
    """Build and reset the Flatland ``RailEnv`` for ``(scenario, seed)``.

    Flatland imports stay local so importing :mod:`d4r` is cheap.
    """
    from flatland.core.env_observation_builder import DummyObservationBuilder
    from flatland.envs.line_generators import sparse_line_generator
    from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen
    from flatland.envs.rail_env import RailEnv
    from flatland.envs.rail_generators import sparse_rail_generator

    malfunctions = ParamMalfunctionGen(
        MalfunctionParameters(
            malfunction_rate=scenario.malfunction_rate,
            min_duration=scenario.malfunction_min,
            max_duration=scenario.malfunction_max,
        )
    )
    env = RailEnv(
        width=scenario.width,
        height=scenario.height,
        rail_generator=sparse_rail_generator(
            max_num_cities=scenario.n_cities,
            max_rails_between_cities=scenario.max_rails_between_cities,
            max_rail_pairs_in_city=scenario.max_rail_pairs_in_city,
            seed=seed,
        ),
        # Called without a seed on purpose: passing one only triggers a noisy upstream warning;
        # the line generator draws from the env RNG seeded below.
        line_generator=sparse_line_generator(
            speed_ratio_map=MIXED_SPEEDS if scenario.mixed else None
        ),
        number_of_agents=scenario.n_trains,
        malfunction_generator=malfunctions,
        obs_builder_object=DummyObservationBuilder(),
        random_seed=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env.reset(random_seed=seed)
    return env
