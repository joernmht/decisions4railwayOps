"""Deadlock detection by wait-for cycles.

Flatland 4.2.6's built-in deadlock counter is always zero: ``RailEnv.record_timestep`` tests
``agent.position in motion_check.deadlocked``, but that set holds agent *handles*, not positions
(verified 2026-09-28). We therefore detect deadlocks ourselves.

An on-map train A *waits for* train B if the next cell A intends to enter is occupied by B. Trains
cannot reverse on open track, so a cycle in this wait-for graph never resolves; its members and
every train transitively waiting on them are deadlocked. We use the *intended* next cell (from the
train's route), not the action actually sent: a safety layer that answers a real deadlock with
STOP on both trains would otherwise hide it. Adapted from the research census script (2026-09-28).
"""

from __future__ import annotations

from collections.abc import Mapping

__all__ = ["deadlocked", "wait_for_deadlocks"]

Cell = tuple[int, int]


def deadlocked(occupant: Mapping[Cell, int], intended: Mapping[int, Cell]) -> set[int]:
    """Handles in, or transitively waiting on, a cycle of the wait-for graph.

    Args:
        occupant: cell -> handle of the train occupying it (on-map trains only).
        intended: handle -> next cell the train intends to enter (active on-map trains only).
    """
    waits: dict[int, int] = {}
    for handle, cell in intended.items():
        other = occupant.get(cell)
        if other is not None and other != handle:
            waits[handle] = other

    dead: set[int] = set()
    for start in sorted(waits):
        seen: list[int] = []
        x = start
        while x in waits and x not in seen:
            seen.append(x)
            x = waits[x]
        if x in seen:
            dead.update(seen[seen.index(x) :])
    changed = True
    while changed:
        changed = False
        for a, b in sorted(waits.items()):
            if b in dead and a not in dead:
                dead.add(a)
                changed = True
    return dead


def wait_for_deadlocks(env, actions: Mapping[int, int]) -> set[int]:  # env: flatland RailEnv
    """Action-based variant: the intended cell is where each train's action would take it.

    Trains sent STOP are ignored, so this misses deadlocks a safety layer is already holding;
    prefer :func:`deadlocked` with route-based intentions where routes are known.
    """
    from flatland.envs.rail_env_action import RailEnvActions
    from flatland.envs.step_utils.states import TrainState

    stop = RailEnvActions.STOP_MOVING.value
    occupant = {tuple(a.position): a.handle for a in env.agents if a.position is not None}
    intended: dict[int, Cell] = {}
    for a in env.agents:
        if a.position is None or a.state == TrainState.DONE:
            continue
        act = int(getattr(actions.get(a.handle, 0), "value", actions.get(a.handle, 0)))
        if act == stop:
            continue
        nxt = env.rail.apply_action_independent(
            RailEnvActions.from_value(act), (tuple(a.position), int(a.direction))
        )
        if nxt is None:
            continue
        (cell, _heading), _ = nxt
        intended[a.handle] = cell
    return deadlocked(occupant, intended)
