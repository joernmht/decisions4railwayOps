# ruff: noqa
# mypy: ignore-errors
# Vendored from flatland-association/flatland-baselines @ c1733cbae8f9abc6c1e73fa16ec9119872abd292
# (flatland_baselines/deadlock_avoidance_heuristic/policy/set_path_policy.py), MIT License, (c) 2025 Flatland
# Association; see LICENSE in this directory. Local changes: matplotlib imported lazily (debug
# plotting only) and package-relative imports. The upstream package pins flatland-rl 4.3.0; this
# copy runs on 4.2.6, which the ECML 2026 challenge used.
import warnings
from collections import defaultdict
from functools import lru_cache
from typing import List, Dict, Tuple

import numpy as np

from flatland.envs.agent_chains import AgentHandle
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_env_policy import RailEnvPolicy
from flatland.envs.rail_env_shortest_paths import get_k_shortest_paths
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.step_utils.states import TrainState


class SetPathPolicy(RailEnvPolicy[RailEnv, RailEnv, RailEnvActions]):
    """
    Works with `FullEnvObservation` only.

    Policy where agents follow a set path.
    """

    def __init__(self,
                 k_shortest_path_cutoff: int = None,
                 use_always_first_strategy: int = None,
                 verbose: bool = False,
                 ):
        """

        Parameters
        ----------
        k_shortest_path_cutoff : Optional[int]
            Global cutoff for shortest path finding. Use with care as it can cause agents to have no set path at all.
        use_always_first_strategy : Optional[int]
            If not set or zero, ignore intermediate stops altogether. Otherwise, use shortest path between first option at all intermediate stops.
        verbose : bool
        """
        super().__init__()
        self._set_paths: Dict[AgentHandle, Tuple[Waypoint]] = {}
        self.k_shortest_path_cutoff = k_shortest_path_cutoff
        self.use_alternative_at_first_intermediate_and_then_always_first_strategy = use_always_first_strategy
        self.verbose = verbose

    def _act(self, env: RailEnv, agent: EnvAgent):
        if agent.current_configuration is None:
            return RailEnvActions.MOVE_FORWARD

        if len(self._set_paths[agent.handle]) == 0:
            return RailEnvActions.DO_NOTHING

        for a in {RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_RIGHT}:
            new_cell_valid, (new_position, new_direction), transition_valid, preprocessed_action = env.rail.check_action_on_agent(
                RailEnvActions.from_value(a), agent.current_configuration
            )
            if new_cell_valid and transition_valid and (
                    new_position == self._set_paths[agent.handle][1].position and new_direction == self._set_paths[agent.handle][1].direction):
                return a
        raise Exception("Invalid state")

    def act_many(self, handles: List[int], observations: List[RailEnv], **kwargs):
        actions = {}
        for handle, env in zip(handles, observations):
            agent = env.agents[handle]
            self._update_agent(agent, env)
            actions[handle] = self._act(env, agent)
        return actions

    def _update_agent(self, agent: EnvAgent, env: RailEnv):
        """
        Build `_shortest_paths`.
        """
        if agent.state == TrainState.DONE:
            self._set_paths.pop(agent.handle, None)
            return

        if agent.handle not in self._set_paths:
            if self.use_alternative_at_first_intermediate_and_then_always_first_strategy is not None and self.use_alternative_at_first_intermediate_and_then_always_first_strategy > 0:
                always_first_waypoint = _always_first_waypoint_from_flexible_groups(agent.waypoints)
                if self.verbose:
                    print(f"get path for agent {agent.handle} using always-first strategy on {agent.waypoints}")
                if self.audit is not None:
                    self.audit.append({"env_time": self.rail_env._elapsed_steps, "agent_id": agent.handle, "k": "audit",
                                       "v": f"get path for agent {agent.handle} using always-first strategy on {agent.waypoints}"})
                self._set_paths[agent.handle] = self._shortest_path_from_non_flexible_waypoints(always_first_waypoint, env.rail,
                                                                                                debug_label=f"Agent {agent.handle}")
            else:
                if self.verbose:
                    print(f"get path for agent {agent.handle} ignoring intermediate stops on {agent.waypoints}")
                if self.audit is not None:
                    self.audit.append({"env_time": self.rail_env._elapsed_steps, "agent_id": agent.handle, "k": "audit",
                                       "v": f"get path for agent {agent.handle} ignoring intermediate stops on {agent.waypoints}"})
                self._set_paths[agent.handle] = self._shortest_path_from_non_flexible_waypoints(
                    _always_first_waypoint_from_flexible_groups([agent.waypoints[0], agent.waypoints[-1]]), env.rail,
                    debug_label=f"Agent {agent.handle}")
        if self._set_paths[agent.handle] is None:
            # loopy path
            return
        if agent.current_configuration is None:
            # not on map
            return

        position = agent.current_configuration[0]
        while len(self._set_paths[agent.handle]) > 0 and self._set_paths[agent.handle][0].position != position:
            self._set_paths[agent.handle] = self._set_paths[agent.handle][1:]
        assert self._set_paths[agent.handle][0].position == position

    def _shortest_path_from_non_flexible_waypoints(self, waypoint_groups: List[List[Waypoint]], rail: RailGridTransitionMap, debug_segments: bool = False,
                                                   debug_label: str = ""):
        """
        Computes the shortest path to path built by routing the shortest path between non-flexible waypoints; only target may have flexibility.
        """
        p: List[Waypoint] = []
        for g1, g2 in zip(waypoint_groups, waypoint_groups[1:]):
            # non-target
            assert len(g1) == 1
            p1 = g1[0]
            if len(p) > 0:
                assert p[-1] == p1, (p[-1], p1)

            arrival_directions = {wp.direction for wp in g2}
            target_direction = next(iter(arrival_directions)) if len(arrival_directions) == 1 else None

            path_segment_candidates: List[Tuple[Waypoint]] = _get_k_shortest_paths(None, p1.position, p1.direction, g2[0].position, rail=rail,
                                                                                   target_direction=target_direction, cutoff=self.k_shortest_path_cutoff)
            assert len(path_segment_candidates) > 0, \
                f"[{debug_label}] Not found next path from {p1} to any of {g2}. Either not connected or no path respecting k_shortest_path_cutoff={self.k_shortest_path_cutoff}."
            next_path_segment = path_segment_candidates[0]
            assert g2[0].position == next_path_segment[-1].position
            assert len(set(next_path_segment)) == len(next_path_segment)
            if None not in arrival_directions:
                assert next_path_segment[-1].direction in arrival_directions, \
                    f"[{debug_label}] Not found next path from {p1} to any of {g2}. Either not connected or no path respecting k_shortest_path_cutoff={self.k_shortest_path_cutoff}."
            if len(p) > 0:
                p += next_path_segment[1:]
            else:
                p += next_path_segment
            if debug_segments:
                print(f"Segment {p1} {g2} has len {len(next_path_segment)}")
            if debug_segments:
                cells = [wp.position for wp in next_path_segment]
                height = max([r for (r, c) in cells]) + 1
                width = max([c for (r, c) in cells]) + 1
                data = np.zeros(shape=(height, width))
                for cell in cells:
                    data[cell] = 1
                import matplotlib.pyplot as plt; plt.imshow(data)
                import matplotlib.pyplot as plt; plt.show()
        if len(set(p)) != len(p):
            # TODO dla currently cannot handle this - generalize dla implementation?
            counts = defaultdict(lambda: 0)
            for wp in p:
                counts[wp] += 1
            duplicates = {wp for wp, count in counts.items() if count > 1}

            warnings.warn(f"[{debug_label}] Found loopy line {waypoint_groups} with duplicates {duplicates} in path {p}")
            if self.audit is not None:
                self.audit.append({"env_time": self.rail_env._elapsed_steps, "agent_id": debug_label, "k": "audit",
                                   "v": f"[{debug_label}] Found loopy line {waypoint_groups} with duplicates {duplicates} in path {p}"})
            return None
        return p


@lru_cache
def _get_k_shortest_paths(*args, **kwargs):
    return get_k_shortest_paths(*args, **kwargs)


def _always_first_waypoint_from_flexible_groups(waypoint_groups: List[List[Waypoint]]) -> List[List[Waypoint]]:
    """
    Reduces a list of waypoint-alternatives groups to a single, non-flexible path plan for
    `_shortest_path_from_non_flexible_waypoints`: every intermediate group is narrowed to its first
    alternative (arbitrary but stable), while the *final* group is passed through in
    full.
    """
    if len(waypoint_groups) == 0:
        return []
    return [[pp[0]] for pp in waypoint_groups[:-1]] + [waypoint_groups[-1]]
