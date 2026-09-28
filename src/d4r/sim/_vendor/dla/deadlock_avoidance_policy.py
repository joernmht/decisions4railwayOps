# ruff: noqa
# mypy: ignore-errors
# Vendored from flatland-association/flatland-baselines @ c1733cbae8f9abc6c1e73fa16ec9119872abd292
# (flatland_baselines/deadlock_avoidance_heuristic/policy/deadlock_avoidance_policy.py), MIT License, (c) 2025 Flatland
# Association; see LICENSE in this directory. Local changes: matplotlib imported lazily (debug
# plotting only) and package-relative imports. The upstream package pins flatland-rl 4.3.0; this
# copy runs on 4.2.6, which the ECML 2026 challenge used.
import copy
import os
import warnings
from collections import defaultdict
from functools import lru_cache
from typing import List, Any, Dict, Set, Tuple, Optional

import numpy as np

from flatland.core.env_observation_builder import AgentHandle
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.fast_methods import fast_count_nonzero
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.step_utils.states import TrainState
from .set_path_policy import SetPathPolicy, _get_k_shortest_paths, \
    _always_first_waypoint_from_flexible_groups

# activate LRU caching
flatland_deadlock_avoidance_policy_lru_cache_functions = []


def _enable_flatland_deadlock_avoidance_policy_lru_cache(*args, **kwargs):
    def decorator(func):
        func = lru_cache(*args, **kwargs)(func)
        flatland_deadlock_avoidance_policy_lru_cache_functions.append(func)
        return func

    return decorator


def _send_flatland_deadlock_avoidance_policy_data_change_signal_to_reset_lru_cache():
    for func in flatland_deadlock_avoidance_policy_lru_cache_functions:
        func.cache_clear()


class DeadLockAvoidancePolicy(SetPathPolicy):
    def __init__(self,
                 min_free_cell: int = 1,
                 show_debug_plot: bool = False,
                 count_num_opp_agents_towards_min_free_cell: bool = True,
                 use_switches_heuristic: bool = True,
                 use_entering_prevention: bool = False,
                 use_alternative_at_first_intermediate_and_then_always_first_strategy: int = None,
                 drop_next_threshold: int = None,
                 k_shortest_path_cutoff: int = None,
                 seed: int = None,
                 verbose: bool = False,
                 audit: bool = False,
                 ):
        """

        Parameters
        ----------
        min_free_cell : int
            How many cells must be left empty ahead to avoid collisions.
        show_debug_plot : bool
            Show plots with set path and own path till first opposing agent for all agents.
        count_num_opp_agents_towards_min_free_cell : bool
            Subtract the number of other trains oncoming on my own path from the number of free cells ahead of me.
        use_switches_heuristic : bool
            Subtract the number of switches on my own path from the number of free cells ahead of me.
        use_entering_prevention : bool
            Prevent one of two agents entering at the same timestep if `min_free_cells` is not respected.
            No alternative paths, only set path considered for detection. Agents always enter with initial set path.
        use_alternative_at_first_intermediate_and_then_always_first_strategy : Optional[int]
            If set and non-zero, use `use_always_first_strategy` initially for initial set path.
            If entered and blocked by DLA, sample from `k` shortest paths for all options at the next intermediate stop.
        drop_next_threshold : Optional[int]
            When threshold of time steps blocked consecutively, drop the next intermediate stop for the agent. Done iteratively.
        k_shortest_path_cutoff : Optional[int]
            Global cutoff for shortest path finding. Use with care as it can cause agents to have no set path at all.
        seed : Optional[int]
            Seed for sampling of altneratives.
        verbose :bool
        audit :bool
        """
        super().__init__(
            k_shortest_path_cutoff=k_shortest_path_cutoff,
            use_always_first_strategy=use_alternative_at_first_intermediate_and_then_always_first_strategy is not None and use_alternative_at_first_intermediate_and_then_always_first_strategy > 0,
            verbose=verbose,
        )

        self.loss = 0
        self.show_debug_plot = show_debug_plot
        self.min_free_cell = min_free_cell
        self.count_num_opp_agents_towards_min_free_cell = count_num_opp_agents_towards_min_free_cell
        self.use_switches_heuristic = use_switches_heuristic
        self.use_entering_prevention = use_entering_prevention
        self.use_k_alternatives_at_first_intermediate_and_then_always_first_strategy = use_alternative_at_first_intermediate_and_then_always_first_strategy
        self.drop_next_threshold = drop_next_threshold
        self.k_shortest_path_cutoff = k_shortest_path_cutoff

        # will be injected from observation (`FullEnvObservation`)
        self.rail_env: Optional[RailEnv] = None

        # start_step (1): -1 if no agent, agent handle otherwise
        self.agent_positions = None

        # start_step (3): next (r,c,d) and action to get there; or no entry if train must not move
        self.agent_can_move: Dict[AgentHandle, Tuple[int, int, int, RailEnvActions]] = {}
        # start_step (2.3.2): set of oncoming agents
        self.opp_agent_map: Dict[AgentHandle, Set[AgentHandle]] = defaultdict(set)

        self.agent_waypoints_done: Dict[AgentHandle, Set[Waypoint]] = defaultdict(set)
        self.agent_waypoints_tried: Dict[AgentHandle, Set[str]] = defaultdict(set)

        self.closed = defaultdict(list)
        self.num_blocked = defaultdict(lambda: 0)
        self.alternatives = defaultdict(list)

        self.np_random = np.random.RandomState(seed)
        self.audit = None
        if audit is True:
            self.audit = []

    def _init_env(self, env: RailEnv):
        # _init_env: 1 if position is a switch, 0 otherwise
        self.switches = None
        if self.use_switches_heuristic:
            self.switches = np.zeros((self.rail_env.height, self.rail_env.width), dtype=int)
            for r in range(self.rail_env.height):
                for c in range(self.rail_env.width):
                    if self._is_switch_cell((r, c)):
                        self.switches[(r, c)] = 1

        # start_step (2.2): 1 if current shortest path (without current cell!), 0 otherwise
        self.full_shortest_distance_agent_map = np.zeros((self.rail_env.get_num_agents(),
                                                          self.rail_env.height,
                                                          self.rail_env.width),
                                                         dtype=int)
        # start_step (2.2): all positions on current shortest path (without current cell!)
        self.shortest_distance_positions_agent_map: Dict[AgentHandle, Set[Tuple[int, int]]] = defaultdict(set)
        # start_step (2.2): directions for all positions on current shortest path (without current cell!)
        self.shortest_distance_positions_directions_agent_map: Dict[AgentHandle, Dict[Tuple[int, int], Set[int]]] = defaultdict(lambda: defaultdict(set))
        # start_step (2.3.3): number of cells till first oncoming agent (without current cell!)
        self.shortest_distance_agent_len: Dict[AgentHandle, int] = defaultdict(lambda: 0)
        # start_step (2.3.3): 1 if current shortest path (without current cell!) before first oncoming train, 0 else.
        self.shortest_distance_agent_map = np.zeros((self.rail_env.get_num_agents(),
                                                     self.rail_env.height,
                                                     self.rail_env.width),
                                                    dtype=int)

    def act_many(self, handles: List[int], observations: List[Any], **kwargs) -> Dict[int, RailEnvActions]:
        assert isinstance(observations[0], RailEnv)
        if self.rail_env is None:
            self.rail_env = observations[0]
            self._init_env(self.rail_env)
        self._start_step()
        return {handle: self._act(handle, observations[handle]) for handle in handles}

    def _act(self, handle: int, state, eps=0.) -> RailEnvActions:
        check = self.agent_can_move.get(handle, None)
        agent = self.rail_env.agents[handle]
        if (agent.handle not in self._set_paths or self._set_paths[agent.handle] is None) and agent.state < TrainState.MOVING:
            # prevent entering map as default!
            act = RailEnvActions.DO_NOTHING
        else:
            act = RailEnvActions.STOP_MOVING

        if agent.current_configuration is not None:
            self.agent_waypoints_done[handle].add(Waypoint(*agent.current_configuration))

        if check is not None:
            act = check[3]
            self.num_blocked[handle] = 0
        else:
            self.num_blocked[handle] += 1
            if agent.state in [TrainState.MOVING, TrainState.STOPPED]:
                self._find_alternative(agent)

        # TODO port to client.py:  File "msgpack/_packer.pyx", line 257, in msgpack._cmsgpack.Packer._pack_inner
        # submission-1      | TypeError: can not serialize 'RailEnvActions' object
        # if isinstance(act, RailEnvActions):
        #    act = act.value
        return act

    def _find_alternative(self, agent: EnvAgent):
        handle = agent.handle
        if self.verbose:
            print(f"considering {handle} at {self.rail_env._elapsed_steps}: {self._set_paths[handle]}")
        if self.audit is not None:
            self.audit.append({"env_time": self.rail_env._elapsed_steps, "agent_id": agent.handle, "k": "audit",
                               "v": f"considering {handle} at {self.rail_env._elapsed_steps}: {self._set_paths[handle]}"})

        # TODO optimization: instead of computing the remaining flexible waypoints, update the list on the go. No priority for now
        remaining_flexible_waypoints = self._get_remaining_flexible_waypoints(agent)
        if self.drop_next_threshold is not None and self.num_blocked[handle] > self.drop_next_threshold and len(remaining_flexible_waypoints) > 1:
            if self.verbose:
                print(f"dropping next intermediate for {agent.handle} at {self.rail_env._elapsed_steps}, blocked for {self.num_blocked[agent.handle]}")
            if self.audit is not None:
                self.audit.append({"env_time": self.rail_env._elapsed_steps, "agent_id": agent.handle, "k": "audit",
                                   "v": f"dropping next intermediate for {agent.handle} at {self.rail_env._elapsed_steps}, blocked for {self.num_blocked[agent.handle]}"})
            remaining_flexible_waypoints = remaining_flexible_waypoints[1:]

        if self.use_k_alternatives_at_first_intermediate_and_then_always_first_strategy is not None and \
                self.use_k_alternatives_at_first_intermediate_and_then_always_first_strategy > 0 and \
                len(remaining_flexible_waypoints[0]) > 0:
            before = self._set_paths[handle]

            if handle not in self.alternatives or self.alternatives[handle][0][0] != Waypoint(*agent.current_configuration):
                if self.verbose:
                    print(f"need to re-compute for agent {handle} at {agent.current_configuration} at {self.rail_env._elapsed_steps}")
                if self.audit is not None:
                    self.audit.append({"env_time": self.rail_env._elapsed_steps, "agent_id": agent.handle, "k": "audit",
                                       "v": f"need to re-compute for agent {handle} at {agent.current_configuration} at {self.rail_env._elapsed_steps}"})

                alternatives = []
                for first_intermediate in remaining_flexible_waypoints[0]:
                    then_always_first_intermediates = [[first_intermediate]] + _always_first_waypoint_from_flexible_groups(remaining_flexible_waypoints[1:])
                    prefixes = _get_k_shortest_paths(None, agent.current_configuration[0], agent.current_configuration[1], first_intermediate.position,
                                                     target_direction=first_intermediate.direction,
                                                     rail=self.rail_env.rail,
                                                     k=self.use_k_alternatives_at_first_intermediate_and_then_always_first_strategy,
                                                     cutoff=self.k_shortest_path_cutoff)
                    suffix = self._shortest_path_from_non_flexible_waypoints(then_always_first_intermediates, self.rail_env.rail,
                                                                             debug_label=f"Agent {agent.handle}")
                    for prefix in prefixes:
                        alternatives.append(list(prefix) + suffix[1:])
                self.alternatives[handle] = alternatives

            # as in set path before:
            self.closed[handle].append(before)

            # randomize the alternative if all alternatives already tried
            assert len(self.alternatives[handle]) > 0, "Either cutoff too low or not reachable."
            alternative = self.alternatives[handle][self.np_random.randint(len(self.alternatives[handle]))]
            for alt in self.alternatives[handle]:
                if alt not in self.closed[handle]:
                    alternative = alt
            self.closed[handle].append(alternative)

            if self.verbose:
                print(f"get new path for agent {handle} using alternative-at-first-intermediate-and-then-always-first strategy on {agent.waypoints}")
            if self.audit is not None:
                self.audit.append({"env_time": self.rail_env._elapsed_steps, "agent_id": agent.handle, "k": "audit",
                                   "v": f"get new path for agent {handle} using alternative-at-first-intermediate-and-then-always-first strategy on {agent.waypoints}"})

            if before == self._set_paths[handle]:
                if self.verbose:
                    print(
                        f"not changed {handle} at {self.rail_env._elapsed_steps} {len(before)}->{len(self._set_paths[handle])}:\n - {before} \n - {self._set_paths[handle]} ")
                if self.audit is not None:
                    self.audit.append({"env_time": self.rail_env._elapsed_steps, "agent_id": agent.handle, "k": "audit",
                                       "v": f"not changed {handle} at {self.rail_env._elapsed_steps} {len(before) if before is not None else None}->{len(self._set_paths[handle]) if self._set_paths[agent.handle] is not None else None}:\n - {before} \n - {self._set_paths[handle]} "})

            else:
                if self.verbose:
                    print(
                        f"changed {handle} at {self.rail_env._elapsed_steps} {len(before)}->{len(self._set_paths[handle])}:\n - {before} \n - {self._set_paths[handle]}")
                if self.audit is not None:
                    self.audit.append({"env_time": self.rail_env._elapsed_steps, "agent_id": agent.handle, "k": "audit",
                                       "v": f"changed {handle} at {self.rail_env._elapsed_steps} {len(before)}->{len(self._set_paths[handle])}:\n - {before} \n - {self._set_paths[handle]}"})

            if self._set_paths[handle] is None or len(self._set_paths[handle]) == 0:
                self._set_paths[handle] = before
            self.init_shortest_distance_positions(agent, handle)
            self.opp_agent_map.pop(handle, None)

    def _get_remaining_flexible_waypoints(self, agent):
        remaining_flexible_waypoints: List[List[Waypoint]] = copy.deepcopy(agent.waypoints)
        while True:
            if set(remaining_flexible_waypoints[0]).isdisjoint(self.agent_waypoints_done[agent.handle]):
                break
            remaining_flexible_waypoints = remaining_flexible_waypoints[1:]
        assert len(remaining_flexible_waypoints) > 0
        return remaining_flexible_waypoints

    def _start_step(self):
        # (1)
        self._build_agent_position_map()

        # (2)
        self._update_shortest_distance_maps_and_opp_agent_map()

        # (3)
        self._extract_agent_can_move()

    def _build_agent_position_map(self):
        """
        start_step (1): update agent positions at start of step.
        More precisely, updates:
        - `self.agent_positions`
        """
        # build map with agent positions (only active agents)
        self.agent_positions = np.zeros((self.rail_env.height, self.rail_env.width), dtype=int) - 1
        for handle in range(self.rail_env.get_num_agents()):
            agent = self.rail_env.agents[handle]
            if agent.state in [TrainState.MOVING, TrainState.STOPPED, TrainState.MALFUNCTION]:
                if agent.current_configuration is not None:
                    self.agent_positions[agent.current_configuration[0]] = handle

    def _update_shortest_distance_maps_and_opp_agent_map(self):
        """
       start_step (2): update the shortest paths to current position and update opposing agent as well as bitmap representation at start of step.
        More precisely, updates:
        - `super()._shortest_paths`
        - `self.shortest_distance_agent_map` (2.2)
        - `self.full_shortest_distance_agent_map` (2.2)
        - `self.shortest_distance_positions_directions_agent_map` (2.2)
        - `self.opp_agent_map` (2.3.2)
        - `self.shortest_distance_agent_map` (2.3.3)
        - `self.shortest_distance_agent_len` (2.3.3)
        """
        # (2.0)
        all_agent_positions: Set[Tuple[int, int]] = self._collect_all_agent_positions()

        for agent in self.rail_env.agents:
            handle = agent.handle

            # (2.1)
            super()._update_agent(agent, self.rail_env)

            # (2.2)
            self._build_full_shortest_distance_agent_map(agent, handle)
            if agent.state == TrainState.DONE or agent.state == TrainState.WAITING:
                continue

            # (2.3)
            self._build_shortest_distance_agent_map(agent, handle, all_agent_positions)

    def _collect_all_agent_positions(self):
        """
        start_step (2.0)
        """
        all_agent_positions = set()
        for agent in self.rail_env.agents:
            all_agent_positions.add(agent.current_configuration[0] if agent.current_configuration is not None else None)
        return all_agent_positions

    def _build_full_shortest_distance_agent_map(self, agent, handle):
        """
        start_step (2.2)

        Updates:
        - `self.full_shortest_distance_agent_map`
        - `self.shortest_distance_agent_map`
        - `self.shortest_distance_positions_directions_agent_map`
        """
        # (2.2.1)
        if self.rail_env._elapsed_steps == 1:
            self.init_shortest_distance_positions(agent, handle)
        # (2.2.2)
        position = agent.current_configuration[0] if agent.current_configuration is not None else None
        old_position = agent.old_configuration[0] if agent.old_configuration is not None else None
        if position is not None and position != old_position:
            assert position == self._set_paths[agent.handle][0].position
            if position not in {wp.position for wp in self._set_paths[agent.handle][1:]}:
                self.full_shortest_distance_agent_map[(handle, position[0], position[1])] = 0
                # the initial position is never added to shortest_distance_positions_agent_map
                if old_position is not None:
                    self.shortest_distance_positions_agent_map[handle].remove(position)
            # the initial position is never added to shortest_distance_positions_agent_map
            # N.B. We must remove each direction separately, as there can be "loopy" paths going through same cell twice but with different direction!
            if old_position is not None:
                self.shortest_distance_positions_directions_agent_map[handle][position].remove(int(agent.current_configuration[1]))

    def init_shortest_distance_positions(self, agent, handle):
        """
        start_step (2.2.1)

        Initializes:
        - `self.shortest_distance_agent_map`
        - `self.shortest_distance_positions_directions_agent_map`
        """
        self.full_shortest_distance_agent_map[handle].fill(0)
        self.shortest_distance_positions_agent_map[handle] = set()
        self.shortest_distance_positions_directions_agent_map[handle] = defaultdict(set)
        if self._set_paths[agent.handle] is None:
            return
        for wp in self._set_paths[agent.handle][1:]:
            position, direction = wp.position, wp.direction
            self.full_shortest_distance_agent_map[(handle, position[0], position[1])] = 1
            self.shortest_distance_positions_agent_map[handle].add(position)
            self.shortest_distance_positions_directions_agent_map[handle][position].add(direction)

    def _build_shortest_distance_agent_map(self, agent, handle, all_agent_positions):
        """
        start_step (2.3)
        """
        # (2.3.1)
        prev_opp_agents = self.opp_agent_map[handle]
        overlap = self.shortest_distance_positions_agent_map[handle].intersection(all_agent_positions)
        if overlap == prev_opp_agents:
            return

        # (2.3.2)
        self._rebuild_opp_agent_map(handle, overlap)
        # (2.3.3)
        self._rebuild_shortest_distance_agent_map(agent, handle)

    def _rebuild_shortest_distance_agent_map(self, agent, handle):
        """
        start_step (2.3.3)

        Updates:
        - `self.shortest_distance_agent_map`
        - `self.shortest_distance_agent_len`
        """
        # TODO performance improvement idea: we could update over the previous opposing trains (beware of close-following, map must be non-binary) and update their position and we only have to look at non-facing switches where new opposing trains can occur.
        self.shortest_distance_agent_map[handle].fill(0)
        self.shortest_distance_agent_len[handle] = 0
        num_opp_agents = 0
        for wp in self._set_paths[agent.handle][1:]:
            position, direction = wp.position, wp.direction
            opp_a = self.agent_positions[position]
            if opp_a != -1 and opp_a != handle:
                if self.rail_env.agents[opp_a].current_configuration[1] != direction:
                    num_opp_agents += 1
                    break
            if num_opp_agents == 0:
                self.shortest_distance_agent_len[handle] += 1
                self.shortest_distance_agent_map[(handle, position[0], position[1])] = 1

    def _rebuild_opp_agent_map(self, handle, overlap):
        """
        start_step (2.3.2)
        Updates:
        - `self.opp_agent_map` (2.3.2)
        """
        self.opp_agent_map[handle] = set()
        for position in overlap:
            opp_a = self.agent_positions[position]
            if opp_a != -1 and opp_a != handle:
                directions = self.shortest_distance_positions_directions_agent_map[handle][position]
                assert len(directions) > 0, f"Inconsistency for agent {handle} at {self.rail_env._elapsed_steps}: no directions for position {position}"
                for direction in directions:
                    if self.rail_env.agents[opp_a].current_configuration[1] != direction:
                        self.opp_agent_map[handle].add(opp_a)

    def _get_action(self, configuration: Tuple[Tuple[int, int], int], next_configuration: Tuple[Tuple[int, int], int]):
        for action in [RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_RIGHT]:
            new_cell_valid, new_configuration, transition_valid, preprocessed_action, _ = self.rail_env.rail._check_action_on_agent(action, configuration)
            if new_configuration == next_configuration:
                return preprocessed_action
        raise

    def _extract_agent_can_move(self):
        """
        start_step (3): update whether agent can move. More precisely, updates:
        - `self.agent_can_move`
        """
        self.agent_can_move = {}

        for handle in range(self.rail_env.get_num_agents()):
            agent = self.rail_env.agents[handle]
            if TrainState.DONE > agent.state >= TrainState.WAITING:
                if self._check_agent_can_move(
                        self.shortest_distance_agent_map[handle],
                        self.shortest_distance_agent_len[handle],
                        self.opp_agent_map.get(handle, set()),
                        self.full_shortest_distance_agent_map,
                        agent.handle,
                        self.switches,
                        self.count_num_opp_agents_towards_min_free_cell,
                ):
                    if agent.current_configuration is not None:
                        position = agent.current_configuration[0]
                        direction = agent.current_configuration[1]
                    else:
                        position = agent.initial_configuration[0]
                        direction = agent.initial_configuration[1]
                    # Guard against loopy lines and invalid initial positions:
                    if self._set_paths[agent.handle] is None or len(self._set_paths[agent.handle]) < 2:
                        warnings.warn(f"No shortest path for agent {agent.handle}. Found: {self._set_paths[agent.handle]}")
                        if self.audit is not None:
                            self.audit.append({"env_time": self.rail_env._elapsed_steps, "agent_id": handle, "k": "audit",
                                               "v": f"No shortest path for agent {agent.handle}. Found: {self._set_paths[agent.handle]}"})
                        continue
                    next_position = self._set_paths[agent.handle][1].position
                    next_direction = self._set_paths[agent.handle][1].direction
                    action = self._get_action((position, direction), (next_position, next_direction))

                    self.agent_can_move.update({handle: [next_position[0], next_position[1], next_direction, action]})
        if self.use_entering_prevention:
            entering_agents = [handle for handle, agent in enumerate(self.rail_env.agents) if
                               agent.state == TrainState.READY_TO_DEPART and self.agent_can_move.get(handle, None)]
            if len(entering_agents) > 0:
                if self.verbose:
                    print(f" ++++ {self.rail_env._elapsed_steps} entering {entering_agents}")
                for a1 in entering_agents:
                    for a2 in entering_agents:
                        if a1 != a2 and a1 in self.agent_can_move and a2 in self.agent_can_move:
                            free = self._get_free(a1, a2)
                            if len(free) < self.min_free_cell:
                                self.agent_can_move.pop(a1)
                                if self.verbose:
                                    print(f"!!!! prevent entering conflict {a1, a2} -> let not enter {a1}")

        if self.show_debug_plot:
            a = np.floor(np.sqrt(self.rail_env.get_num_agents()))
            b = np.ceil(self.rail_env.get_num_agents() / a)
            for handle in range(self.rail_env.get_num_agents()):
                import matplotlib.pyplot as plt; plt.subplot(a, b, handle + 1)
                import matplotlib.pyplot as plt; plt.imshow(self.full_shortest_distance_agent_map[handle] + self.shortest_distance_agent_map[handle])
            import matplotlib.pyplot as plt; plt.show(block=False)
            import matplotlib.pyplot as plt; plt.pause(0.01)

    def _check_agent_can_move(
            self,
            my_shortest_walking_path: np.ndarray,
            my_shortest_walking_path_len: int,
            opp_agents: Set,
            full_shortest_distance_agent_map: np.ndarray,
            handle: AgentHandle,
            switches: Optional[np.ndarray] = None,
            count_num_opp_agents_towards_min_free_cell: bool = False,
            debug: bool = False,
    ):
        """
        The algorithm collects for each train along its route all trains that are currently on a resource in the route.
        For each collected train (`opp_agents`), the method has to decide at which position along the route the train
        must let pass the collected opposing train:  by searching the train's path required resources backward along the path
        starting at the collected train position; stop the search when the resource along the collected train's path is not equal.
        This yields `free_cells` ahead of the agent without overlap with any opposing agent's travelling path.
        If `free_cells >= min_free_cells >= 1` for all opposing agents, then the agent can move.
        A deadlock can only occur if a jam "fills in" the free space and is not detected by the algorithm.

        To determine `free_cells`, the implementation compares takes the difference of
        - the bitmap of the agent's shortest path (up to first opposing agent) and
        - the bitmap opposing agent's path
        and counts the positive elements.

        The forward and backward traveling along the train and the collected train path must be done step-by-step synchronous.
        If the first non-equal resource position along the train's path is more than one resource from train's current location away,
        then the train can move and no deadlock will occur for the next time step.

        2 heuristics to avoid "fill-in":
        - switches: if switches is given, then switches do not count towards free cells
        - count_num_opp_agents_towards_min_free_cell: the number of opposing agents is added to `min_free_cell`

        Parameters
        ----------
        my_shortest_walking_path : np.ndarray
            the shortest path up to first opposing agent
        my_shortest_walking_path_len: int
            length of shortest path up to first opposing agent
        opp_agents : np.ndarray
            positions of opposing agents
        full_shortest_distance_agent_map : np.ndarray
            full shortest path
        switches: Optional[np.ndarray] = None
            positions of switches. If None, disable the switches heuristics.
        count_num_opp_agents_towards_min_free_cell: bool = False
            the number of opposing agents is added to `min_free_cell`.
        """
        len_opp_agents = len(opp_agents)
        if len_opp_agents == 0:
            return True

        # TODO does this make sense? min_free_cell - len_opp_agents often <= 0?
        if my_shortest_walking_path_len < self.min_free_cell - len_opp_agents:
            if self.verbose:
                print(f" *** {self.rail_env._elapsed_steps}: agent cannot move")
            return False
        min_free_cell = self.min_free_cell
        if count_num_opp_agents_towards_min_free_cell:
            min_free_cell += len_opp_agents

        for opp_a in opp_agents:
            opp = full_shortest_distance_agent_map[opp_a]
            if switches is None:
                free_cells = np.count_nonzero((my_shortest_walking_path - opp) > 0)
            else:
                free_cells = np.count_nonzero((my_shortest_walking_path - switches - opp) > 0)

            if free_cells < min_free_cell:
                free = self._get_free(handle, opp_a)

                if self.verbose:
                    print(
                        f" *** {self.rail_env._elapsed_steps}: agent {handle} blocked by {opp_a} with {free_cells}: {free}. All oncoming agents on path {opp_agents}")
                if self.audit is not None:
                    self.audit.append({"env_time": self.rail_env._elapsed_steps, "agent_id": handle, "k": "audit",
                                       "v": f" *** {self.rail_env._elapsed_steps}: agent {handle} blocked by {opp_a} with {free_cells}: {free}. All oncoming agents on path {opp_agents}"})
                if debug:
                    cells_1 = [wp.position for wp in self._set_paths[handle]]
                    cells_2 = [wp.position for wp in self._set_paths[opp_a]]
                    if self.verbose:
                        print(f"cells_1 = {cells_1}; cells_2={cells_2}")
                    im1 = np.zeros((self.rail_env.height, self.rail_env.width))
                    for cell in cells_1:
                        im1[cell] = 1
                    import matplotlib.pyplot as plt; ax = plt.subplot(1, 2, 1)
                    ax.set_title(f"Agent {handle} set path ({len(cells_1)})")
                    import matplotlib.pyplot as plt; plt.imshow(im1)

                    im2 = np.zeros((self.rail_env.height, self.rail_env.width))
                    for cell in cells_2:
                        im2[cell] = 1
                    import matplotlib.pyplot as plt; ax = plt.subplot(1, 2, 2)
                    ax.set_title(f"Agent {opp_a} set path ({len(cells_2)})")
                    import matplotlib.pyplot as plt; plt.imshow(im2)
                    import matplotlib.pyplot as plt; plt.show()

                    import matplotlib.pyplot as plt; ax = plt.subplot(4, 1, 1)
                    ax.set_title(f"Agent {handle} full path ({np.count_nonzero(full_shortest_distance_agent_map[handle])})")
                    import matplotlib.pyplot as plt; plt.imshow(full_shortest_distance_agent_map[handle])

                    import matplotlib.pyplot as plt; ax = plt.subplot(4, 1, 2)
                    ax.set_title(f"Agent {handle} my_shortest_walking_path ({np.count_nonzero(my_shortest_walking_path)})")
                    import matplotlib.pyplot as plt; plt.imshow(my_shortest_walking_path)

                    import matplotlib.pyplot as plt; ax = plt.subplot(4, 1, 3)
                    ax.set_title(f"Agent {opp_a} full path ({np.count_nonzero(opp)})")
                    import matplotlib.pyplot as plt; plt.imshow(opp)

                    import matplotlib.pyplot as plt; ax = plt.subplot(4, 1, 4)
                    ax.set_title(f"Agent {handle} - agent free_cells  {opp_a} ({free_cells})")
                    import matplotlib.pyplot as plt; plt.imshow(my_shortest_walking_path - opp)
                    import matplotlib.pyplot as plt; plt.show()

                return False
        return True

    def _get_free(self, handle: AgentHandle, opp_a: AgentHandle):
        """
        Hown many cells free ahead of me till other agent.
        Returns zero if no path set for myself.
        Returns full path minus current position if no overlap or no path set for other agent.
        """
        own_path = self._set_paths.get(handle, None)
        opp_path = self._set_paths.get(opp_a, None)
        if own_path is None:
            return 0
        elif opp_path is None:
            return len(own_path) - 1

        return _get_free_from_path(own_path, opp_path)

    def save(self, filename):
        pass

    def load(self, filename):
        pass

    @_enable_flatland_deadlock_avoidance_policy_lru_cache(maxsize=100000)
    def _is_switch_cell(self, position) -> bool:
        for new_dir in range(4):
            possible_transitions = self.rail_env.rail.get_transitions((position, new_dir))
            num_transitions = fast_count_nonzero(possible_transitions)
            if num_transitions > 1:
                return True
        return False


def _get_free_from_path(own_path: List[Waypoint], opp_path: List[Waypoint]):
    my_cells = {wp.position for wp in own_path[1:]}
    opp_cells = {wp.position for wp in opp_path}
    my_cells_own = my_cells.difference(opp_cells)
    num = 0
    for i, wp in enumerate(own_path):
        num = i
        if wp.position not in my_cells_own:
            num = i - 1
            break
    free = own_path[:num + 1]
    return free


class DeadlockAvoidanceHeuristics(DeadLockAvoidancePolicy):
    def __init__(self,
                 use_alternative_at_first_intermediate_and_then_always_first_strategy=None,
                 seed: int = None,
                 audit: bool = False,
                 ):
        seed = os.environ.get("DLA_SEED", seed)
        if seed is not None:
            seed = int(seed)
        super().__init__(
            count_num_opp_agents_towards_min_free_cell=False,
            use_switches_heuristic=False,
            use_entering_prevention=True,
            use_alternative_at_first_intermediate_and_then_always_first_strategy=use_alternative_at_first_intermediate_and_then_always_first_strategy,
            drop_next_threshold=20,
            k_shortest_path_cutoff=450,
            seed=seed,
            audit=audit,
        )
