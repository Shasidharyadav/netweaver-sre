# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Netweaver Sre Environment Implementation.
Autonomous fault detection and routing for 100k GPU clusters.
"""

import random
import os
from uuid import uuid4

# Module-level task level control — set via /set_level HTTP endpoint
_FORCED_TASK_LEVEL: str = ""

def set_task_level(level: str) -> None:
    global _FORCED_TASK_LEVEL
    _FORCED_TASK_LEVEL = level.lower().strip()

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import NetweaverSreAction, NetweaverSreObservation
except ImportError:
    from models import NetweaverSreAction, NetweaverSreObservation

class NetweaverSreEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    MAX_ATTEMPTS: int = 15

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._queue_depths = {"spine_1": 10.0, "spine_2": 10.0}
        self._gradient_vars = [0.01] * 16
        self._logs = []
        self._faulty_node_id = ""
        self._active_task = "easy"
        self._target_pfc = 0.0

    def reset(self, **kwargs) -> NetweaverSreObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        # Randomly choose task difficulty if not explicitly passed
        # Priority: HTTP /set_level call > FORCE_TASK_LEVEL env var > random
        global _FORCED_TASK_LEVEL
        forced = _FORCED_TASK_LEVEL or os.getenv("FORCE_TASK_LEVEL", "")
        self._active_task = forced if forced else kwargs.get("task_level", random.choice(["easy", "medium", "hard"]))
        self._faulty_node_id = f"node_{random.randint(0, 99)}"
        self._target_pfc = float(random.randint(40, 80))
        
        self._queue_depths = {"spine_1": 10.0, "spine_2": 10.0}
        self._gradient_vars = [0.01] * 16
        self._logs = [f"CLUSTER INIT: Clos Topology Active. Running mode: {self._active_task.upper()}"]
        
        if self._active_task == "easy":
            self._logs.append(f"ALERT: Node offline. Isolate {self._faulty_node_id} using DRAIN_TRAFFIC.")
        elif self._active_task == "medium":
            self._queue_depths["spine_1"] = 99.9  # Buffer Congestion
            self._logs.append(f"WARN: Buffer overshoot on spine_1. Identify target threshold (Target: {self._target_pfc}) and adjust via TUNE_PFC_THRESHOLD.")
        elif self._active_task == "hard":
            fault_idx = int(self._faulty_node_id.split("_")[1]) // 10  # Map node number 0-99 to 10 sub-clusters
            self._gradient_vars[fault_idx] = 999.9  # Massive NaN contagion variance in specific index
            self._logs.append("CRITICAL: Loss diverging rapidly. Perform binary search triage using RUN_MINI_ITERATION (e.g., target='0-5') to locate NaN source then DRAIN_TRAFFIC the node.")
        
        return self._get_obs(done=False, reward=0.0)

    def step(self, action: NetweaverSreAction) -> NetweaverSreObservation:  # type: ignore[override]
        self._state.step_count += 1
        cmd = action.command.upper()
        tgt = action.target
        val = action.value
        
        done = False
        reward = 0.0

        if cmd == "DRAIN_TRAFFIC":
            if tgt == self._faulty_node_id:
                self._logs.append(f"SUCCESS: Faulty node {tgt} isolated.")
                reward = max(0.0, 1.0 - ((self._state.step_count - 1) * 0.1))
                done = True
            else:
                self._logs.append(f"ERROR: Drained healthy node {tgt}.")
                reward = -0.2

        elif cmd == "TUNE_PFC_THRESHOLD":
            if self._active_task == "medium":
                if val is not None:
                    self._logs.append(f"EXEC: Tuned PFC Threshold to {val}.")
                    distance = abs(self._target_pfc - float(val))
                    rel_improvement = max(0.0, 1.0 - (distance / 40.0))
                    self._queue_depths["spine_1"] = max(10.0, 99.9 - (rel_improvement * 89.9))
                    
                    if distance <= 5.0: # Agent tuned it close enough
                        self._logs.append("SUCCESS: Congestion fully mitigated.")
                        reward = max(0.0, 1.0 - ((self._state.step_count - 1) * 0.1))
                        done = True
                    else:
                        self._logs.append(f"INFO: Network partially relieved. Still experiencing collision.")
                else:
                    self._logs.append("ERROR: TUNE_PFC_THRESHOLD requires a value parameter.")
            else:
                self._logs.append("ERROR: Tuning PFC is irrelevant to the current fault.")

        elif cmd == "RUN_MINI_ITERATION":
            if self._active_task == "hard":
                try:
                    start, end = map(int, tgt.split("-"))
                    fault_idx = int(self._faulty_node_id.split("_")[1]) // 10
                    span = end - start
                    if start <= fault_idx <= end:
                        if span == 0:
                            self._logs.append(f"TRIAGE CONFIRMED: NaN source is {self._faulty_node_id} (sub-cluster {start}). Issue DRAIN_TRAFFIC with target='{self._faulty_node_id}'.")
                        else:
                            mid = (start + end) // 2
                            self._logs.append(f"TRIAGE HIT: Fault is inside range {start}-{end}. Narrow it — try {start}-{mid} or {mid+1}-{end} next.")
                    else:
                        self._logs.append(f"TRIAGE CLEAR: Range {start}-{end} is healthy. Search the complementary range (0-9 excluding this one).")
                except Exception:
                    self._logs.append("ERROR: RUN_MINI_ITERATION target must be format 'start-end' (e.g. '0-5').")
            else:
                self._logs.append("ERROR: Reductive triage only effective for tracking SDC faults.")
                
        else:
            self._logs.append(f"UNKNOWN/INVALID ACTION: {cmd}")
            
        if self._state.step_count >= self.MAX_ATTEMPTS and not done:
            self._logs.append("SLA BREACH: Timeout limit reached.")
            done = True
            reward = 0.0

        return self._get_obs(done, reward)

    @property
    def state(self) -> State:
        return self._state

    def _get_obs(self, done, reward) -> NetweaverSreObservation:
        return NetweaverSreObservation(
            done=done,
            reward=reward,
            queue_depths=self._queue_depths.copy(),
            gradient_variances=self._gradient_vars.copy(),
            hardware_logs=self._logs[-5:],
            system_health=1.0
        )
