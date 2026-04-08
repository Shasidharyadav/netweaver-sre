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
    SUPPORTS_CONCURRENT_SESSIONS: bool = False
    MAX_ATTEMPTS: int = 15

_GLOBAL_CACHE = {
    "state": State(episode_id=str(uuid4()), step_count=0),
    "queue_depths": {"spine_1": 10.0, "spine_2": 10.0},
    "gradient_vars": [0.01] * 16,
    "logs": [],
    "faulty_node_id": "",
    "active_task": "easy",
    "target_pfc": 0.0
}

class NetweaverSreEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = False
    MAX_ATTEMPTS: int = 15

    def __init__(self):
        pass # state is managed globally

    def reset(self, **kwargs) -> NetweaverSreObservation:
        global _FORCED_TASK_LEVEL, _GLOBAL_CACHE
        
        _GLOBAL_CACHE["state"] = State(episode_id=str(uuid4()), step_count=0)
        
        forced = _FORCED_TASK_LEVEL or os.getenv("FORCE_TASK_LEVEL", "")
        _GLOBAL_CACHE["active_task"] = forced if forced else kwargs.get("task_level", random.choice(["easy", "medium", "hard"]))
        _GLOBAL_CACHE["faulty_node_id"] = f"node_{random.randint(0, 99)}"
        _GLOBAL_CACHE["target_pfc"] = float(random.randint(40, 80))
        
        _GLOBAL_CACHE["queue_depths"] = {"spine_1": 10.0, "spine_2": 10.0}
        _GLOBAL_CACHE["gradient_vars"] = [0.01] * 16
        _GLOBAL_CACHE["logs"] = [f"CLUSTER INIT: Clos Topology Active. Running mode: {_GLOBAL_CACHE['active_task'].upper()}"]
        
        task = _GLOBAL_CACHE["active_task"]
        fnode = _GLOBAL_CACHE["faulty_node_id"]
        
        if task == "easy":
            _GLOBAL_CACHE["logs"].append(f"ALERT: Node offline. Isolate {fnode} using DRAIN_TRAFFIC.")
        elif task == "medium":
            _GLOBAL_CACHE["queue_depths"]["spine_1"] = 99.9  # Buffer Congestion
            _GLOBAL_CACHE["logs"].append(f"WARN: Buffer overshoot on spine_1. Identify target threshold (Target: {_GLOBAL_CACHE['target_pfc']}) and adjust via TUNE_PFC_THRESHOLD.")
        elif task == "hard":
            fault_idx = int(fnode.split("_")[1]) // 10  # Map node number 0-99 to 10 sub-clusters
            _GLOBAL_CACHE["gradient_vars"][fault_idx] = 999.9  # Massive NaN contagion variance
            _GLOBAL_CACHE["logs"].append("CRITICAL: Loss diverging rapidly. Perform binary search triage using RUN_MINI_ITERATION (e.g., target='0-5') to locate NaN source then DRAIN_TRAFFIC the node.")
        
        return self._get_obs(done=False, reward=0.0)

    def step(self, action: NetweaverSreAction) -> NetweaverSreObservation:  # type: ignore[override]
        global _GLOBAL_CACHE
        _GLOBAL_CACHE["state"].step_count += 1
        
        cmd = action.command.upper()
        tgt = action.target
        val = action.value
        
        done = False
        reward = 0.0
        
        task = _GLOBAL_CACHE["active_task"]
        fnode = _GLOBAL_CACHE["faulty_node_id"]
        logs = _GLOBAL_CACHE["logs"]
        qdepths = _GLOBAL_CACHE["queue_depths"]

        if cmd == "DRAIN_TRAFFIC":
            if tgt == fnode:
                logs.append(f"SUCCESS: Faulty node {tgt} isolated.")
                step_penalty = 0.05 if task == "hard" else 0.1
                raw_score = 1.0 - ((_GLOBAL_CACHE["state"].step_count - 1) * step_penalty)
                reward = min(0.99, max(0.01, round(raw_score, 2)))
                done = True
            else:
                logs.append(f"ERROR: Drained healthy node {tgt}.")
                reward = 0.01

        elif cmd == "TUNE_PFC_THRESHOLD":
            if task == "medium":
                if val is not None:
                    logs.append(f"EXEC: Tuned PFC Threshold to {val}.")
                    distance = abs(_GLOBAL_CACHE["target_pfc"] - float(val))
                    rel_improvement = max(0.0, 1.0 - (distance / 40.0))
                    qdepths["spine_1"] = max(10.0, 99.9 - (rel_improvement * 89.9))
                    
                    if distance <= 5.0: # Agent tuned it close enough
                        logs.append("SUCCESS: Congestion fully mitigated.")
                        raw_score = 1.0 - ((_GLOBAL_CACHE["state"].step_count - 1) * 0.1)
                        reward = min(0.99, max(0.01, round(raw_score, 2)))
                        done = True
                    else:
                        logs.append(f"INFO: Network partially relieved. Still experiencing collision.")
                else:
                    logs.append("ERROR: TUNE_PFC_THRESHOLD requires a value parameter.")
            else:
                logs.append("ERROR: Tuning PFC is irrelevant to the current fault.")

        elif cmd == "RUN_MINI_ITERATION":
            if task == "hard":
                try:
                    start, end = map(int, tgt.split("-"))
                    fault_idx = int(fnode.split("_")[1]) // 10
                    span = end - start
                    if start <= fault_idx <= end:
                        if span == 0:
                            logs.append(f"TRIAGE CONFIRMED: NaN source is {fnode} (sub-cluster {start}). Issue DRAIN_TRAFFIC with target='{fnode}'.")
                        else:
                            mid = (start + end) // 2
                            logs.append(f"TRIAGE HIT: Fault is inside range {start}-{end}. Narrow it — try {start}-{mid} or {mid+1}-{end} next.")
                    else:
                        logs.append(f"TRIAGE CLEAR: Range {start}-{end} is healthy. Search the complementary range (0-9 excluding this one).")
                except Exception:
                    logs.append("ERROR: RUN_MINI_ITERATION target must be format 'start-end' (e.g. '0-5').")
            else:
                logs.append("ERROR: Reductive triage only effective for tracking SDC faults.")
                
        else:
            logs.append(f"UNKNOWN/INVALID ACTION: {cmd}")
            
        if _GLOBAL_CACHE["state"].step_count >= self.MAX_ATTEMPTS and not done:
            logs.append("SLA BREACH: Timeout limit reached.")
            done = True
            reward = 0.01

        return self._get_obs(done, reward)

    @property
    def state(self) -> State:
        return _GLOBAL_CACHE["state"]

    def _get_obs(self, done, reward) -> NetweaverSreObservation:
        global _GLOBAL_CACHE
        return NetweaverSreObservation(
            done=done,
            reward=reward,
            queue_depths=_GLOBAL_CACHE["queue_depths"].copy(),
            gradient_variances=_GLOBAL_CACHE["gradient_vars"].copy(),
            hardware_logs=_GLOBAL_CACHE["logs"][-5:],
            system_health=1.0
        )
