# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
"""
Netweaver Sre Environment Implementation.
Autonomous fault detection and routing for 100k GPU clusters.
All scores are clamped strictly to (0.001, 0.999) — never 0.0 or 1.0.
"""

import random
import os
from uuid import uuid4

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

def clamp_score(raw: float) -> float:
    score = float(raw) if raw is not None else 0.001
    return max(0.001, min(0.999, score))

_GLOBAL_CACHE = {
    "state": State(episode_id=str(uuid4()), step_count=0),
    "queue_depths": {"sw_core_01": 10.0, "sw_core_02": 10.0, "sw_spine_01": 10.0},
    "gradient_vars": [0.01] * 10,
    "gpu_memory": [40.0] * 10,
    "logs": [],
    "faulty_node_id": "",
    "target_val": 0,
    "active_task": "t01",
    "last_grader_score": 0.001,
}

class NetweaverSreEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = False
    MAX_ATTEMPTS: int = 15

    def reset(self, **kwargs) -> NetweaverSreObservation:
        global _FORCED_TASK_LEVEL, _GLOBAL_CACHE
        _GLOBAL_CACHE["state"] = State(episode_id=str(uuid4()), step_count=0)

        forced = os.getenv("FORCE_TASK_LEVEL", "")
        forced = _FORCED_TASK_LEVEL if _FORCED_TASK_LEVEL else forced
        
        all_tasks = [f"t{i:02d}" for i in range(1, 21)]
        task = forced if forced in all_tasks else random.choice(all_tasks)
        if forced in ['easy', 'medium', 'hard']:
            if forced == "easy": task = random.choice(all_tasks[:7])
            elif forced == "medium": task = random.choice(all_tasks[7:14])
            else: task = random.choice(all_tasks[14:])

        _GLOBAL_CACHE["active_task"] = task
        fnode = f"node_{random.randint(0, 99)}"
        _GLOBAL_CACHE["faulty_node_id"] = fnode
        tval = random.randint(100, 900)
        _GLOBAL_CACHE["target_val"] = tval
        _GLOBAL_CACHE["last_grader_score"] = 0.001

        _GLOBAL_CACHE["queue_depths"] = {"sw_core_01": 10.0, "sw_core_02": 10.0, "sw_spine_01": 10.0, "sw_leaf_04": 10.0}
        _GLOBAL_CACHE["gradient_vars"] = [0.01] * 10
        _GLOBAL_CACHE["gpu_memory"] = [40.0] * 10
        _GLOBAL_CACHE["logs"] = [f"CLUSTER INIT: Clos Topology Active. Running task: {task.upper()}"]

        # Inject anomaly based on task
        if task == "t01":
            _GLOBAL_CACHE["logs"].append(f"ALERT: Node offline. Isolate {fnode} using DRAIN_TRAFFIC.")
        elif task == "t02":
            _GLOBAL_CACHE["logs"].append(f"ALERT: DNS resolution failing for {fnode}. Issue CLEAR_DNS_CACHE.")
        elif task == "t03":
            _GLOBAL_CACHE["logs"].append(f"CRITICAL: Service {fnode} crashed due to OOM. Issue RESTART_SERVICE.")
        elif task == "t04":
            _GLOBAL_CACHE["logs"].append(f"WARN: TLS cert expired on {fnode}. Issue RENEW_CERTIFICATE.")
        elif task == "t05":
            _GLOBAL_CACHE["logs"].append(f"WARN: Disk space 100% on {fnode}. Issue CLEAR_TEMP_FILES.")
        elif task == "t06":
            _GLOBAL_CACHE["logs"].append(f"ALERT: Pod {fnode} stuck in CrashLoop. Issue RESTART_POD.")
        elif task == "t07":
            _GLOBAL_CACHE["logs"].append(f"WARN: Zombie process holding lock on {fnode}. Issue KILL_ZOMBIE_PROCESS.")
        elif task == "t08":
            _GLOBAL_CACHE["logs"].append(f"WARN: Buffer overshoot on sw_core_01. Identify target threshold (Target: {tval}) and adjust via TUNE_PFC_THRESHOLD.")
        elif task == "t09":
            _GLOBAL_CACHE["logs"].append(f"WARN: Thermal throttle on {fnode}. Adjust power cap to {tval} W.")
        elif task == "t10":
            _GLOBAL_CACHE["logs"].append(f"WARN: BGP flap on {fnode} (AS {tval}). Suppress route using MITIGATE_ROUTE_FLAP.")
        elif task == "t11":
            _GLOBAL_CACHE["logs"].append(f"WARN: Jumbo frames dropping on {fnode}. Recommend INCREASE_MTU to {tval}.")
        elif task == "t12":
            _GLOBAL_CACHE["logs"].append(f"WARN: DDoS detected on {fnode}. Issue SET_RATE_LIMIT to {tval} req/s.")
        elif task == "t13":
            _GLOBAL_CACHE["logs"].append(f"WARN: Connection pool exhausted on {fnode}. Issue SCALE_CONN_POOL to {tval}.")
        elif task == "t14":
            _GLOBAL_CACHE["logs"].append(f"WARN: High context switching on {fnode}. Issue PIN_CPU_THREADS to {tval}.")
        elif task == "t15":
            fault_idx = int(fnode.split("_")[1]) // 10
            _GLOBAL_CACHE["gradient_vars"][fault_idx] = 999.9
            _GLOBAL_CACHE["logs"].append("CRITICAL: Loss diverging rapidly. RUN_MINI_ITERATION to locate NaN source then DRAIN_TRAFFIC.")
        elif task == "t16":
            _GLOBAL_CACHE["queue_depths"]["sw_core_02"] = 99.9
            _GLOBAL_CACHE["logs"].append("WARN: High latency detected. Inspect queue depths. Use ISOLATE_BROADCAST_STORM.")
        elif task == "t17":
            fault_idx = int(fnode.split("_")[1]) // 10
            _GLOBAL_CACHE["gpu_memory"][fault_idx] = 99.9
            _GLOBAL_CACHE["logs"].append("WARN: Sluggish training. Inspect GPU memory out of bounds. Use RESTART_GPU_DAEMON with sub_X.")
        elif task == "t18":
            _GLOBAL_CACHE["gradient_vars"] = [0.0] * 10
            _GLOBAL_CACHE["logs"].append("CRITICAL: Telemetry frozen. Issue ISSUE_GLOBAL_ROLLBACK on cluster_0.")
        elif task == "t19":
            _GLOBAL_CACHE["queue_depths"] = {"sw_core_01": 0.0, "sw_leaf_04": 99.9}
            _GLOBAL_CACHE["logs"].append("WARN: Reachability severed. Split queue topology. Use REBOOT_LEAF_SWITCHES on pod_x.")
        elif task == "t20":
            _GLOBAL_CACHE["logs"].append("CRITICAL: Health falling. Check storage integrity. Use PURGE_CORRUPT_BLOCK on db_cluster.")

        return self._get_obs(done=False, reward=clamp_score(0.01))

    def step(self, action: NetweaverSreAction) -> NetweaverSreObservation:
        global _GLOBAL_CACHE
        _GLOBAL_CACHE["state"].step_count += 1

        cmd = action.command.upper()
        tgt = action.target
        val = action.value

        done = False
        reward = clamp_score(0.01)

        task = _GLOBAL_CACHE["active_task"]
        fnode = _GLOBAL_CACHE["faulty_node_id"]
        tval = _GLOBAL_CACHE["target_val"]
        logs = _GLOBAL_CACHE["logs"]

        step_pen = 0.02
        hard_pen = 0.04
        
        c_map = {
            "t01": "DRAIN_TRAFFIC", "t02": "CLEAR_DNS_CACHE", "t03": "RESTART_SERVICE",
            "t04": "RENEW_CERTIFICATE", "t05": "CLEAR_TEMP_FILES", "t06": "RESTART_POD",
            "t07": "KILL_ZOMBIE_PROCESS"
        }
        m_map = {
            "t08": "TUNE_PFC_THRESHOLD", "t09": "ADJUST_POWER_CAP", "t10": "MITIGATE_ROUTE_FLAP",
            "t11": "INCREASE_MTU", "t12": "SET_RATE_LIMIT", "t13": "SCALE_CONN_POOL", "t14": "PIN_CPU_THREADS"
        }

        # Validate EASY
        if task in c_map:
            req_cmd = c_map[task]
            if cmd == req_cmd:
                logs.append(f"SUCCESS: Executed {cmd} on {tgt}.")
                reward = clamp_score(1.0 - ((_GLOBAL_CACHE["state"].step_count - 1) * step_pen))
                done = True
            else:
                logs.append(f"ERROR: Wrong command {cmd}. Expected {req_cmd}.")

        # Validate MEDIUM
        elif task in m_map:
            req_cmd = m_map[task]
            if cmd == req_cmd:
                if val is not None:
                    logs.append(f"EXEC: Configured {cmd} with {val}.")
                    distance = abs(tval - float(val))
                    if distance == 0:
                        logs.append("SUCCESS: Fully mitigated.")
                        reward = clamp_score(1.0 - ((_GLOBAL_CACHE["state"].step_count - 1) * step_pen))
                        done = True
                    else:
                        logs.append("INFO: Partially relieved. Wrong value.")
                        reward = clamp_score(0.5)
                else:
                    logs.append("ERROR: Value parameter required.")
            else:
                logs.append(f"ERROR: Wrong command {cmd}. Expected {req_cmd}.")

        # Validate HARD
        elif task == "t15":
            if cmd == "RUN_MINI_ITERATION":
                try:
                    start, end = map(int, tgt.split("-"))
                    f_idx = int(fnode.split("_")[1]) // 10
                    if start <= f_idx <= end:
                        if end - start == 0: logs.append(f"TRIAGE CONFIRMED: NaN source is {fnode}")
                        else: logs.append(f"TRIAGE HIT: inside range {start}-{end}")
                    else: logs.append(f"TRIAGE CLEAR: Range {start}-{end} healthy")
                except:
                    logs.append(f"ERROR: Invalid target format {tgt}. Expected start-end.")
            elif cmd == "DRAIN_TRAFFIC" and tgt == fnode:
                logs.append(f"SUCCESS: Faulty node {tgt} isolated.")
                reward = clamp_score(1.0 - ((_GLOBAL_CACHE["state"].step_count - 2) * hard_pen))
                done = True
            else: logs.append(f"ERROR: Invalid action for t15")

        elif task == "t16":
            if cmd == "ISOLATE_BROADCAST_STORM" and tgt == "sw_core_02":
                logs.append(f"SUCCESS: Broadcast storm isolated on {tgt}.")
                reward = clamp_score(1.0 - ((_GLOBAL_CACHE["state"].step_count - 1) * hard_pen))
                done = True

        elif task == "t17":
            if cmd == "RESTART_GPU_DAEMON":
                f_idx = int(fnode.split("_")[1]) // 10
                if tgt == f"sub_{f_idx}":
                    logs.append(f"SUCCESS: GPU daemon restarted on {tgt}.")
                    reward = clamp_score(1.0 - ((_GLOBAL_CACHE["state"].step_count - 1) * hard_pen))
                    done = True

        elif task == "t18":
            if cmd == "ISSUE_GLOBAL_ROLLBACK" and tgt == "cluster_0":
                logs.append("SUCCESS: Deadlock broken via global rollback.")
                reward = clamp_score(1.0 - ((_GLOBAL_CACHE["state"].step_count - 1) * hard_pen))
                done = True

        elif task == "t19":
            if cmd == "REBOOT_LEAF_SWITCHES" and tgt == "pod_x":
                logs.append("SUCCESS: Leaf switches rebooted, partition resolved.")
                reward = clamp_score(1.0 - ((_GLOBAL_CACHE["state"].step_count - 1) * hard_pen))
                done = True

        elif task == "t20":
            if cmd == "PURGE_CORRUPT_BLOCK" and tgt == "db_cluster":
                logs.append("SUCCESS: Corrupt block purged.")
                reward = clamp_score(1.0 - ((_GLOBAL_CACHE["state"].step_count - 1) * hard_pen))
                done = True

        if _GLOBAL_CACHE["state"].step_count >= self.MAX_ATTEMPTS and not done:
            logs.append("SLA BREACH: Timeout limit reached.")
            done = True
            reward = clamp_score(0.01)

        _GLOBAL_CACHE["last_grader_score"] = float(reward)
        return self._get_obs(done, reward)

    @property
    def state(self) -> State:
        return _GLOBAL_CACHE["state"]

    def _get_obs(self, done, reward) -> NetweaverSreObservation:
        global _GLOBAL_CACHE
        return NetweaverSreObservation(
            done=done,
            reward=float(reward),
            step_count=_GLOBAL_CACHE["state"].step_count,
            queue_depths=_GLOBAL_CACHE["queue_depths"].copy(),
            gradient_variances=_GLOBAL_CACHE["gradient_vars"].copy(),
            gpu_memory_usage=_GLOBAL_CACHE["gpu_memory"].copy(),
            hardware_logs=_GLOBAL_CACHE["logs"][-5:],
            system_health=1.0 if not done and _GLOBAL_CACHE['active_task'] != 't20' else 0.5,
        )

    def grader(self, *args, **kwargs) -> float:
        global _GLOBAL_CACHE
        return float(_GLOBAL_CACHE.get("last_grader_score", 0.001))

