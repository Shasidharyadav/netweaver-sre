# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
"""
NetWeaver SRE Environment Implementation.

Single source of truth for fault scenarios, randomized targets,
and per-step validation. The FastAPI server in `server/app.py`
delegates `/reset` and `/step` to this class — there is no
parallel `FAULT_SCENARIOS` dictionary anymore.

All reported scores are clamped strictly to (0.001, 0.999).
"""

import os
import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import NetweaverSreAction, NetweaverSreObservation
except ImportError:
    from models import NetweaverSreAction, NetweaverSreObservation


# ── Forced-task control (used by /set_level) ────────────────────────────────
_FORCED_TASK_LEVEL: str = ""


def set_task_level(level: str) -> None:
    """Force the next reset() to use a specific task level (e.g. 't01' or 'easy')."""
    global _FORCED_TASK_LEVEL
    _FORCED_TASK_LEVEL = (level or "").lower().strip()


def clear_task_level() -> None:
    global _FORCED_TASK_LEVEL
    _FORCED_TASK_LEVEL = ""


def clamp_score(raw: float) -> float:
    score = float(raw) if raw is not None else 0.001
    return max(0.001, min(0.999, score))


# ── Per-task metadata ────────────────────────────────────────────────────────
TASK_FAULT_TYPES = {
    "t01": "node_offline",      "t02": "dns_cache",
    "t03": "oom_crash",         "t04": "tls_expiry",
    "t05": "disk_full",         "t06": "unhealthy_pod",
    "t07": "zombie_process",    "t08": "pfc_congestion",
    "t09": "power_throttle",    "t10": "bgp_flap",
    "t11": "packet_drop",       "t12": "ddos",
    "t13": "conn_exhaustion",   "t14": "cpu_context_switch",
    "t15": "nan_contagion",     "t16": "broadcast_storm",
    "t17": "gpu_memory_leak",   "t18": "cluster_deadlock",
    "t19": "network_partition", "t20": "corrupt_db",
    "t21": "cascading_failure", "t22": "gradient_poisoning",
}

ALL_TASKS = [f"t{i:02d}" for i in range(1, 23)]
EASY_TASKS = ALL_TASKS[:7]    # t01–t07
MEDIUM_TASKS = ALL_TASKS[7:14]  # t08–t14
HARD_TASKS = ALL_TASKS[14:]   # t15–t22


# ── In-memory single-session state ───────────────────────────────────────────
_GLOBAL_CACHE = {
    "state": State(episode_id=str(uuid4()), step_count=0),
    "active_task": "t01",
    "fault_type": "node_offline",
    "alert": "",
    "queue_depths": {},
    "gradient_vars": [0.01] * 10,
    "gpu_memory": [0.70] * 6,
    "logs": [],
    # Randomized per-task targets
    "faulty_node_id": "",
    "target_val": 0,
    "storm_switch": "",
    "deadlock_cluster": "",
    "partition_pod": "",
    "db_target": "",
    "gpu_leak_idx": 0,
    "router_id": "",
    "gateway_id": "",
    "as_number": 0,
    "service_name": "",
    "pod_name": "",
    # Multi-step progress
    "t15_progress": set(),
    "t21_progress": set(),
    "t22_progress": set(),
    "t21_node": "",
    "t21_db": "",
    "t22_cluster": "",
    # Outcome tracking
    "last_grader_score": 0.001,
    "system_health": 1.0,
    "is_done": False,
}


def _pick(choices):
    return random.choice(choices)


def _new_node():
    return f"node_{random.randint(0, 99):02d}"


def _new_pod():
    return _pick(["pod_a", "pod_b", "pod_c", "pod_d", "pod_e"])


def _new_cluster():
    return _pick(["cluster_0", "cluster_1", "cluster_2", "cluster_3", "cluster_4"])


def _new_db():
    return _pick(["db_cluster_0", "db_cluster_1", "db_cluster_2", "db_cluster_3"])


def _new_switch():
    return _pick([
        "sw_core_01", "sw_core_02", "sw_spine_01", "sw_spine_02",
        "sw_leaf_01", "sw_leaf_02", "sw_leaf_03", "sw_leaf_04",
    ])


def _new_router():
    return _pick(["router_spine_01", "router_spine_02", "router_leaf_01"])


def _new_gateway():
    return _pick(["api_gateway_01", "api_gateway_02", "edge_gateway_01"])


def _new_service():
    return _pick([
        "training_coordinator", "metrics_exporter", "checkpoint_manager",
        "scheduler_daemon", "rank_broker",
    ])


class NetweaverSreEnvironment(Environment):
    """OpenEnv Environment for autonomous SRE on a 100-node GPU cluster."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = False
    MAX_ATTEMPTS: int = 15

    # ── Public helpers ───────────────────────────────────────────────────────

    @property
    def task_id(self) -> str:
        return f"netweaver_sre_{_GLOBAL_CACHE['active_task']}"

    @property
    def fault_type(self) -> str:
        return _GLOBAL_CACHE["fault_type"]

    @property
    def active_task(self) -> str:
        return _GLOBAL_CACHE["active_task"]

    @property
    def is_done(self) -> bool:
        return bool(_GLOBAL_CACHE.get("is_done", False))

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def reset(self, **kwargs) -> NetweaverSreObservation:
        global _FORCED_TASK_LEVEL
        _GLOBAL_CACHE["state"] = State(episode_id=str(uuid4()), step_count=0)
        _GLOBAL_CACHE["is_done"] = False
        _GLOBAL_CACHE["last_grader_score"] = 0.001

        # Pick task: explicit kwarg > set_task_level > FORCE_TASK_LEVEL env > random
        forced = (kwargs.get("task_level") or "").lower().strip()
        if not forced:
            forced = _FORCED_TASK_LEVEL or os.getenv("FORCE_TASK_LEVEL", "").lower().strip()

        if forced in ALL_TASKS:
            task = forced
        elif forced == "easy":
            task = _pick(EASY_TASKS)
        elif forced == "medium":
            task = _pick(MEDIUM_TASKS)
        elif forced == "hard":
            task = _pick(HARD_TASKS)
        else:
            task = _pick(ALL_TASKS)

        _GLOBAL_CACHE["active_task"] = task
        _GLOBAL_CACHE["fault_type"] = TASK_FAULT_TYPES[task]

        # Randomized entities (re-rolled every reset)
        fnode = _new_node()
        _GLOBAL_CACHE["faulty_node_id"] = fnode
        _GLOBAL_CACHE["service_name"] = _new_service()
        _GLOBAL_CACHE["pod_name"] = _new_pod()
        _GLOBAL_CACHE["storm_switch"] = _new_switch()
        _GLOBAL_CACHE["deadlock_cluster"] = _new_cluster()
        _GLOBAL_CACHE["partition_pod"] = _new_pod()
        _GLOBAL_CACHE["db_target"] = _new_db()
        _GLOBAL_CACHE["router_id"] = _new_router()
        _GLOBAL_CACHE["gateway_id"] = _new_gateway()
        _GLOBAL_CACHE["as_number"] = random.randint(64512, 65535)
        _GLOBAL_CACHE["gpu_leak_idx"] = random.randint(0, 5)

        _GLOBAL_CACHE["t15_progress"] = set()
        _GLOBAL_CACHE["t21_progress"] = set()
        _GLOBAL_CACHE["t22_progress"] = set()
        _GLOBAL_CACHE["t21_node"] = fnode
        _GLOBAL_CACHE["t21_db"] = _new_db()
        _GLOBAL_CACHE["t22_cluster"] = _new_cluster()

        # Numeric value targets (per-task constraints)
        if task == "t08":
            tval = random.choice([2048, 4096, 6144, 8192])  # PFC threshold (1000-9000)
        elif task == "t09":
            tval = random.choice([250, 300, 350, 400])      # power cap watts (100-400)
        elif task == "t10":
            tval = _GLOBAL_CACHE["as_number"]               # AS number
        elif task == "t11":
            tval = 9000                                     # MTU (bug 3 fix — always 9000)
        elif task == "t12":
            tval = random.choice([100, 500, 1000, 5000])    # rate limit req/s
        elif task == "t13":
            tval = random.choice([200, 400, 800, 1600])     # pool size
        elif task == "t14":
            tval = random.choice([32, 48, 64, 96, 128])     # CPU thread count
        else:
            tval = random.randint(100, 900)
        _GLOBAL_CACHE["target_val"] = tval

        # Reset telemetry to healthy baseline
        _GLOBAL_CACHE["queue_depths"] = {
            "sw_core_01": round(random.uniform(8.0, 14.0), 1),
            "sw_core_02": round(random.uniform(8.0, 14.0), 1),
            "sw_spine_01": round(random.uniform(8.0, 14.0), 1),
            "sw_leaf_04": round(random.uniform(8.0, 14.0), 1),
        }
        _GLOBAL_CACHE["gradient_vars"] = [round(random.uniform(0.005, 0.03), 3) for _ in range(10)]
        _GLOBAL_CACHE["gpu_memory"] = [round(random.uniform(0.65, 0.78), 2) for _ in range(6)]
        _GLOBAL_CACHE["system_health"] = 1.0

        init_log = f"CLUSTER INIT: Clos topology active. Running task {task.upper()} ({_GLOBAL_CACHE['fault_type']})."
        _GLOBAL_CACHE["logs"] = [init_log]

        # ── Inject the fault narrative (alert + telemetry) ──────────────────
        alert, extra_logs = self._build_fault(task, fnode, tval)
        _GLOBAL_CACHE["alert"] = alert
        _GLOBAL_CACHE["logs"].extend(extra_logs)

        return self._get_obs(done=False, reward=clamp_score(0.01))

    # ── Fault narrative builder ──────────────────────────────────────────────
    def _build_fault(self, task: str, fnode: str, tval: int):
        c = _GLOBAL_CACHE
        if task == "t01":
            return (
                f"CRITICAL: GPU node {fnode} has gone offline. Training throughput dropped 12%. Isolate immediately.",
                [
                    f"{fnode}: heartbeat timeout after 30s",
                    f"{fnode}: NIC link down detected on eth0",
                    f"{fnode}: removed from training ring by watchdog. Issue DRAIN_TRAFFIC on {fnode}.",
                ],
            )
        if task == "t02":
            return (
                f"ERROR: DNS resolution failures on {fnode}. Service discovery broken.",
                [
                    f"{fnode}: DNS SERVFAIL for service.internal (cached entry stale)",
                    f"{fnode}: 412 DNS timeouts in last 60s. Issue CLEAR_DNS_CACHE on {fnode}.",
                ],
            )
        if task == "t03":
            svc = c["service_name"]
            return (
                f"CRITICAL: {svc} service crashed with OOM on {fnode}. Restart required.",
                [
                    f"{fnode}: {svc} killed by OOM killer (rss=48GB, limit=32GB)",
                    f"{fnode}: service {svc} state=crashed. Issue RESTART_SERVICE on {svc}.",
                ],
            )
        if task == "t04":
            return (
                f"ERROR: mTLS handshake failures on {fnode}. Certificate expired.",
                [
                    f"{fnode}: TLS handshake failed: certificate has expired",
                    f"{fnode}: 1,204 connection rejections in last 5min. Issue RENEW_CERTIFICATE on {fnode}.",
                ],
            )
        if task == "t05":
            return (
                f"CRITICAL: Disk on {fnode} at 100% capacity. Checkpoint saves failing.",
                [
                    f"{fnode}: /dev/nvme0n1 usage=100% (2.0TB/2.0TB)",
                    f"{fnode}: /tmp 98% full with stale core dumps. Issue CLEAR_TEMP_FILES on {fnode}.",
                ],
            )
        if task == "t06":
            pod = c["pod_name"]
            return (
                f"WARNING: Kubernetes pod {pod} stuck in CrashLoopBackOff on {fnode}.",
                [
                    f"{fnode}: pod {pod} CrashLoopBackOff (restarts=18)",
                    f"{fnode}: pod {pod} not ready for 12 minutes. Issue RESTART_POD on {pod}.",
                ],
            )
        if task == "t07":
            return (
                f"WARNING: Zombie processes accumulating on {fnode}. PID table filling up.",
                [
                    f"{fnode}: zombie process count=143 (ppid=1, state=Z)",
                    f"{fnode}: PID table 91% full. Issue KILL_ZOMBIE_PROCESS on {fnode}.",
                ],
            )
        if task == "t08":
            sw = c["storm_switch"]
            c["queue_depths"][sw] = 97.3
            return (
                f"WARNING: PFC buffer congestion on {sw}. Packet loss on RDMA traffic.",
                [
                    f"{sw}: PFC PAUSE frames excessive on port 24",
                    f"{sw}: buffer utilization 97.3%. Issue TUNE_PFC_THRESHOLD on {sw} with value {tval}.",
                ],
            )
        if task == "t09":
            return (
                f"WARNING: {fnode} throttling due to power cap. GPU compute reduced 40%.",
                [
                    f"{fnode}: power cap hit: current=320W, limit=250W, throttling active",
                    f"{fnode}: GPU clock reduced 1800MHz -> 1100MHz. Issue ADJUST_POWER_CAP on {fnode} with value {tval}.",
                ],
            )
        if task == "t10":
            router = c["router_id"]
            asn = c["as_number"]
            return (
                f"CRITICAL: BGP session flapping on {router}. Routes withdrawn every 8s.",
                [
                    f"{router}: BGP session to AS{asn} flapping (up/down 47x in 10min)",
                    f"{router}: hold timer expired for peer AS{asn}. Issue MITIGATE_ROUTE_FLAP on {router} with value {asn}.",
                ],
            )
        if task == "t11":
            sw = c["storm_switch"]
            c["queue_depths"][sw] = 34.5
            return (
                f"ERROR: Jumbo frame packet drops on {sw}. RDMA throughput degraded 60%.",
                [
                    f"{sw}: MTU mismatch — interface MTU=1500, jumbo frames=9000",
                    f"{sw}: dropping 12,400 packets/s. Issue INCREASE_MTU on {sw} with value 9000.",
                ],
            )
        if task == "t12":
            gw = c["gateway_id"]
            c["queue_depths"][gw] = 99.1
            return (
                f"CRITICAL: DDoS detected on {gw}. 480,000 req/s from spoofed IPs.",
                [
                    f"{gw}: request rate 480,000 req/s (normal: 1,200 req/s)",
                    f"{gw}: connection queue saturated. Issue SET_RATE_LIMIT on {gw} with value {tval}.",
                ],
            )
        if task == "t13":
            db = c["db_target"]
            c["queue_depths"][db] = 98.7
            return (
                f"CRITICAL: Database connection pool exhausted on {db}. All 100 connections in use.",
                [
                    f"{db}: connection pool exhausted (100/100 active)",
                    f"{db}: 847 connection requests queued. Issue SCALE_CONN_POOL on {db} with value {tval}.",
                ],
            )
        if task == "t14":
            return (
                f"WARNING: Excessive CPU context switches on {fnode}. Step time increased 3x.",
                [
                    f"{fnode}: context switches 2,400,000/s (normal: 50,000/s)",
                    f"{fnode}: 128 threads competing for 64 cores. Issue PIN_CPU_THREADS on {fnode} with value {tval}.",
                ],
            )
        if task == "t15":
            cluster = c["deadlock_cluster"]  # reuse cluster pool
            c["t21_node"] = cluster  # ensure target consistency below isn't needed
            # Pick a NaN rank index
            nan_idx = random.randint(0, 9)
            c["gradient_vars"][nan_idx] = -1.0
            return (
                f"CRITICAL: Silent NaN contagion in gradient sync. Rank {nan_idx} corrupted on {cluster}.",
                [
                    f"{cluster}: gradient sync anomaly detected rank={nan_idx}",
                    f"{cluster}: NaN propagating. Sequence: RUN_MINI_ITERATION on {cluster}, then DRAIN_TRAFFIC on {cluster}.",
                ],
            )
        if task == "t16":
            sw = c["storm_switch"]
            c["queue_depths"][sw] = 99.4
            return (
                f"CRITICAL: Broadcast storm detected on {sw}.",
                [
                    f"{sw}: broadcast flood detected on VLAN 100",
                    f"{sw}: 98% of bandwidth consumed by broadcast frames. Issue ISOLATE_BROADCAST_STORM on {sw}.",
                ],
            )
        if task == "t17":
            cluster = c["deadlock_cluster"]
            idx = c["gpu_leak_idx"]
            c["gpu_memory"][idx] = 0.97
            return (
                f"WARNING: GPU memory leak on {cluster}. Memory climbing; OOM imminent.",
                [
                    f"{cluster}: GPU memory usage increasing 2% per minute on rank {idx}",
                    f"{cluster}: memory fragmentation in CUDA allocator. Issue RESTART_GPU_DAEMON on {cluster}.",
                ],
            )
        if task == "t18":
            cluster = c["deadlock_cluster"]
            c["gradient_vars"] = [0.0] * 10
            c["queue_depths"] = {k: 0.0 for k in c["queue_depths"]}
            c["gpu_memory"] = [0.0] * 6
            c["system_health"] = 0.05
            return (
                f"CRITICAL: Full cluster deadlock on {cluster}. All subsystems unresponsive.",
                [
                    f"{cluster}: watchdog timeout — no heartbeat for 120s",
                    f"{cluster}: deadlock detected across all ranks. Issue ISSUE_GLOBAL_ROLLBACK on {cluster}.",
                ],
            )
        if task == "t19":
            pod = c["partition_pod"]
            healthy_sw = "sw_leaf_01"
            broken_sw = "sw_leaf_04"
            c["queue_depths"] = {healthy_sw: 0.01, broken_sw: 99.9}
            c["system_health"] = 0.40
            return (
                f"CRITICAL: Network partition detected. {pod} cannot communicate with peers.",
                [
                    f"{pod}: cannot reach peers (packet loss 100%)",
                    f"{pod}: leaf switch link down. Issue REBOOT_LEAF_SWITCHES on {pod}.",
                ],
            )
        if task == "t20":
            db = c["db_target"]  # always starts with db_cluster_*
            c["system_health"] = 0.55
            return (
                f"CRITICAL: Corrupt block detected on {db}. Checkpoint health degrading.",
                [
                    f"{db}: block checksum mismatch at offset 0x3A4F000",
                    f"{db}: storage controller reports I/O error. Issue PURGE_CORRUPT_BLOCK on {db}.",
                ],
            )
        if task == "t21":
            node = c["t21_node"]
            db = c["t21_db"]
            return (
                f"CRITICAL: Cascading fault chain on {node}. Power throttle -> CPU storm -> exhausted DB pool ({db}).",
                [
                    f"{node}: power cap throttle active, clocks reduced 43%",
                    f"{node}: context switches 2,900,000/s after throttle event",
                    f"{db}: connection pool exhausted (400/400)",
                    (
                        f"REMEDIATION ORDER: (1) ADJUST_POWER_CAP on {node} value 350, "
                        f"(2) PIN_CPU_THREADS on {node} value 64, "
                        f"(3) SCALE_CONN_POOL on {db} value 800."
                    ),
                ],
            )
        if task == "t22":
            cluster = c["t22_cluster"]
            sw = c["storm_switch"]
            nan_idx = random.randint(0, 9)
            c["gradient_vars"][nan_idx] = 999.9
            c["queue_depths"][sw] = 99.6
            return (
                f"CRITICAL: Gradient poisoning on {cluster} amplified by broadcast storm on {sw}.",
                [
                    f"{cluster}: NaN source in rank {nan_idx}",
                    f"{sw}: broadcast flood >97% fabric utilization",
                    (
                        f"REMEDIATION ORDER: (1) RUN_MINI_ITERATION on {cluster}, "
                        f"(2) ISOLATE_BROADCAST_STORM on {sw}, "
                        f"(3) DRAIN_TRAFFIC on {cluster}."
                    ),
                ],
            )
        return ("", [])

    # ── Step ─────────────────────────────────────────────────────────────────
    def step(self, action: NetweaverSreAction) -> NetweaverSreObservation:
        c = _GLOBAL_CACHE
        c["state"].step_count += 1

        cmd = (action.command or "").upper()
        tgt = (action.target or "")
        val = action.value

        done = False
        reward = clamp_score(0.05)
        task = c["active_task"]
        fnode = c["faulty_node_id"]
        tval = c["target_val"]
        logs = c["logs"]
        steps = c["state"].step_count

        step_pen = 0.02
        hard_pen = 0.04

        easy_map = {
            "t01": ("DRAIN_TRAFFIC", fnode),
            "t02": ("CLEAR_DNS_CACHE", fnode),
            "t03": ("RESTART_SERVICE", c["service_name"]),
            "t04": ("RENEW_CERTIFICATE", fnode),
            "t05": ("CLEAR_TEMP_FILES", fnode),
            "t06": ("RESTART_POD", c["pod_name"]),
            "t07": ("KILL_ZOMBIE_PROCESS", fnode),
        }
        med_map = {
            "t08": ("TUNE_PFC_THRESHOLD", c["storm_switch"]),
            "t09": ("ADJUST_POWER_CAP", fnode),
            "t10": ("MITIGATE_ROUTE_FLAP", c["router_id"]),
            "t11": ("INCREASE_MTU", c["storm_switch"]),
            "t12": ("SET_RATE_LIMIT", c["gateway_id"]),
            "t13": ("SCALE_CONN_POOL", c["db_target"]),
            "t14": ("PIN_CPU_THREADS", fnode),
        }

        # ── EASY ────────────────────────────────────────────────────────────
        if task in easy_map:
            req_cmd, req_tgt = easy_map[task]
            if cmd == req_cmd:
                if tgt and tgt == req_tgt:
                    logs.append(f"SUCCESS: Executed {cmd} on {tgt}.")
                    reward = clamp_score(1.0 - (steps - 1) * step_pen)
                    done = True
                else:
                    logs.append(f"PARTIAL: Correct command {cmd}, but target '{tgt}' != expected.")
                    reward = clamp_score(0.40)
            else:
                logs.append(f"ERROR: Wrong command {cmd}. Expected {req_cmd}.")

        # ── MEDIUM ──────────────────────────────────────────────────────────
        elif task in med_map:
            req_cmd, req_tgt = med_map[task]
            if cmd == req_cmd:
                target_ok = bool(tgt) and tgt == req_tgt
                value_ok = (val is not None) and (int(val) == int(tval))
                if target_ok and value_ok:
                    logs.append(f"SUCCESS: {cmd} on {tgt} value={val}.")
                    reward = clamp_score(1.0 - (steps - 1) * step_pen)
                    done = True
                elif target_ok and val is not None:
                    logs.append(f"PARTIAL: Correct command+target. Value {val} != expected {tval}.")
                    reward = clamp_score(0.55)
                elif target_ok:
                    logs.append(f"ERROR: Value parameter required for {cmd}.")
                    reward = clamp_score(0.20)
                else:
                    logs.append(f"PARTIAL: Correct command but target '{tgt}' != expected {req_tgt}.")
                    reward = clamp_score(0.30)
            else:
                logs.append(f"ERROR: Wrong command {cmd}. Expected {req_cmd}.")

        # ── HARD: t15 (multi-step NaN contagion) ────────────────────────────
        elif task == "t15":
            cluster = c["deadlock_cluster"]
            progress = c["t15_progress"]
            if cmd == "RUN_MINI_ITERATION" and tgt == cluster and "triage" not in progress:
                progress.add("triage")
                logs.append(f"STAGE 1/2: NaN source isolated on {cluster}.")
                reward = clamp_score(0.45)
            elif cmd == "DRAIN_TRAFFIC" and tgt == cluster and "triage" in progress and "drain" not in progress:
                progress.add("drain")
                logs.append(f"STAGE 2/2: Traffic drained from {cluster}.")
                reward = clamp_score(1.0 - (steps - 2) * hard_pen)
                done = True
            elif cmd == "RUN_MINI_ITERATION":
                logs.append(f"ERROR: Wrong RUN_MINI_ITERATION target. Expected {cluster}, got {tgt}.")
            elif cmd == "DRAIN_TRAFFIC":
                if "triage" not in progress:
                    logs.append("ERROR: DRAIN_TRAFFIC issued before RUN_MINI_ITERATION triage.")
                else:
                    logs.append(f"ERROR: Wrong DRAIN_TRAFFIC target. Expected {cluster}, got {tgt}.")
            else:
                logs.append(f"ERROR: {cmd} not part of t15 remediation.")

        elif task == "t16":
            sw = c["storm_switch"]
            if cmd == "ISOLATE_BROADCAST_STORM" and tgt == sw:
                logs.append(f"SUCCESS: Broadcast storm isolated on {sw}.")
                reward = clamp_score(1.0 - (steps - 1) * hard_pen)
                done = True
            elif cmd == "ISOLATE_BROADCAST_STORM":
                logs.append(f"ERROR: Wrong target. Expected {sw}, got {tgt}.")
                reward = clamp_score(0.30)
            else:
                logs.append(f"ERROR: Wrong command for t16. Expected ISOLATE_BROADCAST_STORM.")

        elif task == "t17":
            cluster = c["deadlock_cluster"]
            if cmd == "RESTART_GPU_DAEMON" and tgt == cluster:
                logs.append(f"SUCCESS: GPU daemon restarted on {cluster}.")
                reward = clamp_score(1.0 - (steps - 1) * hard_pen)
                done = True
            elif cmd == "RESTART_GPU_DAEMON":
                logs.append(f"ERROR: Wrong target. Expected {cluster}, got {tgt}.")
                reward = clamp_score(0.30)
            else:
                logs.append("ERROR: Wrong command for t17. Expected RESTART_GPU_DAEMON.")

        elif task == "t18":
            cluster = c["deadlock_cluster"]
            if cmd == "ISSUE_GLOBAL_ROLLBACK" and tgt == cluster:
                logs.append(f"SUCCESS: Deadlock broken on {cluster} via global rollback.")
                reward = clamp_score(1.0 - (steps - 1) * hard_pen)
                done = True
            elif cmd == "ISSUE_GLOBAL_ROLLBACK":
                logs.append(f"ERROR: Wrong target. Expected {cluster}, got {tgt}.")
                reward = clamp_score(0.30)
            else:
                logs.append("ERROR: Wrong command for t18. Expected ISSUE_GLOBAL_ROLLBACK.")

        elif task == "t19":
            pod = c["partition_pod"]
            if cmd == "REBOOT_LEAF_SWITCHES" and tgt == pod:
                logs.append(f"SUCCESS: Leaf switches rebooted, partition {pod} resolved.")
                reward = clamp_score(1.0 - (steps - 1) * hard_pen)
                done = True
            elif cmd == "REBOOT_LEAF_SWITCHES":
                logs.append(f"ERROR: Wrong target. Expected {pod}, got {tgt}.")
                reward = clamp_score(0.30)
            else:
                logs.append("ERROR: Wrong command for t19. Expected REBOOT_LEAF_SWITCHES.")

        elif task == "t20":
            db = c["db_target"]
            if cmd == "PURGE_CORRUPT_BLOCK" and tgt == db:
                logs.append(f"SUCCESS: Corrupt block purged on {db}.")
                reward = clamp_score(1.0 - (steps - 1) * hard_pen)
                done = True
            elif cmd == "PURGE_CORRUPT_BLOCK":
                logs.append(f"ERROR: Wrong target. Expected {db}, got {tgt}.")
                reward = clamp_score(0.30)
            else:
                logs.append("ERROR: Wrong command for t20. Expected PURGE_CORRUPT_BLOCK.")

        elif task == "t21":
            node = c["t21_node"]
            db = c["t21_db"]
            progress = c["t21_progress"]

            if cmd == "ADJUST_POWER_CAP" and tgt == node and "power" not in progress:
                progress.add("power")
                logs.append(f"STAGE 1/3: Power cap adjusted on {node}.")
                reward = clamp_score(0.30)
            elif cmd == "PIN_CPU_THREADS" and tgt == node and "power" in progress and "cpu" not in progress:
                progress.add("cpu")
                logs.append(f"STAGE 2/3: CPU threads pinned on {node}.")
                reward = clamp_score(0.55)
            elif cmd == "SCALE_CONN_POOL" and tgt == db and "cpu" in progress and "db" not in progress:
                progress.add("db")
                logs.append(f"STAGE 3/3: Connection pool scaled on {db}.")
                reward = clamp_score(1.0 - (steps - 3) * hard_pen)
                done = True
            else:
                if cmd == "PIN_CPU_THREADS" and "power" not in progress:
                    logs.append("ERROR: PIN_CPU_THREADS issued before ADJUST_POWER_CAP. Wrong order.")
                elif cmd == "SCALE_CONN_POOL" and "cpu" not in progress:
                    logs.append("ERROR: SCALE_CONN_POOL issued before PIN_CPU_THREADS. Wrong order.")
                elif cmd in {"ADJUST_POWER_CAP", "PIN_CPU_THREADS", "SCALE_CONN_POOL"}:
                    logs.append(f"ERROR: Wrong target for {cmd}. Check the alert.")
                else:
                    logs.append(f"ERROR: {cmd} not part of t21 remediation.")
                reward = clamp_score(0.10)

        elif task == "t22":
            cluster = c["t22_cluster"]
            sw = c["storm_switch"]
            progress = c["t22_progress"]

            if cmd == "RUN_MINI_ITERATION" and tgt == cluster and "triage" not in progress:
                progress.add("triage")
                logs.append(f"STAGE 1/3: NaN source identified on {cluster}.")
                reward = clamp_score(0.30)
            elif cmd == "ISOLATE_BROADCAST_STORM" and tgt == sw and "triage" in progress and "storm" not in progress:
                progress.add("storm")
                logs.append(f"STAGE 2/3: Broadcast storm isolated on {sw}.")
                reward = clamp_score(0.55)
            elif cmd == "DRAIN_TRAFFIC" and tgt == cluster and "storm" in progress and "drain" not in progress:
                progress.add("drain")
                logs.append(f"STAGE 3/3: Traffic drained from {cluster}.")
                reward = clamp_score(1.0 - (steps - 3) * hard_pen)
                done = True
            else:
                if cmd == "ISOLATE_BROADCAST_STORM" and "triage" not in progress:
                    logs.append("ERROR: ISOLATE_BROADCAST_STORM before RUN_MINI_ITERATION. Wrong order.")
                elif cmd == "DRAIN_TRAFFIC" and "storm" not in progress:
                    logs.append("ERROR: DRAIN_TRAFFIC before ISOLATE_BROADCAST_STORM. Wrong order.")
                elif cmd in {"RUN_MINI_ITERATION", "ISOLATE_BROADCAST_STORM", "DRAIN_TRAFFIC"}:
                    logs.append(f"ERROR: Wrong target for {cmd}. Check the alert.")
                else:
                    logs.append(f"ERROR: {cmd} not part of t22 remediation.")
                reward = clamp_score(0.10)

        # ── SLA timeout ─────────────────────────────────────────────────────
        if c["state"].step_count >= self.MAX_ATTEMPTS and not done:
            logs.append("SLA BREACH: Timeout limit reached.")
            done = True
            reward = clamp_score(0.05)

        # On success of single-step tasks, set health to 1.0 (already healthy)
        if done and task in (set(easy_map) | set(med_map) | {"t15", "t16", "t17", "t19", "t21", "t22"}):
            c["system_health"] = 1.0
        elif done and task == "t18":
            c["system_health"] = 1.0
        elif done and task == "t20":
            c["system_health"] = 0.85

        c["last_grader_score"] = float(reward)
        c["is_done"] = bool(done)
        return self._get_obs(done, reward)

    # ── Observation ──────────────────────────────────────────────────────────
    @property
    def state(self) -> State:
        return _GLOBAL_CACHE["state"]

    def _get_obs(self, done, reward) -> NetweaverSreObservation:
        c = _GLOBAL_CACHE
        return NetweaverSreObservation(
            done=bool(done),
            reward=clamp_score(reward),
            step_count=c["state"].step_count,
            alert=c.get("alert", ""),
            queue_depths=dict(c["queue_depths"]),
            gradient_variances=list(c["gradient_vars"]),
            gpu_memory_usage=list(c["gpu_memory"]),
            hardware_logs=list(c["logs"][-6:]),
            system_health=float(c.get("system_health", 1.0)),
            active_connections=0,
            error_rate=0.0,
        )

    def grader(self, *args, **kwargs) -> float:
        return clamp_score(_GLOBAL_CACHE.get("last_grader_score", 0.001))
