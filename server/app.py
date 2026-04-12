# server/app.py
# NetWeaver SRE — FastAPI environment server
# Integrates per-task graders and per-step reward shaping

import random
import json
import os
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import our new modules (now in root)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graders import compute_grader_score, GRADER_CONFIG
from reward_shaper import compute_step_reward, record_obs_access, DESTRUCTIVE_COMMANDS
from models import NetweaverSreGraderResponse

app = FastAPI(title="NetWeaver SRE", version="2.0.0")

# ── Static files (playground UI) ─────────────────────────────────────────────
_assets_dir = os.path.join(os.path.dirname(__file__), "assets")
if os.path.isdir(_assets_dir):
    app.mount("/assets", StaticFiles(directory=_assets_dir), name="assets")

# ── Episode state (in-memory, single-session) ─────────────────────────────────
SESSION: Dict[str, Any] = {}

# ── Fault scenarios — one per task ───────────────────────────────────────────
FAULT_SCENARIOS = {
    "t01": {
        "task_id": "netweaver_sre_t01", "fault_type": "node_offline",
        "alert": "CRITICAL: GPU node node_07 has gone offline. Training throughput dropped 12%. Isolate immediately.",
        "hardware_logs": [
            "node_07: heartbeat timeout after 30s",
            "node_07: NIC link down detected on eth0",
            "node_07: removed from training ring by watchdog",
        ],
        "queue_depths": {"switch_a": 12.3, "switch_b": 8.1},
        "gradient_variances": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        "gpu_memory_usage": [0.72, 0.71, 0.73, 0.0, 0.72, 0.71],
        "system_health": 0.88,
    },
    "t02": {
        "task_id": "netweaver_sre_t02", "fault_type": "dns_cache",
        "alert": "ERROR: DNS resolution failures on node_12. Service discovery broken. Pods cannot reach each other.",
        "hardware_logs": [
            "node_12: DNS SERVFAIL for service.internal (cached entry stale)",
            "node_12: /etc/resolv.conf points to 10.0.0.1 but cache poisoned",
            "node_12: 412 DNS timeouts in last 60s",
        ],
        "queue_depths": {"switch_a": 5.0, "switch_b": 4.8},
        "gradient_variances": [0.02]*10,
        "gpu_memory_usage": [0.70]*6,
        "system_health": 0.91,
    },
    "t03": {
        "task_id": "netweaver_sre_t03", "fault_type": "oom_crash",
        "alert": "CRITICAL: training_coordinator service crashed with OOM on node_04. Restart required.",
        "hardware_logs": [
            "node_04: training_coordinator killed by OOM killer (rss=48GB, limit=32GB)",
            "node_04: oom_score_adj=500 triggered at 03:14:22",
            "node_04: service training_coordinator state=crashed",
        ],
        "queue_depths": {"switch_a": 3.1, "switch_b": 2.9},
        "gradient_variances": [0.01]*10,
        "gpu_memory_usage": [0.68]*6,
        "system_health": 0.85,
    },
    "t04": {
        "task_id": "netweaver_sre_t04", "fault_type": "tls_expiry",
        "alert": "ERROR: mTLS handshake failures on node_19. Certificate expired 2 days ago.",
        "hardware_logs": [
            "node_19: TLS handshake failed: certificate has expired (notAfter=Apr 10 2026)",
            "node_19: peer rejected connection, error=SSL_ERROR_RX_RECORD_TOO_LONG",
            "node_19: 1,204 connection rejections in last 5min",
        ],
        "queue_depths": {"switch_a": 4.2, "switch_b": 3.8},
        "gradient_variances": [0.02]*10,
        "gpu_memory_usage": [0.69]*6,
        "system_health": 0.87,
    },
    "t05": {
        "task_id": "netweaver_sre_t05", "fault_type": "disk_full",
        "alert": "CRITICAL: Disk on node_22 at 100% capacity. Checkpoint saves failing.",
        "hardware_logs": [
            "node_22: /dev/nvme0n1 usage=100% (2.0TB/2.0TB)",
            "node_22: checkpoint save failed: No space left on device",
            "node_22: /tmp is 98% full with stale core dumps",
        ],
        "queue_depths": {"switch_a": 6.0, "switch_b": 5.5},
        "gradient_variances": [0.01]*10,
        "gpu_memory_usage": [0.71]*6,
        "system_health": 0.80,
    },
    "t06": {
        "task_id": "netweaver_sre_t06", "fault_type": "unhealthy_pod",
        "alert": "WARNING: Kubernetes pod metrics-exporter-9bxlk stuck in CrashLoopBackOff on node_03.",
        "hardware_logs": [
            "node_03: pod metrics-exporter-9bxlk CrashLoopBackOff (restarts=18)",
            "node_03: pod last exit code=137 (OOMKilled)",
            "node_03: pod metrics-exporter-9bxlk not ready for 12 minutes",
        ],
        "queue_depths": {"switch_a": 7.1, "switch_b": 6.9},
        "gradient_variances": [0.02]*10,
        "gpu_memory_usage": [0.73]*6,
        "system_health": 0.93,
    },
    "t07": {
        "task_id": "netweaver_sre_t07", "fault_type": "zombie_process",
        "alert": "WARNING: Zombie processes accumulating on node_15. PID table filling up.",
        "hardware_logs": [
            "node_15: zombie process count=143 (ppid=1, state=Z)",
            "node_15: PID table 91% full (30,847/32,768)",
            "node_15: new process forks failing: EAGAIN",
        ],
        "queue_depths": {"switch_a": 4.5, "switch_b": 4.3},
        "gradient_variances": [0.01]*10,
        "gpu_memory_usage": [0.70]*6,
        "system_health": 0.90,
    },
    "t08": {
        "task_id": "netweaver_sre_t08", "fault_type": "pfc_congestion",
        "alert": "WARNING: PFC buffer congestion on switch_spine_02. Packet loss detected on RDMA traffic.",
        "hardware_logs": [
            "switch_spine_02: PFC PAUSE frames excessive on port 24",
            "switch_spine_02: buffer utilization 97.3%, threshold not set",
        ],
        "queue_depths": {"switch_spine_01": 14.2, "switch_spine_02": 97.3, "switch_leaf_01": 11.1},
        "gradient_variances": [0.03]*10,
        "gpu_memory_usage": [0.72]*6,
        "system_health": 0.78,
    },
    "t09": {
        "task_id": "netweaver_sre_t09", "fault_type": "power_throttle",
        "alert": "WARNING: node_31 throttling due to power cap. GPU compute reduced by 40%.",
        "hardware_logs": [
            "node_31: power cap hit: current=320W, limit=250W, throttling active",
            "node_31: GPU clock reduced from 1800MHz to 1100MHz",
            "node_31: recommend ADJUST_POWER_CAP to 350W",
        ],
        "queue_depths": {"switch_a": 5.5, "switch_b": 5.2},
        "gradient_variances": [0.02]*10,
        "gpu_memory_usage": [0.71]*6,
        "system_health": 0.82,
    },
    "t10": {
        "task_id": "netweaver_sre_t10", "fault_type": "bgp_flap",
        "alert": "CRITICAL: BGP session flapping on router_spine_01. Routes withdrawn and re-announced every 8s.",
        "hardware_logs": [
            "router_spine_01: BGP session to AS64512 flapping (up/down 47 times in 10min)",
            "router_spine_01: hold timer expired for peer 10.0.1.1 AS64512",
            "router_spine_01: route table instability detected",
        ],
        "queue_depths": {"switch_a": 8.0, "switch_b": 7.5},
        "gradient_variances": [0.03]*10,
        "gpu_memory_usage": [0.70]*6,
        "system_health": 0.75,
    },
    "t11": {
        "task_id": "netweaver_sre_t11", "fault_type": "packet_drop",
        "alert": "ERROR: Jumbo frame packet drops on switch_leaf_07. RDMA throughput degraded 60%.",
        "hardware_logs": [
            "switch_leaf_07: MTU mismatch — interface MTU=1500, jumbo frames=9000",
            "switch_leaf_07: dropping 12,400 packets/s on port 18",
            "switch_leaf_07: fix: set interface MTU to 9000",
        ],
        "queue_depths": {"switch_leaf_07": 34.5, "switch_leaf_08": 9.1},
        "gradient_variances": [0.02]*10,
        "gpu_memory_usage": [0.72]*6,
        "system_health": 0.80,
    },
    "t12": {
        "task_id": "netweaver_sre_t12", "fault_type": "ddos",
        "alert": "CRITICAL: DDoS detected on api_gateway_01. 480,000 req/s from spoofed IPs.",
        "hardware_logs": [
            "api_gateway_01: request rate 480,000 req/s (normal: 1,200 req/s)",
            "api_gateway_01: connection queue saturated",
        ],
        "queue_depths": {"api_gateway_01": 99.1, "switch_a": 11.2},
        "gradient_variances": [0.01]*10,
        "gpu_memory_usage": [0.70]*6,
        "system_health": 0.60,
    },
    "t13": {
        "task_id": "netweaver_sre_t13", "fault_type": "conn_exhaustion",
        "alert": "CRITICAL: Database connection pool exhausted on db_node_02. All 100 connections in use.",
        "hardware_logs": [
            "db_node_02: connection pool exhausted (100/100 active)",
            "db_node_02: 847 connection requests queued",
        ],
        "queue_depths": {"db_node_02": 98.7, "switch_a": 5.1},
        "gradient_variances": [0.01]*10,
        "gpu_memory_usage": [0.69]*6,
        "system_health": 0.72,
    },
    "t14": {
        "task_id": "netweaver_sre_t14", "fault_type": "cpu_context_switch",
        "alert": "WARNING: Excessive CPU context switches on node_08. Training step time increased 3x.",
        "hardware_logs": [
            "node_08: context switches 2,400,000/s (normal: 50,000/s)",
            "node_08: 128 threads competing for 64 cores",
            "node_08: recommend PIN_CPU_THREADS to 64",
        ],
        "queue_depths": {"switch_a": 6.0, "switch_b": 5.8},
        "gradient_variances": [0.03]*10,
        "gpu_memory_usage": [0.73]*6,
        "system_health": 0.79,
    },
    "t15": {
        "task_id": "netweaver_sre_t15", "fault_type": "nan_contagion",
        "alert": "CRITICAL: Silent NaN contagion detected in gradient sync layer. Rank 4 corrupted.",
        "hardware_logs": [
            "cluster_2: gradient sync anomaly detected rank=4",
            "cluster_2: NaN propagating to dependent ranks",
        ],
        "queue_depths": {"switch_a": 8.0, "switch_b": 7.8},
        "gradient_variances": [0.01, 0.01, 0.01, 0.01, -1.0, 0.01, 0.01, 0.01, 0.01, 0.01],
        "gpu_memory_usage": [0.70]*6,
        "system_health": 0.65,
    },
    "t16": {
        "task_id": "netweaver_sre_t16", "fault_type": "broadcast_storm",
        "alert": "CRITICAL: Broadcast storm detected. switch_leaf_03 saturated.",
        "hardware_logs": [
            "switch_leaf_03: broadcast flood detected on VLAN 100",
            "switch_leaf_03: 98% of bandwidth consumed by broadcast frames",
        ],
        "queue_depths": {"switch_leaf_01": 12.1, "switch_leaf_02": 9.3, "switch_leaf_03": 99.4, "switch_leaf_04": 10.2},
        "gradient_variances": [0.02]*10,
        "gpu_memory_usage": [0.71]*6,
        "system_health": 0.55,
    },
    "t17": {
        "task_id": "netweaver_sre_t17", "fault_type": "gpu_memory_leak",
        "alert": "WARNING: GPU memory leak on cluster_4. Memory usage climbing; OOM imminent.",
        "hardware_logs": [
            "cluster_4: GPU memory usage increasing 2% per minute",
            "cluster_4: memory fragmentation detected in CUDA allocator",
        ],
        "queue_depths": {"switch_a": 7.0, "switch_b": 6.8},
        "gradient_variances": [0.02]*10,
        "gpu_memory_usage": [0.71, 0.72, 0.70, 0.71, 0.97, 0.72],
        "system_health": 0.74,
    },
    "t18": {
        "task_id": "netweaver_sre_t18", "fault_type": "cluster_deadlock",
        "alert": "CRITICAL: Full cluster deadlock. All subsystems unresponsive. No heartbeat.",
        "hardware_logs": [
            "cluster_0: watchdog timeout — no heartbeat for 120s",
            "cluster_0: all process states frozen",
            "cluster_0: deadlock detected across all ranks",
        ],
        "queue_depths": {"switch_a": 0.0, "switch_b": 0.0},
        "gradient_variances": [0.0]*10,
        "gpu_memory_usage": [0.0]*6,
        "system_health": 0.0,
    },
    "t19": {
        "task_id": "netweaver_sre_t19", "fault_type": "network_partition",
        "alert": "CRITICAL: Network partition detected. Pod-A and Pod-B cannot communicate.",
        "hardware_logs": [
            "pod_b: cannot reach pod_a (packet loss 100%)",
            "pod_b: leaf switch link to pod_a down",
        ],
        "queue_depths": {"pod_a_switch": 0.01, "pod_b_switch": 99.9},
        "gradient_variances": [0.04]*10,
        "gpu_memory_usage": [0.70]*6,
        "system_health": 0.40,
    },
    "t20": {
        "task_id": "netweaver_sre_t20", "fault_type": "corrupt_db",
        "alert": "CRITICAL: Corrupt database block detected. Checkpoint health degrading on cluster_6.",
        "hardware_logs": [
            "cluster_6: block checksum mismatch at offset 0x3A4F000",
            "cluster_6: health score dropping 5% per checkpoint cycle",
            "cluster_6: storage controller reports I/O error on cluster_6",
        ],
        "queue_depths": {"switch_a": 5.0, "switch_b": 4.8},
        "gradient_variances": [0.01]*10,
        "gpu_memory_usage": [0.71]*6,
        "system_health": 0.62,
    },
}


# ── Request/response models ───────────────────────────────────────────────────

class SetLevelRequest(BaseModel):
    task_level: str

class ActionPayload(BaseModel):
    command: str
    target: str
    value: Optional[int] = None

class StepRequest(BaseModel):
    action: ActionPayload


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_obs(scenario: dict, step_count: int, reward: float, done: bool) -> dict:
    # Check if a corrective command has been successfully issued
    is_resolved = SESSION.get("is_resolved", False)

    if is_resolved:
        return {
            "alert":              "SUCCESS: Resolution confirmed. node_12 DNS cache cleared.",
            "hardware_logs":      ["Status: HEALTHY", "Telemetry: Nominal", "Observation: All nodes reachable"],
            "queue_depths":       {k: 5.0 for k in scenario.get("queue_depths", {})},
            "gradient_variances": [0.01] * len(scenario.get("gradient_variances", [])),
            "gpu_memory_usage":   [0.70] * len(scenario.get("gpu_memory_usage", [])),
            "system_health":      1.0,
            "step_count":         step_count,
            "reward":             round(reward, 4),
            "done":               done,
            "active_connections": random.randint(140, 160),
            "error_rate":         round(SESSION.get("error_count", 0) / max(1, step_count), 3),
        }

    return {
        "alert":              scenario.get("alert", ""),
        "hardware_logs":      scenario.get("hardware_logs", []),
        "queue_depths":       scenario.get("queue_depths", {}),
        "gradient_variances": scenario.get("gradient_variances", []),
        "gpu_memory_usage":   scenario.get("gpu_memory_usage", []),
        "system_health":      scenario.get("system_health", 1.0),
        "step_count":         step_count,
        "reward":             round(reward, 4),
        "done":               done,
        "active_connections": random.randint(80, 120),
        "error_rate":         round(SESSION.get("error_count", 0) / max(1, step_count), 3),
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": cfg["task_id"],
                "fault_type": cfg["fault_type"],
                "difficulty": "Easy" if int(id[1:]) <= 7 else "Medium" if int(id[1:]) <= 14 else "Hard",
            }
            for id, cfg in FAULT_SCENARIOS.items()
        ]
    }


@app.post("/set_level")
def set_level(req: SetLevelRequest):
    SESSION["pending_level"] = req.task_level
    return {"status": "ok", "task_level": req.task_level}


@app.post("/reset")
def reset(body: dict = {}):
    level = SESSION.get("pending_level", "t01")
    scenario = FAULT_SCENARIOS.get(level, FAULT_SCENARIOS["t01"])

    SESSION.clear()
    SESSION["task_level"]        = level
    SESSION["task_id"]           = scenario["task_id"]
    SESSION["fault_type"]        = scenario["fault_type"]
    SESSION["scenario"]          = scenario
    SESSION["step_count"]        = 0
    SESSION["done"]              = False
    SESSION["actions"]           = []
    SESSION["obs_fields_checked"] = set()
    SESSION["rewarded_set"]      = set()
    SESSION["action_history"]    = set()
    SESSION["error_count"]       = 0
    SESSION["destructive_used"]  = False
    SESSION["cumulative_reward"] = 0.0
    SESSION["last_grader"]       = None

    obs = _build_obs(scenario, 0, 0.001, False)
    return {"observation": obs, "done": False, "reward": 0.001}


@app.post("/step")
def step(req: StepRequest):
    if SESSION.get("done"):
        raise HTTPException(status_code=400, detail="Episode is done. Call /reset first.")

    scenario   = SESSION.get("scenario", FAULT_SCENARIOS["t01"])
    fault_type = SESSION.get("fault_type", "node_offline")
    step_count = SESSION.get("step_count", 0) + 1
    SESSION["step_count"] = step_count

    command = req.action.command.upper()
    target  = req.action.target
    value   = req.action.value

    # Record observation field access BEFORE action
    record_obs_access(scenario, SESSION["obs_fields_checked"])

    # Compute shaped reward
    reward, episode_done = compute_step_reward(
        command, target, value, fault_type,
        SESSION["rewarded_set"],
        SESSION["action_history"],
        SESSION["obs_fields_checked"], # pass the checked set
    )
    # Mark as resolved if corrective reward exists in the set
    if any(k.startswith("corrective:") for k in SESSION.get("rewarded_set", set())):
        SESSION["is_resolved"] = True

    # Track errors (commands that had no corrective effect)
    if reward <= 0:
        SESSION["error_count"] = SESSION.get("error_count", 0) + 1

    # Track destructive usage
    if command in DESTRUCTIVE_COMMANDS:
        SESSION["destructive_used"] = True

    # Record action
    SESSION["actions"].append({"command": command, "target": target, "value": value})

    # Cumulative reward clamp
    SESSION["cumulative_reward"] = max(0.001, min(0.999,
        SESSION.get("cumulative_reward", 0.0) + reward
    ))

    # Check max steps
    if step_count >= 15:
        episode_done = True

    SESSION["done"] = episode_done

    # On episode end, compute grader score
    grader_result = None
    if episode_done:
        grader_result = compute_grader_score(SESSION["task_id"], {
            "actions":            SESSION["actions"],
            "steps":              step_count,
            "obs_fields_checked": SESSION["obs_fields_checked"],
            "error_count":        SESSION["error_count"],
            "destructive_used":   SESSION["destructive_used"],
        })
        SESSION["last_grader"] = grader_result
        final_reward = grader_result["total"]
    else:
        final_reward = max(0.001, min(0.999, reward + 0.001))

    obs = _build_obs(scenario, step_count, final_reward, episode_done)
    if episode_done and grader_result:
        obs["grader_score"]     = grader_result["total"]
        obs["grader_breakdown"] = grader_result

    return {
        "observation": obs,
        "done":        episode_done,
        "reward":      round(final_reward, 4),
    }


@app.get("/grader", response_model=NetweaverSreGraderResponse)
def grader():
    """Return the grader score for the last completed episode."""
    last = SESSION.get("last_grader")
    if last is None:
        # Episode still running — return partial grader based on current state
        if not SESSION.get("task_id"):
            return {"total": 0.001, "message": "No episode started"}
        last = compute_grader_score(SESSION["task_id"], {
            "actions":            SESSION.get("actions", []),
            "steps":              SESSION.get("step_count", 0),
            "obs_fields_checked": SESSION.get("obs_fields_checked", set()),
            "error_count":        SESSION.get("error_count", 0),
            "destructive_used":   SESSION.get("destructive_used", False),
        })
    return last


@app.get("/grader/{task_id}", response_model=NetweaverSreGraderResponse)
def grader_for_task(task_id: str):
    last = SESSION.get("last_grader")
    if last and last.get("task_id") == task_id:
        return last
    # compute fresh against current session
    result = compute_grader_score(task_id, {
        "actions":            SESSION.get("actions", []),
        "steps":              SESSION.get("step_count", 0),
        "obs_fields_checked": SESSION.get("obs_fields_checked", set()),
        "error_count":        SESSION.get("error_count", 0),
        "destructive_used":   SESSION.get("destructive_used", False),
    })
    return result


@app.get("/state")
def state():
    return {
        "task_id":          SESSION.get("task_id"),
        "task_level":       SESSION.get("task_level"),
        "fault_type":       SESSION.get("fault_type"),
        "step_count":       SESSION.get("step_count", 0),
        "done":             SESSION.get("done", False),
        "cumulative_reward": SESSION.get("cumulative_reward", 0.001),
        "error_count":      SESSION.get("error_count", 0),
        "actions_taken":    SESSION.get("actions", []),
    }


@app.get("/", response_class=HTMLResponse)
def playground():
    html_path = os.path.join(os.path.dirname(__file__), "playground.html")
    if os.path.exists(html_path):
        with open(html_path, encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>NetWeaver SRE</h1><p>Playground UI not found.</p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)