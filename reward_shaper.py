# reward_shaper.py
# Per-step reward shaping with anti-reward-hacking measures.
#
# This module produces a SHAPING bonus that the server can blend with the
# environment's ground-truth reward. Returned values are clamped to
# (0.001, 0.999) so they are always safe to expose over HTTP — internally
# the shaping is signed (+/-) but exposed as a small positive bonus.

from typing import Iterable, Optional, Set, Tuple

# Which observation fields each command is expected to consult
DIAGNOSTIC_OBS_FIELDS = {
    "hardware_logs": [
        "DRAIN_TRAFFIC", "RESTART_POD", "KILL_ZOMBIE_PROCESS",
        "RESTART_SERVICE", "RENEW_CERTIFICATE", "CLEAR_TEMP_FILES",
        "CLEAR_DNS_CACHE", "ADJUST_POWER_CAP", "MITIGATE_ROUTE_FLAP",
        "PIN_CPU_THREADS", "SCALE_CONN_POOL",
    ],
    "queue_depths": [
        "TUNE_PFC_THRESHOLD", "ISOLATE_BROADCAST_STORM",
        "REBOOT_LEAF_SWITCHES", "INCREASE_MTU", "SET_RATE_LIMIT",
    ],
    "gradient_variances": ["RUN_MINI_ITERATION"],
    "gpu_memory_usage": ["RESTART_GPU_DAEMON"],
    "system_health": ["PURGE_CORRUPT_BLOCK", "ISSUE_GLOBAL_ROLLBACK"],
}

# Corrective command → valid fault types (fault-type gating)
CORRECTIVE_VALID_FAULTS = {
    "DRAIN_TRAFFIC":           ["node_offline", "nan_contagion", "gradient_poisoning"],
    "CLEAR_DNS_CACHE":         ["dns_cache"],
    "RESTART_SERVICE":         ["oom_crash"],
    "RENEW_CERTIFICATE":       ["tls_expiry"],
    "CLEAR_TEMP_FILES":        ["disk_full"],
    "RESTART_POD":             ["unhealthy_pod"],
    "KILL_ZOMBIE_PROCESS":     ["zombie_process"],
    "TUNE_PFC_THRESHOLD":      ["pfc_congestion"],
    "ADJUST_POWER_CAP":        ["power_throttle", "cascading_failure"],
    "MITIGATE_ROUTE_FLAP":     ["bgp_flap"],
    "INCREASE_MTU":            ["packet_drop"],
    "SET_RATE_LIMIT":          ["ddos"],
    "SCALE_CONN_POOL":         ["conn_exhaustion", "cascading_failure"],
    "PIN_CPU_THREADS":         ["cpu_context_switch", "cascading_failure"],
    "RUN_MINI_ITERATION":      ["nan_contagion", "gradient_poisoning"],
    "ISOLATE_BROADCAST_STORM": ["broadcast_storm", "gradient_poisoning"],
    "RESTART_GPU_DAEMON":      ["gpu_memory_leak"],
    "ISSUE_GLOBAL_ROLLBACK":   ["cluster_deadlock"],
    "REBOOT_LEAF_SWITCHES":    ["network_partition"],
    "PURGE_CORRUPT_BLOCK":     ["corrupt_db"],
}

DESTRUCTIVE_COMMANDS = {
    "FORCE_REBOOT_ALL", "KILL_ALL_NODES", "WIPE_CLUSTER", "DROP_ALL_TRAFFIC",
}

CORRECTIVE_REWARD = 0.10
DIAGNOSTIC_REWARD = 0.05
WRONG_FIX_PENALTY = -0.03
DUPLICATE_PENALTY = -0.05
DESTRUCTIVE_PENALTY = -0.50


def _clamp_exposed(x: float) -> float:
    """Clamp to the public spec range (0.001, 0.999) — never 0.0 or 1.0."""
    return max(0.001, min(0.999, float(x)))


def compute_step_reward(
    command: str,
    target: str,
    value,
    fault_type: str,
    rewarded_set: set,
    action_history: set,
    obs_fields_present: Optional[Set[str]] = None,
    had_error: bool = False,
) -> Tuple[float, bool]:
    """
    Returns (reward_in_(0.001, 0.999), episode_done).

    `rewarded_set` and `action_history` are mutated in-place so the caller
    can persist them across steps.

    The internal calculation is signed (negative for wrong/duplicate),
    but the externally returned reward is mapped to (0.001, 0.999) so it
    is always safe to expose over HTTP per the OpenEnv spec.
    """
    cmd = (command or "").upper()
    tgt = (target or "")

    if cmd in DESTRUCTIVE_COMMANDS:
        # Hard terminate; tiny exposed reward, episode_done=True
        return _clamp_exposed(0.001), True

    raw = 0.0
    if had_error:
        raw -= 0.05

    # Corrective reward (fault-type gated, fires once per command type)
    valid_faults = CORRECTIVE_VALID_FAULTS.get(cmd, [])
    if valid_faults:
        ck = f"corrective:{cmd}"
        if fault_type in valid_faults:
            if ck not in rewarded_set:
                raw += CORRECTIVE_REWARD
                rewarded_set.add(ck)
        else:
            raw += WRONG_FIX_PENALTY

    # Duplicate penalty
    action_key = f"{cmd}:{tgt}".lower()
    if action_key in action_history:
        raw += DUPLICATE_PENALTY
    else:
        action_history.add(action_key)

    # Diagnostic reward (obs-field aware, fires once per field)
    if obs_fields_present:
        for field, cmds in DIAGNOSTIC_OBS_FIELDS.items():
            if cmd in cmds and field in obs_fields_present:
                dk = f"diag:{field}"
                if dk not in rewarded_set:
                    raw += DIAGNOSTIC_REWARD
                    rewarded_set.add(dk)

    # Map signed raw [-0.5, +0.5] -> exposed (0.001, 0.999)
    # raw=0 -> ~0.5; raw=+0.5 -> 0.999; raw=-0.5 -> 0.001
    exposed = 0.5 + raw
    return _clamp_exposed(exposed), False


def record_obs_fields(obs: dict) -> Set[str]:
    """
    Inspect an observation dict and return the set of field names
    that contain meaningful (non-trivial) data this turn.
    """
    present: Set[str] = set()
    if obs.get("hardware_logs"):
        present.add("hardware_logs")

    q = obs.get("queue_depths", {}) or {}
    if q and any(float(v) > 15.0 for v in q.values()):
        present.add("queue_depths")

    gv = obs.get("gradient_variances", []) or []
    if gv and any(abs(float(v)) > 0.5 for v in gv):
        present.add("gradient_variances")

    gm = obs.get("gpu_memory_usage", []) or []
    if gm and any(float(v) > 0.85 for v in gm):
        present.add("gpu_memory_usage")

    sh = obs.get("system_health", 1.0)
    try:
        if float(sh) < 0.95:
            present.add("system_health")
    except (TypeError, ValueError):
        pass

    return present


def record_obs_access(scenario: dict, checked_set: set):
    """Backward-compat wrapper retained for older callers; mutates `checked_set`."""
    checked_set.update(record_obs_fields(scenario))
