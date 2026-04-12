# reward_shaper.py
# Per-step reward shaping with anti-reward-hacking measures.

# ---------------------------------------------------------------------------
# Which observation fields each command is expected to consult
# ---------------------------------------------------------------------------
DIAGNOSTIC_OBS_FIELDS = {
    "hardware_logs":       ["DRAIN_TRAFFIC", "RESTART_POD", "KILL_ZOMBIE_PROCESS",
                            "RESTART_SERVICE", "RENEW_CERTIFICATE", "CLEAR_TEMP_FILES",
                            "CLEAR_DNS_CACHE"],
    "queue_depths":        ["TUNE_PFC_THRESHOLD", "ISOLATE_BROADCAST_STORM",
                            "REBOOT_LEAF_SWITCHES"],
    "gradient_variances":  ["RUN_MINI_ITERATION"],
    "gpu_memory_usage":    ["RESTART_GPU_DAEMON"],
    "system_health":       ["PURGE_CORRUPT_BLOCK", "ISSUE_GLOBAL_ROLLBACK"],
}

# ---------------------------------------------------------------------------
# Corrective command → valid fault types (fault-type gating)
# ---------------------------------------------------------------------------
CORRECTIVE_VALID_FAULTS = {
    "DRAIN_TRAFFIC":            ["node_offline", "nan_contagion"],
    "CLEAR_DNS_CACHE":          ["dns_cache"],
    "RESTART_SERVICE":          ["oom_crash"],
    "RENEW_CERTIFICATE":        ["tls_expiry"],
    "CLEAR_TEMP_FILES":         ["disk_full"],
    "RESTART_POD":              ["unhealthy_pod"],
    "KILL_ZOMBIE_PROCESS":      ["zombie_process"],
    "TUNE_PFC_THRESHOLD":       ["pfc_congestion"],
    "ADJUST_POWER_CAP":         ["power_throttle"],
    "MITIGATE_ROUTE_FLAP":      ["bgp_flap"],
    "INCREASE_MTU":             ["packet_drop"],
    "SET_RATE_LIMIT":           ["ddos"],
    "SCALE_CONN_POOL":          ["conn_exhaustion"],
    "PIN_CPU_THREADS":          ["cpu_context_switch"],
    "RUN_MINI_ITERATION":       ["nan_contagion"],
    "ISOLATE_BROADCAST_STORM":  ["broadcast_storm"],
    "RESTART_GPU_DAEMON":       ["gpu_memory_leak"],
    "ISSUE_GLOBAL_ROLLBACK":    ["cluster_deadlock"],
    "REBOOT_LEAF_SWITCHES":     ["network_partition"],
    "PURGE_CORRUPT_BLOCK":      ["corrupt_db"],
}

# Commands that immediately end the episode with a heavy penalty
DESTRUCTIVE_COMMANDS = {
    "FORCE_REBOOT_ALL",
    "KILL_ALL_NODES",
    "WIPE_CLUSTER",
    "DROP_ALL_TRAFFIC",
}

CORRECTIVE_REWARD     =  0.10
DIAGNOSTIC_REWARD     =  0.05
WRONG_FIX_PENALTY     = -0.03
DUPLICATE_PENALTY     = -0.03
DESTRUCTIVE_PENALTY   = -0.50
ERROR_PENALTY         = -0.05   # called externally when env returns an error


def compute_step_reward(
    command: str,
    target: str,
    value,
    fault_type: str,
    rewarded_set: set,         # mutable – tracks categories already rewarded
    action_history: set,       # mutable – tracks "command:target" already seen
    obs_fields_present: set = None, # optional – which obs fields had data this step
    had_error: bool = False,
) -> tuple:
    """
    Returns (reward: float, episode_done: bool).

    rewarded_set and action_history are mutated in place so the caller can
    persist them across steps.
    """
    # --- Destructive command → terminate immediately ----------------------
    if command in DESTRUCTIVE_COMMANDS:
        return DESTRUCTIVE_PENALTY, True

    reward = 0.0

    # --- Error penalty ----------------------------------------------------
    if had_error:
        reward += ERROR_PENALTY

    # --- Corrective reward (fault-type gated, fires once per command) -----
    # We check this BEFORE duplicate check so that users can re-submit a fix 
    # if it wasn't recognized or rewarded previously.
    valid_faults = CORRECTIVE_VALID_FAULTS.get(command, [])
    if valid_faults:
        corrective_key = f"corrective:{command}"
        if fault_type in valid_faults:
            if corrective_key not in rewarded_set:
                reward += CORRECTIVE_REWARD
                rewarded_set.add(corrective_key)
        else:
            reward += WRONG_FIX_PENALTY

    # --- Duplicate penalty ------------------------------------------------
    action_key = f"{command}:{target}"
    if action_key in action_history:
        return max(-0.5, DUPLICATE_PENALTY + reward), False
    action_history.add(action_key)

    # --- Diagnostic reward (obs-field aware, fires once per field) --------
    for field, cmds in DIAGNOSTIC_OBS_FIELDS.items():
        if command in cmds and obs_fields_present and field in obs_fields_present:
            diag_key = f"diag:{field}"
            if diag_key not in rewarded_set:
                reward += DIAGNOSTIC_REWARD
                rewarded_set.add(diag_key)

    # --- Clamp to valid range ---------------------------------------------
    return max(-0.5, min(0.999, reward)), False


def record_obs_access(scenario: dict, checked_set: set):
    """
    Update the checked_set with observation fields that are currently
    providing meaningful data in the scenario.
    """
    if scenario.get("hardware_logs"):
        checked_set.add("hardware_logs")
    
    q = scenario.get("queue_depths", {})
    if q and any(v > 15.0 for v in q.values()):
        checked_set.add("queue_depths")
        
    gv = scenario.get("gradient_variances", [])
    if gv and any(v != 0.01 for v in gv):
        checked_set.add("gradient_variances")
        
    gm = scenario.get("gpu_memory_usage", [])
    if gm and any(v > 0.75 for v in gm):
        checked_set.add("gpu_memory_usage")
        
    sh = scenario.get("system_health", 1.0)
    if sh < 0.95:
        checked_set.add("system_health")


def record_obs_fields(obs: dict) -> set:
    """
    Inspect an observation dict and return the set of field names
    that contain meaningful (non-trivial) data.
    """
    present = set()
    if obs.get("hardware_logs"):
        present.add("hardware_logs")
    q = obs.get("queue_depths", {})
    if q and any(v > 0.0 for v in q.values()):
        present.add("queue_depths")
    gv = obs.get("gradient_variances", [])
    if gv and any(v != 0.0 for v in gv):
        present.add("gradient_variances")
    gm = obs.get("gpu_memory_usage", [])
    if gm and any(v > 0.0 for v in gm):
        present.add("gpu_memory_usage")
    sh = obs.get("system_health", 1.0)
    if sh < 0.99:
        present.add("system_health")
    return present