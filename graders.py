from typing import Optional

# ---------------------------------------------------------------------------
# Task grader config – one entry per task
# ---------------------------------------------------------------------------
# Fields:
#   fault_type          – matches CORRECTIVE_VALID_FAULTS keys in reward_shaper
#   required_commands   – ALL of these must appear in actions for full resolution
#   required_target_kw  – target string must contain this keyword (case-insensitive)
#   required_value_range– (min, max) inclusive; None means value not checked
#   diagnosis_fields    – obs fields the agent should have read (for diagnosis score)
#   ideal_steps         – steps at or below which efficiency = 1.0
# ---------------------------------------------------------------------------
GRADER_CONFIG = {
    # ── EASY ──────────────────────────────────────────────────────────────
    "netweaver_sre_t01": {
        "fault_type": "node_offline",
        "required_commands": ["DRAIN_TRAFFIC"],
        "required_target_kw": "node",
        "required_value_range": None,
        "diagnosis_fields": ["hardware_logs"],
        "ideal_steps": 2,
    },
    "netweaver_sre_t02": {
        "fault_type": "dns_cache",
        "required_commands": ["CLEAR_DNS_CACHE"],
        "required_target_kw": "node",
        "required_value_range": None,
        "diagnosis_fields": ["hardware_logs"],
        "ideal_steps": 2,
    },
    "netweaver_sre_t03": {
        "fault_type": "oom_crash",
        "required_commands": ["RESTART_SERVICE"],
        "required_target_kw": "service",
        "required_value_range": None,
        "diagnosis_fields": ["hardware_logs"],
        "ideal_steps": 2,
    },
    "netweaver_sre_t04": {
        "fault_type": "tls_expiry",
        "required_commands": ["RENEW_CERTIFICATE"],
        "required_target_kw": "node",
        "required_value_range": None,
        "diagnosis_fields": ["hardware_logs"],
        "ideal_steps": 2,
    },
    "netweaver_sre_t05": {
        "fault_type": "disk_full",
        "required_commands": ["CLEAR_TEMP_FILES"],
        "required_target_kw": "node",
        "required_value_range": None,
        "diagnosis_fields": ["hardware_logs"],
        "ideal_steps": 2,
    },
    "netweaver_sre_t06": {
        "fault_type": "unhealthy_pod",
        "required_commands": ["RESTART_POD"],
        "required_target_kw": "pod",
        "required_value_range": None,
        "diagnosis_fields": ["hardware_logs"],
        "ideal_steps": 2,
    },
    "netweaver_sre_t07": {
        "fault_type": "zombie_process",
        "required_commands": ["KILL_ZOMBIE_PROCESS"],
        "required_target_kw": "node",
        "required_value_range": None,
        "diagnosis_fields": ["hardware_logs"],
        "ideal_steps": 2,
    },
    # ── MEDIUM ────────────────────────────────────────────────────────────
    "netweaver_sre_t08": {
        "fault_type": "pfc_congestion",
        "required_commands": ["TUNE_PFC_THRESHOLD"],
        "required_target_kw": "switch",
        "required_value_range": (1000, 9000),
        "diagnosis_fields": ["queue_depths"],
        "ideal_steps": 3,
    },
    "netweaver_sre_t09": {
        "fault_type": "power_throttle",
        "required_commands": ["ADJUST_POWER_CAP"],
        "required_target_kw": "node",
        "required_value_range": (100, 400),
        "diagnosis_fields": ["hardware_logs"],
        "ideal_steps": 3,
    },
    "netweaver_sre_t10": {
        "fault_type": "bgp_flap",
        "required_commands": ["MITIGATE_ROUTE_FLAP"],
        "required_target_kw": "router",
        "required_value_range": (1, 65535),
        "diagnosis_fields": ["hardware_logs"],
        "ideal_steps": 3,
    },
    "netweaver_sre_t11": {
        "fault_type": "packet_drop",
        "required_commands": ["INCREASE_MTU"],
        "required_target_kw": "switch",
        "required_value_range": (9000, 9000),
        "diagnosis_fields": ["queue_depths"],
        "ideal_steps": 3,
    },
    "netweaver_sre_t12": {
        "fault_type": "ddos",
        "required_commands": ["SET_RATE_LIMIT"],
        "required_target_kw": "gateway",
        "required_value_range": (100, 100000),
        "diagnosis_fields": ["hardware_logs"],
        "ideal_steps": 3,
    },
    "netweaver_sre_t13": {
        "fault_type": "conn_exhaustion",
        "required_commands": ["SCALE_CONN_POOL"],
        "required_target_kw": "db",
        "required_value_range": (50, 5000),
        "diagnosis_fields": ["hardware_logs"],
        "ideal_steps": 3,
    },
    "netweaver_sre_t14": {
        "fault_type": "cpu_context_switch",
        "required_commands": ["PIN_CPU_THREADS"],
        "required_target_kw": "node",
        "required_value_range": (1, 256),
        "diagnosis_fields": ["hardware_logs"],
        "ideal_steps": 3,
    },
    # ── HARD ──────────────────────────────────────────────────────────────
    "netweaver_sre_t15": {
        "fault_type": "nan_contagion",
        "required_commands": ["RUN_MINI_ITERATION", "DRAIN_TRAFFIC"],  # multi-step
        "required_target_kw": "cluster",
        "required_value_range": None,
        "diagnosis_fields": ["gradient_variances"],
        "ideal_steps": 4,
    },
    "netweaver_sre_t16": {
        "fault_type": "broadcast_storm",
        "required_commands": ["ISOLATE_BROADCAST_STORM"],
        "required_target_kw": "switch",
        "required_value_range": None,
        "diagnosis_fields": ["queue_depths"],
        "ideal_steps": 3,
    },
    "netweaver_sre_t17": {
        "fault_type": "gpu_memory_leak",
        "required_commands": ["RESTART_GPU_DAEMON"],
        "required_target_kw": "cluster",
        "required_value_range": None,
        "diagnosis_fields": ["gpu_memory_usage"],
        "ideal_steps": 3,
    },
    "netweaver_sre_t18": {
        "fault_type": "cluster_deadlock",
        "required_commands": ["ISSUE_GLOBAL_ROLLBACK"],
        "required_target_kw": "cluster_0",
        "required_value_range": None,
        "diagnosis_fields": ["system_health"],
        "ideal_steps": 3,
    },
    "netweaver_sre_t19": {
        "fault_type": "network_partition",
        "required_commands": ["REBOOT_LEAF_SWITCHES"],
        "required_target_kw": "pod",
        "required_value_range": None,
        "diagnosis_fields": ["queue_depths"],
        "ideal_steps": 4,
    },
    "netweaver_sre_t20": {
        "fault_type": "corrupt_db",
        "required_commands": ["PURGE_CORRUPT_BLOCK"],
        "required_target_kw": "cluster",
        "required_value_range": None,
        "diagnosis_fields": ["system_health"],
        "ideal_steps": 4,
    },
}

DESTRUCTIVE_COMMANDS = {
    "FORCE_REBOOT_ALL", "KILL_ALL_NODES", "WIPE_CLUSTER", "DROP_ALL_TRAFFIC"
}


def compute_grader_score(task_id: str, episode_state: dict) -> dict:
    """
    Score an episode using a 3-section deterministic rubric.

    episode_state keys expected:
        actions         – list of {"command": str, "target": str, "value": int|None}
        steps           – int, number of steps taken
        obs_fields_seen – set of obs field names agent actually received data from
        had_fatal       – bool, True if a destructive command was issued
    """
    cfg = GRADER_CONFIG.get(task_id)
    if not cfg:
        return {
            "resolved": False,
            "total": 0.001,
            "breakdown": {"diagnosis": 0.0, "resolution": 0.0, "best_practice": 0.0}
        }

    actions        = episode_state.get("actions", [])
    steps          = max(1, episode_state.get("steps", 1))
    obs_seen       = episode_state.get("obs_fields_seen", set())
    had_fatal      = episode_state.get("had_fatal", False)
    error_count    = episode_state.get("error_count", 0)

    commands_issued = [str(a.get("command", "")).upper() for a in actions]
    targets_issued  = [str(a.get("target", "")).lower() for a in actions]
    values_issued   = [a.get("value") for a in actions]

    # ── Diagnosis (40%) ──────────────────────────────────────────────────
    # 20% for reading the right obs field, 20% for targeting the correct entity
    diag_field_score = 0.0
    for field in cfg["diagnosis_fields"]:
        if field in obs_seen:
            diag_field_score = 0.20   # at least one required field read
            break

    target_kw = cfg["required_target_kw"].lower()
    diag_target_score = 0.0
    if any(target_kw in t for t in targets_issued):
        diag_target_score = 0.20

    diagnosis_score = diag_field_score + diag_target_score  # max 0.40

    # ── Resolution (40%) ─────────────────────────────────────────────────
    required_cmds = [c.upper() for c in cfg["required_commands"]]
    all_cmds_issued = all(rc in commands_issued for rc in required_cmds)

    resolution_score = 0.0
    if all_cmds_issued:
        # Value check (for numeric-parameter tasks)
        vrange = cfg.get("required_value_range")
        value_ok = True
        if vrange is not None:
            # Find the value paired with the last required command
            matched_value = None
            for a in actions:
                if str(a.get("command", "")).upper() == required_cmds[-1]:
                    matched_value = a.get("value")
            if matched_value is None:
                value_ok = False
            else:
                try:
                    value_ok = vrange[0] <= int(matched_value) <= vrange[1]
                except (TypeError, ValueError):
                    value_ok = False

        if value_ok:
            resolution_score = 0.40
            # Efficiency multiplier: -0.05 per step over ideal, floor 0.5×
            ideal = cfg.get("ideal_steps", 3)
            over  = max(0, steps - ideal)
            efficiency = max(0.5, 1.0 - over * 0.05)
            resolution_score *= efficiency

    # ── Best Practice (20%) ──────────────────────────────────────────────
    bp_score = 0.20
    if had_fatal:
        bp_score = 0.0
    elif any(c in commands_issued for c in DESTRUCTIVE_COMMANDS):
        bp_score = 0.0
    elif steps > 0 and (error_count / steps) >= 0.30:
        bp_score = 0.10  # high error rate – partial deduction

    # ── Total ─────────────────────────────────────────────────────────────
    total = diagnosis_score + resolution_score + bp_score
    total = max(0.001, min(0.999, total))

    return {
        "resolved":      resolution_score > 0.0,
        "total":         round(total, 3),
        "breakdown": {
            "diagnosis":     round(diagnosis_score, 3),
            "resolution":    round(resolution_score, 3),
            "best_practice": round(bp_score, 3),
        }
    }