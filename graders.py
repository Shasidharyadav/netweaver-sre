# graders.py
# Per-task rubric grader. Returns {resolved, total, breakdown}.
# All scores are clamped strictly to (0.001, 0.999).

from typing import Optional

# ---------------------------------------------------------------------------
# Task grader config – one entry per task.
# Fields:
#   fault_type           – matches CORRECTIVE_VALID_FAULTS keys in reward_shaper
#   required_commands    – ALL of these must appear in actions
#   required_target_kw   – target string must contain this keyword (lowercased)
#   required_value_range – (min, max) inclusive; None means value not checked
#   diagnosis_fields     – obs fields the agent should have read for diagnosis
#   ideal_steps          – steps at or below which efficiency = 1.0
#   enforce_order        – if True, required_commands must appear in this order
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
        "required_target_kw": "",  # service name is dynamic; accept any
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
        "required_target_kw": "sw",
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
        "required_target_kw": "sw",
        "required_value_range": (9000, 9000),
        "diagnosis_fields": ["queue_depths"],
        "ideal_steps": 3,
    },
    "netweaver_sre_t12": {
        "fault_type": "ddos",
        "required_commands": ["SET_RATE_LIMIT"],
        "required_target_kw": "gateway",
        "required_value_range": (100, 100000),
        "diagnosis_fields": ["queue_depths"],
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
        "required_commands": ["RUN_MINI_ITERATION", "DRAIN_TRAFFIC"],
        "required_target_kw": "cluster",
        "required_value_range": None,
        "diagnosis_fields": ["gradient_variances"],
        "ideal_steps": 4,
        "enforce_order": True,
    },
    "netweaver_sre_t16": {
        "fault_type": "broadcast_storm",
        "required_commands": ["ISOLATE_BROADCAST_STORM"],
        "required_target_kw": "sw",
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
        "required_target_kw": "cluster",
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
        "required_target_kw": "db",
        "required_value_range": None,
        "diagnosis_fields": ["system_health"],
        "ideal_steps": 4,
    },
    "netweaver_sre_t21": {
        "fault_type": "cascading_failure",
        "required_commands": ["ADJUST_POWER_CAP", "PIN_CPU_THREADS", "SCALE_CONN_POOL"],
        "required_target_kw": "node",
        "required_value_range": None,
        "diagnosis_fields": ["hardware_logs"],
        "ideal_steps": 6,
        "enforce_order": True,
    },
    "netweaver_sre_t22": {
        "fault_type": "gradient_poisoning",
        "required_commands": ["RUN_MINI_ITERATION", "ISOLATE_BROADCAST_STORM", "DRAIN_TRAFFIC"],
        "required_target_kw": "cluster",
        "required_value_range": None,
        "diagnosis_fields": ["gradient_variances", "queue_depths"],
        "ideal_steps": 6,
        "enforce_order": True,
    },
}

DESTRUCTIVE_COMMANDS = {
    "FORCE_REBOOT_ALL", "KILL_ALL_NODES", "WIPE_CLUSTER", "DROP_ALL_TRAFFIC",
}


def _commands_appear_in_order(commands_issued, required):
    """True iff `required` is a subsequence of `commands_issued`."""
    i = 0
    for cmd in commands_issued:
        if i < len(required) and cmd == required[i]:
            i += 1
    return i == len(required)


def compute_grader_score(task_id: str, episode_state: dict) -> dict:
    """Score an episode using OpenEnv's Rubric system (RFC 004).

    Delegates to `rubrics.NetWeaverSREComposedRubric` which composes
    three child rubrics:
        DiagnosisRubric (40%) + ResolutionRubric (40%) + BestPracticeRubric (20%)

    All sub-scores and the total are clamped to (0.001, 0.999).
    """
    from rubrics import compute_grader_score as _impl
    return _impl(task_id, episode_state)
