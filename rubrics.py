"""Composable rubrics for the NetWeaver SRE environment.

Built on `openenv.core.rubrics.base.Rubric` (RFC 004) — child rubrics are
auto-registered as attributes, so the whole grader is introspectable and
each component can be inspected via `last_score`.

Three sub-rubrics, composed by `NetWeaverSREComposedRubric`:

  DiagnosisRubric    (40%)  — read the right obs field + target the right entity
  ResolutionRubric   (40%)  — issue all required commands (in order if required),
                              with values in the valid range, efficiency-adjusted
  BestPracticeRubric (20%)  — no destructive commands, error rate < 30%

Each `forward(action, observation)` here takes an `action` whose shape is
`{"task_id": str, "episode_state": dict}` (because graders score whole
episodes, not individual `(action, obs)` pairs). This keeps the Rubric
contract intact while letting us reuse the same logic from `graders.py`.

The legacy free function `compute_grader_score(task_id, episode_state)` is
preserved for backward compatibility — it now delegates here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from openenv.core.rubrics.base import Rubric

# ---------------------------------------------------------------------------
# Per-task config (mirrors graders.GRADER_CONFIG)
# ---------------------------------------------------------------------------

GRADER_CONFIG: Dict[str, Dict[str, Any]] = {
    # ── EASY ──────────────────────────────────────────────────────────────
    "netweaver_sre_t01": {"fault_type": "node_offline", "required_commands": ["DRAIN_TRAFFIC"], "required_target_kw": "node", "required_value_range": None, "diagnosis_fields": ["hardware_logs"], "ideal_steps": 2},
    "netweaver_sre_t02": {"fault_type": "dns_cache", "required_commands": ["CLEAR_DNS_CACHE"], "required_target_kw": "node", "required_value_range": None, "diagnosis_fields": ["hardware_logs"], "ideal_steps": 2},
    "netweaver_sre_t03": {"fault_type": "oom_crash", "required_commands": ["RESTART_SERVICE"], "required_target_kw": "", "required_value_range": None, "diagnosis_fields": ["hardware_logs"], "ideal_steps": 2},
    "netweaver_sre_t04": {"fault_type": "tls_expiry", "required_commands": ["RENEW_CERTIFICATE"], "required_target_kw": "node", "required_value_range": None, "diagnosis_fields": ["hardware_logs"], "ideal_steps": 2},
    "netweaver_sre_t05": {"fault_type": "disk_full", "required_commands": ["CLEAR_TEMP_FILES"], "required_target_kw": "node", "required_value_range": None, "diagnosis_fields": ["hardware_logs"], "ideal_steps": 2},
    "netweaver_sre_t06": {"fault_type": "unhealthy_pod", "required_commands": ["RESTART_POD"], "required_target_kw": "pod", "required_value_range": None, "diagnosis_fields": ["hardware_logs"], "ideal_steps": 2},
    "netweaver_sre_t07": {"fault_type": "zombie_process", "required_commands": ["KILL_ZOMBIE_PROCESS"], "required_target_kw": "node", "required_value_range": None, "diagnosis_fields": ["hardware_logs"], "ideal_steps": 2},
    # ── MEDIUM ────────────────────────────────────────────────────────────
    "netweaver_sre_t08": {"fault_type": "pfc_congestion", "required_commands": ["TUNE_PFC_THRESHOLD"], "required_target_kw": "sw", "required_value_range": (1000, 9000), "diagnosis_fields": ["queue_depths"], "ideal_steps": 3},
    "netweaver_sre_t09": {"fault_type": "power_throttle", "required_commands": ["ADJUST_POWER_CAP"], "required_target_kw": "node", "required_value_range": (100, 400), "diagnosis_fields": ["hardware_logs"], "ideal_steps": 3},
    "netweaver_sre_t10": {"fault_type": "bgp_flap", "required_commands": ["MITIGATE_ROUTE_FLAP"], "required_target_kw": "router", "required_value_range": (1, 65535), "diagnosis_fields": ["hardware_logs"], "ideal_steps": 3},
    "netweaver_sre_t11": {"fault_type": "packet_drop", "required_commands": ["INCREASE_MTU"], "required_target_kw": "sw", "required_value_range": (9000, 9000), "diagnosis_fields": ["queue_depths"], "ideal_steps": 3},
    "netweaver_sre_t12": {"fault_type": "ddos", "required_commands": ["SET_RATE_LIMIT"], "required_target_kw": "gateway", "required_value_range": (100, 100000), "diagnosis_fields": ["queue_depths"], "ideal_steps": 3},
    "netweaver_sre_t13": {"fault_type": "conn_exhaustion", "required_commands": ["SCALE_CONN_POOL"], "required_target_kw": "db", "required_value_range": (50, 5000), "diagnosis_fields": ["hardware_logs"], "ideal_steps": 3},
    "netweaver_sre_t14": {"fault_type": "cpu_context_switch", "required_commands": ["PIN_CPU_THREADS"], "required_target_kw": "node", "required_value_range": (1, 256), "diagnosis_fields": ["hardware_logs"], "ideal_steps": 3},
    # ── HARD ──────────────────────────────────────────────────────────────
    "netweaver_sre_t15": {"fault_type": "nan_contagion", "required_commands": ["RUN_MINI_ITERATION", "DRAIN_TRAFFIC"], "required_target_kw": "cluster", "required_value_range": None, "diagnosis_fields": ["gradient_variances"], "ideal_steps": 4, "enforce_order": True},
    "netweaver_sre_t16": {"fault_type": "broadcast_storm", "required_commands": ["ISOLATE_BROADCAST_STORM"], "required_target_kw": "sw", "required_value_range": None, "diagnosis_fields": ["queue_depths"], "ideal_steps": 3},
    "netweaver_sre_t17": {"fault_type": "gpu_memory_leak", "required_commands": ["RESTART_GPU_DAEMON"], "required_target_kw": "cluster", "required_value_range": None, "diagnosis_fields": ["gpu_memory_usage"], "ideal_steps": 3},
    "netweaver_sre_t18": {"fault_type": "cluster_deadlock", "required_commands": ["ISSUE_GLOBAL_ROLLBACK"], "required_target_kw": "cluster", "required_value_range": None, "diagnosis_fields": ["system_health"], "ideal_steps": 3},
    "netweaver_sre_t19": {"fault_type": "network_partition", "required_commands": ["REBOOT_LEAF_SWITCHES"], "required_target_kw": "pod", "required_value_range": None, "diagnosis_fields": ["queue_depths"], "ideal_steps": 4},
    "netweaver_sre_t20": {"fault_type": "corrupt_db", "required_commands": ["PURGE_CORRUPT_BLOCK"], "required_target_kw": "db", "required_value_range": None, "diagnosis_fields": ["system_health"], "ideal_steps": 4},
    "netweaver_sre_t21": {"fault_type": "cascading_failure", "required_commands": ["ADJUST_POWER_CAP", "PIN_CPU_THREADS", "SCALE_CONN_POOL"], "required_target_kw": "node", "required_value_range": None, "diagnosis_fields": ["hardware_logs"], "ideal_steps": 6, "enforce_order": True},
    "netweaver_sre_t22": {"fault_type": "gradient_poisoning", "required_commands": ["RUN_MINI_ITERATION", "ISOLATE_BROADCAST_STORM", "DRAIN_TRAFFIC"], "required_target_kw": "cluster", "required_value_range": None, "diagnosis_fields": ["gradient_variances", "queue_depths"], "ideal_steps": 6, "enforce_order": True},
}

DESTRUCTIVE_COMMANDS = {"FORCE_REBOOT_ALL", "KILL_ALL_NODES", "WIPE_CLUSTER", "DROP_ALL_TRAFFIC"}


def _commands_in_order(commands_issued: List[str], required: List[str]) -> bool:
    i = 0
    for cmd in commands_issued:
        if i < len(required) and cmd == required[i]:
            i += 1
    return i == len(required)


def _clamp(x: float) -> float:
    return max(0.001, min(0.999, float(x)))


# ---------------------------------------------------------------------------
# Sub-rubrics
# ---------------------------------------------------------------------------

class DiagnosisRubric(Rubric):
    """20% for reading the right obs field + 20% for targeting the right entity.

    `action` is `{"task_id": str, "episode_state": {...}}`.
    `observation` is unused (graders score whole episodes).
    """

    MAX_SCORE: float = 0.40

    def forward(self, action: Any, observation: Any = None) -> float:
        cfg = GRADER_CONFIG.get(action["task_id"])
        if not cfg:
            return 0.001
        ep = action["episode_state"]
        obs_seen = set(ep.get("obs_fields_seen", set()) or set())
        targets = [str(a.get("target", "")).lower() for a in ep.get("actions", [])]

        field_ok = any(f in obs_seen for f in cfg.get("diagnosis_fields", []))
        kw = (cfg.get("required_target_kw") or "").lower()
        if kw == "":
            target_ok = any(t for t in targets)
        else:
            target_ok = any(kw in t for t in targets)

        return (0.20 if field_ok else 0.0) + (0.20 if target_ok else 0.0)


class ResolutionRubric(Rubric):
    """40% if all required commands issued (in order if required) with valid value;
    multiplied by an efficiency factor (-5%/step over ideal, floor 50%)."""

    MAX_SCORE: float = 0.40

    def forward(self, action: Any, observation: Any = None) -> float:
        cfg = GRADER_CONFIG.get(action["task_id"])
        if not cfg:
            return 0.001
        ep = action["episode_state"]
        actions = ep.get("actions", []) or []
        steps = max(1, int(ep.get("steps", 1) or 1))

        commands_issued = [str(a.get("command", "")).upper() for a in actions]
        required = [c.upper() for c in cfg["required_commands"]]
        all_issued = all(rc in commands_issued for rc in required)

        if not all_issued:
            hits = sum(1 for rc in required if rc in commands_issued)
            return min(0.20, 0.40 * hits / max(1, len(required)) * 0.5)

        if cfg.get("enforce_order") and not _commands_in_order(commands_issued, required):
            return 0.20

        vrange = cfg.get("required_value_range")
        if vrange is not None:
            matched = None
            for a in actions:
                if str(a.get("command", "")).upper() == required[-1]:
                    matched = a.get("value")
            try:
                if matched is None or not (vrange[0] <= int(matched) <= vrange[1]):
                    return 0.20
            except (TypeError, ValueError):
                return 0.20

        ideal = int(cfg.get("ideal_steps", 3))
        over = max(0, steps - ideal)
        efficiency = max(0.5, 1.0 - over * 0.05)
        return 0.40 * efficiency


class BestPracticeRubric(Rubric):
    """20% baseline; 0 for destructive commands; 10% if error rate >= 30%."""

    MAX_SCORE: float = 0.20

    def forward(self, action: Any, observation: Any = None) -> float:
        ep = action["episode_state"]
        actions = ep.get("actions", []) or []
        steps = max(1, int(ep.get("steps", 1) or 1))
        had_fatal = bool(ep.get("had_fatal", False))
        error_count = int(ep.get("error_count", 0) or 0)
        commands = [str(a.get("command", "")).upper() for a in actions]

        if had_fatal or any(c in commands for c in DESTRUCTIVE_COMMANDS):
            return 0.001
        if error_count / max(1, steps) >= 0.30:
            return 0.10
        return 0.20


# ---------------------------------------------------------------------------
# Composite — the one we expose to the server
# ---------------------------------------------------------------------------

class NetWeaverSREComposedRubric(Rubric):
    """Top-level rubric: diagnosis (40%) + resolution (40%) + best practice (20%)."""

    def __init__(self) -> None:
        super().__init__()
        # These are auto-registered as children by Rubric.__setattr__
        self.diagnosis = DiagnosisRubric()
        self.resolution = ResolutionRubric()
        self.best_practice = BestPracticeRubric()

    def forward(self, action: Any, observation: Any = None) -> float:
        d = self.diagnosis(action, observation)
        r = self.resolution(action, observation)
        b = self.best_practice(action, observation)
        return _clamp(d + r + b)

    def evaluate_episode(self, task_id: str, episode_state: dict) -> dict:
        """Evaluate a full episode and return the structured grader payload.

        This is what `app.py`'s `/grader` endpoint and `graders.compute_grader_score`
        ultimately call.
        """
        action = {"task_id": task_id, "episode_state": episode_state}
        total = self(action, None)  # __call__ -> forward + last_score storage
        d = float(self.diagnosis.last_score or 0.0)
        r = float(self.resolution.last_score or 0.0)
        b = float(self.best_practice.last_score or 0.0)
        return {
            "resolved": r >= 0.20,
            "total": round(_clamp(total), 3),
            "breakdown": {
                "diagnosis": round(_clamp(d), 3),
                "resolution": round(_clamp(r), 3),
                "best_practice": round(_clamp(b), 3),
            },
        }


# ---------------------------------------------------------------------------
# Singleton + convenience wrapper
# ---------------------------------------------------------------------------

_RUBRIC = NetWeaverSREComposedRubric()


def compute_grader_score(task_id: str, episode_state: dict) -> dict:
    """Backward-compatible wrapper used by `graders.py` and `server/app.py`."""
    return _RUBRIC.evaluate_episode(task_id, episode_state)
