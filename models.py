# models.py
# Typed Pydantic models for NetWeaver SRE environment

from openenv.core.env_server.types import Action, Observation
from pydantic import Field, BaseModel
from typing import List, Optional, Dict, Any


class NetweaverSreAction(Action):
    """Action issued by the agent to the SRE environment."""

    command: str = Field(
        ...,
        description=(
            "Remediation command. One of: DRAIN_TRAFFIC, CLEAR_DNS_CACHE, "
            "RESTART_SERVICE, RENEW_CERTIFICATE, CLEAR_TEMP_FILES, RESTART_POD, "
            "KILL_ZOMBIE_PROCESS, TUNE_PFC_THRESHOLD, ADJUST_POWER_CAP, "
            "MITIGATE_ROUTE_FLAP, INCREASE_MTU, SET_RATE_LIMIT, SCALE_CONN_POOL, "
            "PIN_CPU_THREADS, RUN_MINI_ITERATION, ISOLATE_BROADCAST_STORM, "
            "RESTART_GPU_DAEMON, ISSUE_GLOBAL_ROLLBACK, REBOOT_LEAF_SWITCHES, "
            "PURGE_CORRUPT_BLOCK"
        ),
    )
    target: str = Field(
        ...,
        description="Target entity: node ID, switch ID, cluster ID, pod name, etc.",
    )
    value: Optional[int] = Field(
        default=None,
        description=(
            "Numeric parameter when required. Examples: PFC threshold (1000-9000), "
            "power cap watts (100-400), AS number (1-65535), MTU (9000), "
            "rate limit count, pool size, thread count."
        ),
    )


class NetweaverSreObservation(Observation):
    """Telemetry observation returned to the agent after each step."""

    # Episode metadata
    done: bool = Field(default=False, description="Whether the episode is complete.")
    reward: float = Field(default=0.001, description="Per-step shaped reward (0.001-0.999).")
    step_count: int = Field(default=0, description="Current step index (1-15).")
    alert: str = Field(default="", description="Incident alert text for this episode.")

    # Telemetry arrays
    queue_depths: Dict[str, float] = Field(
        default_factory=dict,
        description="Network buffer depths per switch/node. Healthy < 50.0; congested near 99.9.",
    )
    gradient_variances: List[float] = Field(
        default_factory=list,
        description="Per-rank gradient variance. Healthy ~0.0-0.1; NaN contagion shows spikes or NaN.",
    )
    gpu_memory_usage: List[float] = Field(
        default_factory=list,
        description="GPU memory utilisation per sub-cluster (0.0-1.0). Leak shows spike >0.95.",
    )
    hardware_logs: List[str] = Field(
        default_factory=list,
        description="Recent hardware/system log lines. Contains node IDs, error codes, fault descriptions.",
    )

    # Scalar health metrics
    system_health: float = Field(
        default=1.0,
        description="Overall cluster SLA health (0.0-1.0). Drops when faults are unresolved.",
    )
    active_connections: int = Field(
        default=0,
        description="Current active connections to cluster services.",
    )
    error_rate: float = Field(
        default=0.0,
        description="Recent command error rate this episode (0.0-1.0).",
    )

    # Grader hint (populated on final step only)
    grader_score: Optional[float] = Field(
        default=None,
        description="Final grader score if episode is done (0.001-0.999).",
    )
    grader_breakdown: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Diagnosis/resolution/best_practice breakdown on final step.",
    )


class NetweaverSreGraderResponse(BaseModel):
    """Payload returned by the /grader endpoint."""
    resolved: bool = Field(..., description="Whether the incident was fully resolved.")
    total: float = Field(..., description="Total aggregate score (0.001 - 0.999).")
    breakdown: Dict[str, float] = Field(
        ...,
        description="Detailed score breakdown consisting of diagnosis, resolution, and best_practice scores."
    )