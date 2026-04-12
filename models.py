# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import List, Optional, Dict

class NetweaverSreAction(Action):
    """Action for the Netweaver Sre environment."""
    command: str = Field(..., description="Action command")
    target: str = Field(..., description="Target node, switch, or queue")
    value: Optional[int] = Field(default=None, description="Optional numerical value")

class NetweaverSreObservation(Observation):
    """Observation telemetry from the SRE environment."""
    done: bool = Field(default=False, description="Whether the episode is complete")
    reward: float = Field(default=0.001, description="Reward for the last action")
    step_count: int = Field(default=0, description="Current step in the episode")
    queue_depths: Dict[str, float] = Field(default_factory=dict, description="Current depth of network buffers")
    gradient_variances: List[float] = Field(default_factory=list, description="Recent variance of gradients")
    gpu_memory_usage: List[float] = Field(default_factory=list, description="GPU memory util per sub-cluster")
    hardware_logs: List[str] = Field(default_factory=list, description="Recent hardware and system logs")
    system_health: float = Field(default=1.0, description="Overall system SLA health")
