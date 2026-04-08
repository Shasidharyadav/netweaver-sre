# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import List, Optional, Dict

class NetweaverSreAction(Action):
    """Action for the Netweaver Sre environment."""
    command: str = Field(..., description="Action command (e.g., DRAIN_TRAFFIC, TUNE_PFC_THRESHOLD, RUN_MINI_ITERATION)")
    target: str = Field(..., description="Target node, switch, or queue")
    value: Optional[int] = Field(default=None, description="Optional numerical value for threshold tuning")

class NetweaverSreObservation(Observation):
    """Observation telemetry from the SRE environment."""
    queue_depths: Dict[str, float] = Field(default_factory=dict, description="Current depth of network buffers")
    gradient_variances: List[float] = Field(default_factory=list, description="Recent variance of gradients per GPU rank")
    hardware_logs: List[str] = Field(default_factory=list, description="Recent hardware and system logs")
    system_health: float = Field(default=1.0, description="Overall system SLA health")
