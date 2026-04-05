# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Client implementation for the Netweaver Sre environment.
"""

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from openenv.core.env_server.types import State

try:
    from .models import NetweaverSreAction, NetweaverSreObservation
except ImportError:
    from models import NetweaverSreAction, NetweaverSreObservation


class NetweaverSreEnv(EnvClient[NetweaverSreAction, NetweaverSreObservation, State]):
    """Client for connecting to a remote NetweaverSreEnvironment."""

    def _step_payload(self, action: NetweaverSreAction) -> dict:
        payload = {"command": action.command, "target": action.target}
        if action.value is not None:
            payload["value"] = action.value
        return payload

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {})
        
        obs = NetweaverSreObservation(
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            queue_depths=obs_data.get("queue_depths", {}),
            gradient_variances=obs_data.get("gradient_variances", []),
            hardware_logs=obs_data.get("hardware_logs", []),
            system_health=obs_data.get("system_health", 1.0)
        )
        
        return StepResult(
            observation=obs,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> State:
        # We reuse the default State for simplicity, but track metadata inside it or in env logic
        return State(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
        )
