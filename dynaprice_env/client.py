# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dynaprice Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import DynapriceAction, DynapriceObservation
except ImportError:
    from models import DynapriceAction, DynapriceObservation


class DynapriceEnv(
    EnvClient[DynapriceAction, DynapriceObservation, State]
):
    """
    Client for the Dynaprice Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with DynapriceEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(DynapriceAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = DynapriceEnv.from_docker_image("dynaprice_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(DynapriceAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: DynapriceAction) -> Dict:
        """
        Convert DynapriceAction to JSON payload for step message.

        Args:
            action: DynapriceAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action": action.action,
        }

    def _parse_result(self, payload: Dict) -> StepResult[DynapriceObservation]:
        """
        Parse server response into StepResult[DynapriceObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with DynapriceObservation
        """
        obs_data = payload.get("observation", {})
        observation = DynapriceObservation(
            demand=obs_data.get("demand", 0.0),
            supply=obs_data.get("supply", 0.0),
            surge_multiplier=obs_data.get("surge_multiplier", 1.0),
            time_step=obs_data.get("time_step", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
