# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Dynaprice Env Environment.

The dynaprice_env environment simulates a dynamic pricing algorithm for a ride-sharing service.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class DynapriceAction(Action):
    """
    Action for the Dynaprice Env environment to control surge multiplier.
    0: Decrease surge
    1: Keep surge
    2: Increase surge
    """
    action: int = Field(..., description="Action to adjust surge (0: decrease, 1: keep, 2: increase)")


class DynapriceObservation(Observation):
    """Observation from the Dynaprice Env environment tracking market conditions."""

    demand: float = Field(default=0.0, description="Number of ride requests (demand)")
    supply: float = Field(default=0.0, description="Number of available drivers (supply)")
    surge_multiplier: float = Field(default=1.0, description="Current pricing multiplier")
    time_step: int = Field(default=0, description="Current time step within the episode")
