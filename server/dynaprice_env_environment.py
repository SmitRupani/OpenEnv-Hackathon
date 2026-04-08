# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Dynaprice Env Environment Implementation.

A dynamic pricing environment for a ride-sharing simulation.
"""

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import DynapriceAction, DynapriceObservation
except ImportError:
    from models import DynapriceAction, DynapriceObservation


class DynapriceEnvironment(Environment):
    """
    A dynamic pricing environment.
    
    The agent observes the market (demand, supply, surge_multiplier, time_step)
    and chooses an action to modify the surge multiplier.
    The goal is to maximize revenue while minimizing penalties.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the dynaprice_env environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        # Environment configuration
        self.base_rate_per_km = 2.0
        self.average_distance = 5.0
        self.base_fare = self.base_rate_per_km * self.average_distance
        
        self.max_steps = 50
        self.elasticity_d = 0.2
        self.elasticity_s = 0.2
        
        self.task_id = "easy"
        self._setup_task("easy")

    def _setup_task(self, difficulty: str):
        self.task_id = difficulty
        if difficulty == "easy":
            # Normal balanced state
            self.surge_multiplier = 1.0
            self.base_demand = 100.0
            self.base_supply = 100.0
        elif difficulty == "medium":
            # Demand spike, agent must quickly increase surge
            self.surge_multiplier = 1.0
            self.base_demand = 300.0
            self.base_supply = 80.0
        elif difficulty == "hard":
            # Supply shortage, extreme unbalance
            self.surge_multiplier = 1.0
            self.base_demand = 200.0
            self.base_supply = 40.0
            
        self.demand = self.base_demand
        self.supply = self.base_supply

    def reset(self) -> DynapriceObservation:
        """
        Reset the environment.

        Returns:
            DynapriceObservation with initial market conditions
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        # Randomly select a task
        tasks = ["easy", "medium", "hard"]
        self._setup_task(random.choice(tasks))
        
        # Add slight initial stochastic noise
        self.demand = max(0.0, self.base_demand + random.uniform(-10, 10))
        self.supply = max(0.0, self.base_supply + random.uniform(-10, 10))

        return DynapriceObservation(
            demand=self.demand,
            supply=self.supply,
            surge_multiplier=self.surge_multiplier,
            time_step=self._state.step_count,
            done=False,
            reward=0.0,
        )

    def step(self, action: DynapriceAction) -> DynapriceObservation:  # type: ignore[override]
        """
        Execute a step in the environment by adjusting surge and processing market response.

        Args:
            action: DynapriceAction (0: decrease, 1: keep, 2: increase)

        Returns:
            DynapriceObservation with latest market conditions
        """
        self._state.step_count += 1

        # Process action to adjust surge multiplier
        if action.action == 0:
            self.surge_multiplier = max(1.0, self.surge_multiplier - 0.1)
        elif action.action == 2:
            self.surge_multiplier = min(3.0, self.surge_multiplier + 0.1)
            
        # Hard task introduces dynamic shifts over time
        if self.task_id == "hard" and self._state.step_count % 10 == 0:
             self.base_demand += random.uniform(-40, 40)
             self.base_supply += random.uniform(-20, 20)
            
        # Update demand and supply with stochastic noise
        noise_d = random.uniform(-5, 5)
        noise_s = random.uniform(-5, 5)
        
        self.demand = max(0.0, self.base_demand * (1 - self.elasticity_d * (self.surge_multiplier - 1.0)) + noise_d)
        self.supply = max(0.0, self.base_supply * (1 + self.elasticity_s * (self.surge_multiplier - 1.0)) + noise_s)
        
        # Match rides
        completed_rides = min(self.demand, self.supply)
        
        # Compute raw revenue and penalties
        revenue = completed_rides * self.base_fare * self.surge_multiplier
        
        wait_penalty = max(0.0, self.demand - self.supply) * 2.0
        surge_penalty = max(0.0, self.surge_multiplier - 2.0) * 100.0
        
        raw_reward = revenue - wait_penalty - surge_penalty
        
        # Normalize reward between 0.0 and 1.0
        # Assume max theoretical revenue around ~400 units * 10 fare * 3 surge = ~12000
        # If penalty outweighs revenue, clip to 0.0
        theoretical_max = 12000.0
        normalized_reward = max(0.0, min(1.0, raw_reward / theoretical_max))

        done = self._state.step_count >= self.max_steps

        return DynapriceObservation(
            demand=self.demand,
            supply=self.supply,
            surge_multiplier=self.surge_multiplier,
            time_step=self._state.step_count,
            done=done,
            reward=normalized_reward,
            metadata={
                "task_id": self.task_id,
                "raw_reward": raw_reward,
                "completed_rides": completed_rides,
                "revenue": revenue,
                "wait_penalty": wait_penalty,
                "surge_penalty": surge_penalty
            },
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
