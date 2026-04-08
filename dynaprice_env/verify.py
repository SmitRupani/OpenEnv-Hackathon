import time
from server.dynaprice_env_environment import DynapriceEnvironment
from models import DynapriceAction
import random

def main():
    print("Testing dynamic pricing environment logic directly...")
    
    try:
        env = DynapriceEnvironment()
        obs = env.reset()
        print(f"Initial State: Demand={obs.demand:.1f}, Supply={obs.supply:.1f}, Surge={obs.surge_multiplier:.1f}")

        total_reward = 0
        for i in range(20):
            # Select random action: 0 (decrease), 1 (keep), 2 (increase)
            action = random.choice([0, 1, 2])
            
            # Step the environment
            obs = env.step(DynapriceAction(action=action))
            total_reward += obs.reward
            
            # Log output
            print(f"Step {i+1:02d} | Action={action} | Demand={obs.demand:5.1f} | Supply={obs.supply:5.1f} | Surge={obs.surge_multiplier:.1f} | Reward={obs.reward:8.2f}")

            if obs.done:
                print("Episode finished early!")
                break

        print("-" * 50)
        print(f"Test complete. Total accumulated reward over 20 random steps: {total_reward:.2f}")
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    main()
