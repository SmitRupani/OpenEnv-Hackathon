import random

try:
    from dynaprice_env.server.dynaprice_env_environment import DynapriceEnvironment
    from dynaprice_env.models import DynapriceAction
except ImportError:
    from server.dynaprice_env_environment import DynapriceEnvironment
    from models import DynapriceAction


def _assert_open_interval(name: str, value: float) -> None:
    if not (0.0 < value < 1.0):
        raise ValueError(f"{name} must be strictly between 0 and 1, got {value}")

def main():
    print("Testing dynamic pricing environment logic directly...")
    
    try:
        env = DynapriceEnvironment()
        obs = env.reset()
        print(
            f"Initial State: Demand={obs.demand:.1f}, Supply={obs.supply:.1f}, Surge={obs.surge_multiplier:.1f}"
        )

        task_suite = getattr(env, "task_grader_suite", None) or getattr(env, "rubric", None)
        if task_suite is None or not hasattr(task_suite, "task_scores"):
            raise RuntimeError("Environment does not expose a task grader suite")

        task_names = [name for name, _ in task_suite.named_rubrics() if name in {"easy", "medium", "hard"}]
        if len(set(task_names)) < 3:
            raise RuntimeError(f"Expected 3 grader-backed tasks, found: {task_names}")

        if getattr(obs, "reward", None) is not None:
            _assert_open_interval("reset reward", float(obs.reward))

        total_reward = 0
        for i in range(20):
            # Select random action: 0 (decrease), 1 (keep), 2 (increase)
            action = random.choice([0, 1, 2])
            
            # Step the environment
            obs = env.step(DynapriceAction(action=action))
            if obs.reward is None:
                raise RuntimeError("Step reward unexpectedly missing")
            _assert_open_interval("step reward", float(obs.reward))

            task_scores = task_suite.task_scores(DynapriceAction(action=action), obs)
            for task_name, task_score in task_scores.items():
                _assert_open_interval(f"{task_name} task score", float(task_score))

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
