import os
from openai import OpenAI
from client import DynapriceEnv
from models import DynapriceAction
from dotenv import load_dotenv

load_dotenv()

# Hackathon required environment variables
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4-turbo")
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def run_agent():
    print("[START] Inference Process Started")
    
    # Initialize the LLM Client using Hackathon standard variables
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN
    )

    env_url = "http://localhost:8000"
    print(f"[START] Connecting to environment at {env_url}")

    total_reward = 0.0
    
    with DynapriceEnv(base_url=env_url).sync() as env:
        # Step 1: Reset environment
        result = env.reset()
        obs = result.observation
        done = result.done

        print(f"[START] Initial State: Demand={obs.demand:.1f}, Supply={obs.supply:.1f}, Surge={obs.surge_multiplier:.1f}")
        
        while not done:
            # Prepare context for the LLM
            prompt = (
                f"You are managing dynamic pricing for a ride-sharing platform. "
                f"Current State: Demand is {obs.demand:.1f}, Supply is {obs.supply:.1f}, Surge Multiplier is {obs.surge_multiplier:.1f}. "
                f"Your goal is to balance demand and supply by tweaking the surge to maximize revenue and minimize penalties. "
                "Output a single integer representing your chosen action: "
                "0 (Decrease Surge), 1 (Keep Surge), 2 (Increase Surge)."
            )

            # Call LLM
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a reinforcement learning agent. Output only a single digit action."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=10,
                    temperature=0.0
                )
                
                action_text = response.choices[0].message.content.strip()
                action_int = int(action_text)
                if action_int not in [0, 1, 2]:
                    action_int = 1 # fallback
            except Exception as e:
                # Fallback to hold if LLM parsing fails
                action_int = 1

            # Execute action in the environment
            step_action = DynapriceAction(action=action_int)
            result = env.step(step_action)
            obs = result.observation
            done = result.done
            step_reward = result.reward if result.reward is not None else 0.0
            
            total_reward += step_reward

            print(f"[STEP] Action={action_int} | Reward={step_reward:.4f} | State(Demand={obs.demand:.1f}, Supply={obs.supply:.1f}, Surge={obs.surge_multiplier:.1f})")

        print(f"[END] Episode Finished | Total Reward: {total_reward:.4f}")

if __name__ == "__main__":
    run_agent()
