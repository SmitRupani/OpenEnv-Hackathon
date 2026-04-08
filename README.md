---
title: Dynaprice Env Environment Server
emoji: 🚗
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Ride-Sharing Dynamic Pricing Environment

A realistic dynamic pricing reinforcement learning environment based on the OpenEnv specification. Built for the **Meta PyTorch Hackathon**.

The environment simulates a ride-sharing platform where an AI agent dynamically adjusts the surge multiplier to balance rider demand and driver supply while maximizing net platform revenue. Pushing surge too high causes dramatic demand decay, and persistent supply shortages invoke strict waiting penalties.

## Setup Instructions

1. Clone this repository locally.
2. Install Python requirements from the existing Python project or `uv` environments.
3. Configure your endpoint configurations using the `.env` template provided.
   ```bash
   cp .env.example .env
   # Ensure HF_TOKEN is injected!
   ```
4. Start the Environment Server locally:
   ```bash
   python -m server.app --port 8000
   ```
5. Trigger your LLM inference agent in a separate terminal:
   ```bash
   python inference.py
   ```

## Deploying to Hugging Face Spaces

This environment is containerized. It automatically builds via Docker using the provided `Dockerfile` placed in the root directory.

To deploy via `openenv` CLI framework natively:
```bash
openenv push --repo-id <your-hf-username>/dynaprice
```
Or simply create a new Space on [Hugging Face](https://huggingface.co/spaces), select **Docker** as the SDK, and upload the repository contents.

## Environment Specifications

### Tasks / Graders
The environment exposes three explicit grader-backed tasks, and `.reset()` samples one of them:
- `easy`: Balanced market conditions with a preference for steady matching and moderate surge.
- `medium`: Demand-spike conditions that reward faster surge adjustments and lower wait penalties.
- `hard`: Volatile supply conditions that reward stability under shocks and disciplined surge control.

Each step returns an active task score in the open interval `(0, 1)`. The full task breakdown is also exposed in `metadata.task_scores`, which contains `easy`, `medium`, and `hard` grader outputs, each strictly between `0` and `1`.

### Action Space (Discrete)
`DynapriceAction`: integer
- `0`: Decrease surge multiplier by `0.1`
- `1`: Maintain current surge matrix
- `2`: Increase surge multiplier by `0.1`

### Observation Space
`DynapriceObservation`: mapped object
- `demand` (float): Current rider demand.
- `supply` (float): Current driver availability.
- `surge_multiplier` (float): Current surge multiplier.
- `time_step` (int): Active time-step counter terminating at 50 increments.
- `reward` (float): Active grader score, always in the open interval `(0, 1)`.
- `metadata` (dict): Internal metrics, including `task_id`, `raw_reward`, and `task_scores` for all three graders.
