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
The environment samples one of three difficulty contexts natively on environment `.reset()` logic:
- `easy`: Standard equilibrium where supply perfectly offsets demand at standard multipliers.
- `medium`: Extreme demand spike events requiring the agent to rapidly step-up surge barriers safely to cap losses.
- `hard`: Volatile timezone variations where random massive permutations of logic decay the baseline elasticity dynamically across standard evaluation steps.

Score tracking (`reward`) automatically normalizes based on bounded metric caps guaranteeing all grader emissions land reliably between `0.0` to `1.0`.

### Action Space (Discrete)
`DynapriceAction`: integer
- `0`: Decrease surge multiplier by `0.1`
- `1`: Maintain current surge matrix
- `2`: Increase surge multiplier by `0.1`

### Observation Space
`DynapriceObservation`: mapped object
- `demand` (float): Current rider requested tasks in-grid.
- `supply` (float): Current driver idle availability.
- `surge_multiplier` (float): Global surge state context index.
- `time_step` (int): Active temporal loop counter terminating at 50 increments.
- `reward` (float): Bound normalized step grade mapped dynamically against standard theoretical revenue limits.
- `metadata` (dict): Verbose JSON block exposing deep internal parameters like raw wait penalties and actual matching volume.
