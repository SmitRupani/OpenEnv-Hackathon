---
title: DynaPrice Environment
emoji: 🚕
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
---
# DynaPrice — Dynamic Pricing RL Environment

An `openenv`-compatible reinforcement learning environment for testing causal reasoning and multi-step investigation in an AI Agent focused on dynamic pricing for ride-sharing.

The environment simulates a real-world ride-sharing marketplace experiencing supply/demand issues, weather events, and competitor changes. The agent must:
- Parse vague initial alerts.
- Execute diagnostic checks on `demand`, `supply`, `weather`, and `competitor_prices` for particular zones.
- Decide on the correct `pricing_strategy` for the affected `zone`.

The environment uses a **dense, gated reward structure**, which heavily penalizes guessing the pricing strategy without executing the correct sequence of diagnostic queries.

## Action Space
All actions are JSON objects with an `action` field:

| Action | Required Fields | Description |
|---|---|---|
| `query_demand` | `zone`: string | Check the local demand for a given zone. |
| `query_supply` | `zone`: string | Check the local driver supply for a given zone. |
| `query_weather` | `zone`: string | Check local weather which could impact ETAs or trigger red herrings. |
| `query_competitor_prices` | `zone`: string | Query whether competitors are undercutting or surging. |
| `set_price` | `pricing_strategy`: `normal` / `high_surge` / `weather_surge` / `match_competitor`, `zone`: string | Execute the terminal action of setting the optimal price multiplier. |

## Tasks

1. **`easy_surge_pricing`** (Easy)
   - Scenario: High demand, low driver supply in Downtown.
   - Optimal path: Query demand, query supply, set `high_surge`.

2. **`medium_weather_event`** (Medium)
   - Scenario: High pickup ETAs in Suburb. Weather causes traffic.
   - Optimal path: Query demand, query weather, set `weather_surge`. Setting a generic surge without diagnosing the weather will yield less than half the points.

3. **`hard_competitor_drop`** (Hard)
   - Scenario: Major drop in conversion rate in Airport. Weather sensor reports fog (red herring).
   - Optimal path: Ignore the runbook's weather hint. Query competitor prices to find they dropped them by 20%. Set `match_competitor`.

## Usage
Simply run the API via the standard `uvicorn` entry point or use the containerized setup:

```bash
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```

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
