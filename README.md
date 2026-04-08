# Inventory Replenishment — OpenEnv Hackathon Submission

> **Meta × Scaler OpenEnv Hackathon** | Supply Chain Inventory Replenishment

---

## Problem Description

A retail warehouse agent must decide **how much to reorder per SKU each day** to minimise total cost, which is the sum of:

| Cost component      | Formula                             |
|---------------------|-------------------------------------|
| Holding cost        | `0.05 × Σ stock_levels`             |
| Stockout penalty    | `2.0 × units_unfulfilled`           |
| Fixed order cost    | `10.0` if `order_qty > 0` else `0`  |

Dense reward: `reward = -(holding_cost + stockout_penalty + order_cost)`

---

## Environment Details

| Field              | Value                                                   |
|--------------------|---------------------------------------------------------|
| State space        | stock levels, on-order units, 7-day demand history, day |
| Action space       | `{sku_id: str, order_qty: int ∈ [0, 500]}`             |
| Episode end        | After N days (90 / 180 / 365 depending on tier)         |
| Stochasticity      | Normal demand + disruptions (hard tier)                 |

---

## Task Tiers

| Tier   | SKUs | Lead Time      | Capacity | Days | Extra features                     |
|--------|------|----------------|----------|------|------------------------------------|
| easy   | 1    | Fixed 2 days   | 5000     | 90   | —                                  |
| medium | 5    | Random 1–4 days| 2500     | 180  | Shared warehouse capacity           |
| hard   | 10   | Random 1–7 days| 5000     | 365  | Seasonality + random disruptions    |

---

## Project Structure

```
inventory_env/
├── __init__.py                  # Package exports
├── models.py                    # Pydantic Action / Observation
├── client.py                    # HTTP + WebSocket clients
├── openenv.yaml                 # Manifest
├── pyproject.toml
├── inference.py                 # LLM agent + grader
├── Dockerfile                   # Root-level, builds from project root
└── server/
    ├── app.py                   # FastAPI server (HTTP + WebSocket)
    ├── inventory_environment.py # Core simulation
    ├── requirements.txt
    └── Dockerfile               # Server-only build (alternative)
```

---

## Quickstart — Docker

```bash
# 1. Build (from project root — uses root Dockerfile)
docker build -t inventory-env .

# 2. Run
docker run -p 7860:7860 inventory-env

# 3. Test manually
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task":"easy"}'
curl -X POST http://localhost:7860/step  -H "Content-Type: application/json" \
     -d '{"sku_id":"SKU_A","order_qty":100}'

# 4. Interactive docs
open http://localhost:7860/docs
```

---

## Run the LLM Inference Script

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="hf_your_token_here"
export ENV_URL="http://localhost:7860"

pip install httpx openai pydantic
python inference.py
```

---

## Grading

```python
fill_rate = 1 - cumulative_stockout / cumulative_demand
norm_reward = (avg_step_reward + 300) / 300   # clipped to [0, 1]
score = 0.6 * fill_rate + 0.4 * norm_reward    # ∈ [0, 1]
```

---

## HuggingFace Spaces Deployment

```bash
pip install openenv
openenv push --repo-id your-username/inventory-env
```

---

## API Reference

| Endpoint      | Method | Body                          | Returns              |
|---------------|--------|-------------------------------|----------------------|
| `/health`     | GET    | —                             | `{"status":"healthy"}` |
| `/reset`      | POST   | `{"task": "easy"}`            | `InventoryObservation` |
| `/step`       | POST   | `{"sku_id":…, "order_qty":…}` | `InventoryObservation` |
| `/state`      | GET    | —                             | `InventoryObservation` |
| `/ws`         | WS     | JSON messages                 | `InventoryObservation` |
| `/docs`       | GET    | —                             | Swagger UI           |