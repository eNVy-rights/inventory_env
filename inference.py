"""
inference.py — LLM-driven agent for the Inventory Replenishment OpenEnv environment.

Environment variables required:
  API_BASE_URL   → OpenAI-compatible API base, e.g. https://api-inference.huggingface.co/v1
  MODEL_NAME     → e.g. meta-llama/Llama-3.3-70B-Instruct
  HF_TOKEN       → Your HuggingFace token (used as API key)
  ENV_URL        → HTTP base URL of the running environment server (default: http://localhost:7860)

Run:
  python inference.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import List

# ── OpenAI client ─────────────────────────────────────────────────────────────
from openai import OpenAI

# ── Local env client & models ─────────────────────────────────────────────────
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from models import InventoryAction, InventoryObservation
from client import InventoryClient


# ─────────────────────────────────────────────────────────────────────────────
# Config from environment variables
# ─────────────────────────────────────────────────────────────────────────────
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
ENV_URL: str = os.environ.get("ENV_URL", "http://localhost:7860")

llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


# ─────────────────────────────────────────────────────────────────────────────
# Grader
# ─────────────────────────────────────────────────────────────────────────────

def grade_episode(rewards: List[float], total_stockout: int, total_demand: int) -> float:
    """
    Composite score ∈ [0, 1].
      60% service fill rate  (1 - stockout / demand)
      40% normalised reward  (shift reward into [0,1] from expected range [-300, 0])
    """
    fill_rate = max(0.0, 1.0 - total_stockout / max(total_demand, 1))
    avg_reward = sum(rewards) / max(len(rewards), 1)
    norm_reward = min(1.0, max(0.0, (avg_reward + 300.0) / 300.0))
    score = 0.6 * fill_rate + 0.4 * norm_reward
    return round(score, 4)


# ─────────────────────────────────────────────────────────────────────────────
# LLM agent
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert supply chain optimisation agent.
Each turn you receive a JSON observation describing the current warehouse state.
You must respond with ONLY a single valid JSON object — no explanation, no markdown:

{"sku_id": "<sku_id>", "order_qty": <integer 0-500>}

Guidelines:
- If stock_levels for a SKU is LOW (< 100) and on_order is also LOW, order aggressively (200-400).
- If stock_levels is COMFORTABLE (> 300), order 0 to avoid holding costs.
- Prioritise SKUs with the highest recent demand from demand_history.
- When done is true, your response is ignored.
"""


def get_action(obs: InventoryObservation) -> InventoryAction:
    """Call the LLM and parse its action. Falls back to a heuristic on failure."""
    try:
        response = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs.model_dump_json()},
            ],
            max_tokens=64,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        data = json.loads(raw)
        return InventoryAction(**data)
    except Exception as exc:
        # Heuristic fallback: order for the SKU with lowest stock
        fallback_sku = min(obs.stock_levels, key=obs.stock_levels.get)
        qty = 200 if obs.stock_levels[fallback_sku] < 100 else 0
        return InventoryAction(sku_id=fallback_sku, order_qty=qty)


# ─────────────────────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────────────────────

def run_task(task: str, env: InventoryClient) -> float:
    print(f"\n{'='*60}")
    print(f"  Task: {task.upper()}")
    print(f"{'='*60}")

    obs = env.reset(task=task)
    rewards: List[float] = []
    step_count = 0

    while not obs.done:
        action = get_action(obs)
        obs = env.step(action)
        rewards.append(obs.reward)
        step_count += 1

        if step_count % 30 == 0:
            info = obs.info
            print(
                f"  Day {obs.day:>3} | reward={obs.reward:>8.2f} "
                f"| fill={obs.warehouse_fill_pct:.2%} "
                f"| stockout={info.get('cumulative_stockout', '?')}"
            )

    info = obs.info
    total_stockout = info.get("cumulative_stockout", 0)
    total_demand = info.get("cumulative_demand", 1)
    score = grade_episode(rewards, total_stockout, total_demand)

    print(f"\n  Episode complete — {step_count} steps")
    print(f"  Total reward   : {sum(rewards):.2f}")
    print(f"  Fill rate      : {max(0, 1 - total_stockout/max(total_demand,1)):.2%}")
    print(f"  Score          : {score}")
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Connecting to environment at {ENV_URL} …")
    with InventoryClient(base_url=ENV_URL) as env:
        # Confirm server is healthy
        health = env.health()
        print(f"Server health: {health}")

        scores = {}
        for task in ["easy", "medium", "hard"]:
            scores[task] = run_task(task, env)
            time.sleep(0.5)

    print("\n" + "="*60)
    print("FINAL SCORES")
    print("="*60)
    for task, score in scores.items():
        print(f"  {task:<8}: {score:.4f}")
    overall = sum(scores.values()) / len(scores)
    print(f"  {'OVERALL':<8}: {overall:.4f}")
    print("="*60)