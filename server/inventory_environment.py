"""
InventoryEnvironment — core simulation logic for the OpenEnv hackathon submission.

Three task tiers:
  easy   → 1 SKU,  fixed 2-day lead time, 90-day episode
  medium → 5 SKUs, random lead time 1-4, capacity constraint, 180-day episode
  hard   → 10 SKUs, random lead time + seasonality + random disruptions, 365-day episode
"""
from __future__ import annotations

import random
import math
from typing import Any, Dict, List, Optional

import numpy as np

# ── bring in the shared Pydantic models ───────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models import InventoryAction, InventoryObservation


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
WAREHOUSE_CAPACITY = 5000   # total units across all SKUs
HOLDING_COST_PER_UNIT = 0.05
STOCKOUT_PENALTY_PER_UNIT = 2.0
FIXED_ORDER_COST = 10.0     # flat cost whenever order_qty > 0
BASE_DEMAND_MEAN = 50
BASE_DEMAND_STD = 10


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _make_skus(n: int) -> List[str]:
    if n <= 26:
        return [f"SKU_{chr(65 + i)}" for i in range(n)]
    return [f"SKU_{i:03d}" for i in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

class InventoryEnvironment:
    """
    Gymnasium-inspired inventory replenishment environment.

    reset(task)  → InventoryObservation
    step(action) → InventoryObservation
    state        → InventoryObservation  (property)
    """

    def __init__(self) -> None:
        self._task: str = "easy"
        self._rng: np.random.Generator = np.random.default_rng(42)
        self._day: int = 0
        self._done: bool = False
        self._reward: float = 0.0
        self._skus: List[str] = []
        self._stock: Dict[str, int] = {}
        self._on_order: Dict[str, Dict[int, int]] = {}   # sku → {arrival_day: qty}
        self._demand_history: Dict[str, List[int]] = {}
        self._episode_length: int = 90
        self._lead_time: Dict[str, int] = {}
        self._capacity: int = WAREHOUSE_CAPACITY
        self._cumulative_stockout: int = 0
        self._cumulative_demand: int = 0

    # ── public API ──────────────────────────────────────────────────────────

    def reset(self, task: str = "easy") -> InventoryObservation:
        self._task = task
        seed = {"easy": 1, "medium": 2, "hard": 3}.get(task, 42)
        self._rng = np.random.default_rng(seed)
        self._day = 0
        self._done = False
        self._reward = 0.0
        self._cumulative_stockout = 0
        self._cumulative_demand = 0

        if task == "easy":
            self._skus = _make_skus(1)
            self._episode_length = 90
            self._capacity = WAREHOUSE_CAPACITY
            self._stock = {s: 200 for s in self._skus}
            self._lead_time = {s: 2 for s in self._skus}   # fixed

        elif task == "medium":
            self._skus = _make_skus(5)
            self._episode_length = 180
            self._capacity = 2500
            self._stock = {s: 200 for s in self._skus}
            self._lead_time = {}                            # random each order

        else:  # hard
            self._skus = _make_skus(10)
            self._episode_length = 365
            self._capacity = WAREHOUSE_CAPACITY
            self._stock = {s: 150 for s in self._skus}
            self._lead_time = {}

        self._on_order = {s: {} for s in self._skus}
        self._demand_history = {s: [0] * 7 for s in self._skus}

        return self.state

    def step(self, action: InventoryAction) -> InventoryObservation:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() before step().")

        self._day += 1

        # ── 1. Place order ──────────────────────────────────────────────────
        if action.sku_id in self._stock and action.order_qty > 0:
            lt = self._resolve_lead_time(action.sku_id)
            arrival = self._day + lt
            self._on_order[action.sku_id][arrival] = (
                self._on_order[action.sku_id].get(arrival, 0) + action.order_qty
            )

        # ── 2. Receive deliveries ───────────────────────────────────────────
        for sku in self._skus:
            if self._day in self._on_order[sku]:
                received = self._on_order[sku].pop(self._day)
                self._stock[sku] = min(
                    self._stock[sku] + received,
                    self._capacity // len(self._skus),
                )

        # ── 3. Simulate demand & fulfillment ───────────────────────────────
        step_stockout = 0
        step_demand = 0

        for sku in self._skus:
            mu = self._demand_mean(sku)
            raw_demand = int(np.clip(self._rng.normal(mu, BASE_DEMAND_STD), 0, mu * 3))
            # hard: random disruption (20% probability)
            if self._task == "hard" and self._rng.random() < 0.20:
                raw_demand = int(raw_demand * self._rng.uniform(1.5, 3.0))

            step_demand += raw_demand
            sold = min(self._stock[sku], raw_demand)
            shortfall = raw_demand - sold
            self._stock[sku] -= sold
            step_stockout += shortfall

            self._demand_history[sku].append(raw_demand)
            self._demand_history[sku] = self._demand_history[sku][-7:]

        self._cumulative_stockout += step_stockout
        self._cumulative_demand += step_demand

        # ── 4. Compute reward ───────────────────────────────────────────────
        holding_cost = HOLDING_COST_PER_UNIT * sum(self._stock.values())
        stockout_penalty = STOCKOUT_PENALTY_PER_UNIT * step_stockout
        order_cost = FIXED_ORDER_COST if action.order_qty > 0 else 0.0

        self._reward = -(holding_cost + stockout_penalty + order_cost)

        # ── 5. Episode termination ──────────────────────────────────────────
        if self._day >= self._episode_length:
            self._done = True

        return self.state

    @property
    def state(self) -> InventoryObservation:
        total_stock = sum(self._stock.values())
        on_order_flat = {sku: sum(self._on_order[sku].values()) for sku in self._skus}
        return InventoryObservation(
            stock_levels=dict(self._stock),
            on_order=on_order_flat,
            demand_history={k: list(v) for k, v in self._demand_history.items()},
            day=self._day,
            warehouse_fill_pct=round(total_stock / self._capacity, 4),
            reward=round(self._reward, 4),
            done=self._done,
            info={
                "cumulative_stockout": self._cumulative_stockout,
                "cumulative_demand": self._cumulative_demand,
                "task": self._task,
            },
        )

    # ── private helpers ─────────────────────────────────────────────────────

    def _resolve_lead_time(self, sku: str) -> int:
        if self._task == "easy":
            return 2
        if self._task == "medium":
            return int(self._rng.integers(1, 5))   # 1-4 days
        # hard: 1-7 days
        return int(self._rng.integers(1, 8))

    def _demand_mean(self, sku: str) -> float:
        mu = BASE_DEMAND_MEAN
        if self._task == "hard":
            # sinusoidal seasonality (period = 365 days)
            season = 1.0 + 0.4 * math.sin(2 * math.pi * self._day / 365)
            mu *= season
        return mu