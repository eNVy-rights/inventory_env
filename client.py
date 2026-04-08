"""
HTTP + WebSocket client for the Inventory Replenishment environment.

Usage (HTTP):
    client = InventoryClient(base_url="http://localhost:7860")
    obs = client.reset("easy")
    while not obs.done:
        action = InventoryAction(sku_id=list(obs.stock_levels.keys())[0], order_qty=100)
        obs = client.step(action)

Usage (WebSocket, async):
    async with InventoryWSClient("ws://localhost:7860/ws") as client:
        obs = await client.reset("easy")
        while not obs.done:
            obs = await client.step(InventoryAction(sku_id="SKU_A", order_qty=50))
"""
from __future__ import annotations

import json
import asyncio
from typing import Optional

import httpx

try:
    from .models import InventoryAction, InventoryObservation
except ImportError:
    from models import InventoryAction, InventoryObservation


# ─────────────────────────────────────────────────────────────────────────────
# Synchronous HTTP client (simple, good for inference.py)
# ─────────────────────────────────────────────────────────────────────────────

class InventoryClient:
    """Thin synchronous HTTP client wrapping the FastAPI server."""

    def __init__(self, base_url: str = "http://localhost:7860") -> None:
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, timeout=30.0)

    # ── API ──────────────────────────────────────────────────────────────────

    def health(self) -> dict:
        return self._http.get("/health").json()

    def reset(self, task: str = "easy") -> InventoryObservation:
        r = self._http.post("/reset", json={"task": task})
        r.raise_for_status()
        return InventoryObservation.model_validate(r.json())

    def step(self, action: InventoryAction) -> InventoryObservation:
        r = self._http.post("/step", json=action.model_dump())
        r.raise_for_status()
        return InventoryObservation.model_validate(r.json())

    def state(self) -> InventoryObservation:
        r = self._http.get("/state")
        r.raise_for_status()
        return InventoryObservation.model_validate(r.json())

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ─────────────────────────────────────────────────────────────────────────────
# Async WebSocket client (good for parallel RL loops)
# ─────────────────────────────────────────────────────────────────────────────

class InventoryWSClient:
    """Async WebSocket client for the /ws endpoint."""

    def __init__(self, ws_url: str = "ws://localhost:7860/ws") -> None:
        self.ws_url = ws_url
        self._ws = None

    async def __aenter__(self):
        import websockets
        self._ws = await websockets.connect(self.ws_url)
        return self

    async def __aexit__(self, *_):
        if self._ws:
            await self._ws.close()

    async def reset(self, task: str = "easy") -> InventoryObservation:
        await self._ws.send(json.dumps({"type": "reset", "task": task}))
        raw = await self._ws.recv()
        return InventoryObservation.model_validate_json(raw)

    async def step(self, action: InventoryAction) -> InventoryObservation:
        await self._ws.send(json.dumps({"type": "step", "action": action.model_dump()}))
        raw = await self._ws.recv()
        return InventoryObservation.model_validate_json(raw)

    async def state(self) -> InventoryObservation:
        await self._ws.send(json.dumps({"type": "state"}))
        raw = await self._ws.recv()
        return InventoryObservation.model_validate_json(raw)