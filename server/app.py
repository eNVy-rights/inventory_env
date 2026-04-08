"""
FastAPI server for the Inventory Replenishment OpenEnv environment.

Endpoints:
  GET  /health          → {"status": "healthy"}
  POST /reset           → InventoryObservation  (body: {"task": "easy|medium|hard"})
  POST /step            → InventoryObservation  (body: InventoryAction)
  GET  /state           → InventoryObservation  (current state, no side-effects)
  GET  /docs            → Swagger UI (built-in)

WebSocket /ws is provided for real-time RL loops:
  Client sends: {"type": "reset", "task": "easy"} | {"type": "step", "action": {...}}
  Server sends: InventoryObservation JSON
"""
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import sys
import os

# Ensure the server dir (where this file lives) and its parent are on sys.path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from inventory_environment import InventoryEnvironment
from models import InventoryAction, InventoryObservation

logger = logging.getLogger("inventory_env")
logging.basicConfig(level=logging.INFO)

# ── Global environment instance (one per server process) ─────────────────────
_env = InventoryEnvironment()


# ── App factory ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("InventoryEnvironment server starting up ✅")
    yield
    logger.info("InventoryEnvironment server shutting down")


app = FastAPI(
    title="Inventory Replenishment Environment",
    description=(
        "OpenEnv-compatible supply-chain RL environment. "
        "Three difficulty tiers: easy, medium, hard."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Helper models ─────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: str = "easy"


# ── HTTP endpoints ────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
async def health():
    return {"status": "healthy"}


@app.post("/reset", response_model=InventoryObservation, tags=["env"])
async def reset(req: ResetRequest):
    """Reset the environment for a given task tier and return the initial observation."""
    obs = _env.reset(task=req.task)
    return obs


@app.post("/step", response_model=InventoryObservation, tags=["env"])
async def step(action: InventoryAction):
    """Apply an action and return the next observation."""
    obs = _env.step(action)
    return obs


@app.get("/state", response_model=InventoryObservation, tags=["env"])
async def state():
    """Return current environment state without side-effects."""
    return _env.state


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Real-time RL loop over WebSocket.

    Send JSON messages:
      {"type": "reset", "task": "easy"}
      {"type": "step", "action": {"sku_id": "SKU_A", "order_qty": 100}}
      {"type": "state"}

    Receive: serialised InventoryObservation JSON.
    """
    await websocket.accept()
    logger.info("WebSocket client connected")
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "invalid JSON"}))
                continue

            msg_type = msg.get("type")

            if msg_type == "reset":
                task = msg.get("task", "easy")
                obs = _env.reset(task=task)
                await websocket.send_text(obs.model_dump_json())

            elif msg_type == "step":
                action_data = msg.get("action", {})
                try:
                    action = InventoryAction(**action_data)
                    obs = _env.step(action)
                    await websocket.send_text(obs.model_dump_json())
                except Exception as exc:
                    await websocket.send_text(json.dumps({"error": str(exc)}))

            elif msg_type == "state":
                await websocket.send_text(_env.state.model_dump_json())

            else:
                await websocket.send_text(json.dumps({"error": f"unknown type: {msg_type}"}))

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")