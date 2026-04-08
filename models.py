"""
Pydantic models for the Inventory Replenishment OpenEnv environment.
Action  → InventoryAction
Observation → InventoryObservation
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class InventoryAction(BaseModel):
    """Reorder decision: which SKU to restock and by how much."""

    sku_id: str = Field(
        ...,
        description="Unique identifier of the SKU (or 'SKU@warehouse' in hard mode).",
    )
    order_qty: int = Field(
        default=0,
        ge=0,
        le=500,
        description="Units to order (0–500). 0 means no order this step.",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class InventoryObservation(BaseModel):
    """Full environment state returned after every reset() / step()."""

    stock_levels: Dict[str, int] = Field(
        ..., description="Current on-hand stock per SKU."
    )
    on_order: Dict[str, int] = Field(
        ..., description="Units in transit (ordered but not yet arrived) per SKU."
    )
    demand_history: Dict[str, List[int]] = Field(
        ..., description="Last 7 days of observed demand per SKU."
    )
    day: int = Field(..., description="Current simulation day (0-indexed).")
    warehouse_fill_pct: float = Field(
        ..., description="Fraction of total warehouse capacity currently occupied."
    )
    reward: float = Field(
        default=0.0, description="Reward for the last step (negative cost)."
    )
    done: bool = Field(default=False, description="True when the episode has ended.")
    info: Dict[str, Any] = Field(
        default_factory=dict, description="Auxiliary diagnostic info."
    )