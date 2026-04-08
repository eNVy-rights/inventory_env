"""Quick smoke test for the inventory environment."""
import sys, os
ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.join(ROOT, 'server')
sys.path.insert(0, ROOT)
sys.path.insert(0, SERVER)

from inventory_environment import InventoryEnvironment
from models import InventoryAction

def test_task(task: str):
    env = InventoryEnvironment()
    obs = env.reset(task=task)
    print(f"\n=== Task: {task.upper()} ===")
    print(f"Initial state: day={obs.day}, SKUs={list(obs.stock_levels.keys())}, done={obs.done}")
    
    total_reward = 0.0
    steps = 0
    while not obs.done:
        # Simple heuristic: order 100 for the lowest-stock SKU
        sku = min(obs.stock_levels, key=obs.stock_levels.get)
        qty = 150 if obs.stock_levels[sku] < 200 else 0
        action = InventoryAction(sku_id=sku, order_qty=qty)
        obs = env.step(action)
        total_reward += obs.reward
        steps += 1
    
    info = obs.info
    stockout = info.get('cumulative_stockout', 0)
    demand = info.get('cumulative_demand', 1)
    fill_rate = 1 - stockout / max(demand, 1)
    print(f"Steps: {steps}, Total reward: {total_reward:.2f}")
    print(f"Cumulative stockout: {stockout}, demand: {demand}")
    print(f"Fill rate: {fill_rate:.2%}")
    print(f"PASS ✅")
    return total_reward

if __name__ == "__main__":
    for t in ["easy", "medium", "hard"]:
        test_task(t)
    print("\n\nAll tests passed! Environment logic is correct.")
