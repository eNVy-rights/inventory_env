"""Start server and run API tests in the same process."""
import sys
import os
import threading
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.join(ROOT, 'server')
sys.path.insert(0, ROOT)
sys.path.insert(0, SERVER)

import uvicorn
from app import app  # imports from server/

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=7861, log_level="warning")

# Start server in thread
t = threading.Thread(target=run_server, daemon=True)
t.start()
time.sleep(2.5)  # wait for server

# Now test
import httpx
BASE = "http://127.0.0.1:7861"
PASS = 0
FAIL = 0

def check(label, cond, detail=""):
    global PASS, FAIL
    icon = "✅" if cond else "❌"
    print(f"  {icon} {label}" + (f" — {detail}" if detail else ""))
    if cond: PASS += 1
    else: FAIL += 1

c = httpx.Client(base_url=BASE, timeout=10)

print("\n=== Testing Endpoints ===")

# Health
r = c.get("/health")
check("GET /health 200", r.status_code == 200)
check("/health returns healthy", r.json().get("status") == "healthy", str(r.json()))

# Reset easy
r = c.post("/reset", json={"task": "easy"})
check("POST /reset easy 200", r.status_code == 200)
obs = r.json()
check("easy: day=0", obs["day"] == 0)
check("easy: 1 SKU (SKU_A)", list(obs["stock_levels"].keys()) == ["SKU_A"], str(list(obs["stock_levels"].keys())))
check("easy: not done", not obs["done"])

# Step
r = c.post("/step", json={"sku_id": "SKU_A", "order_qty": 100})
check("POST /step 200", r.status_code == 200)
obs = r.json()
check("step: day=1", obs["day"] == 1)
check("step: negative reward (cost)", obs["reward"] < 0, f"reward={obs['reward']}")

# Docs endpoint
r = c.get("/docs")
check("GET /docs 200", r.status_code == 200)

# State
r = c.get("/state")
check("GET /state 200", r.status_code == 200)
check("state: day=1", r.json()["day"] == 1)

# Medium
r = c.post("/reset", json={"task": "medium"})
check("POST /reset medium 200", r.status_code == 200)
obs = r.json()
check("medium: 5 SKUs", len(obs["stock_levels"]) == 5, str(list(obs["stock_levels"].keys())))

# Hard
r = c.post("/reset", json={"task": "hard"})
check("POST /reset hard 200", r.status_code == 200)
obs = r.json()
check("hard: 10 SKUs", len(obs["stock_levels"]) == 10)

# Run 5-step hard episode
for i in range(5):
    sku = sorted(obs["stock_levels"].keys())[0]
    r2 = c.post("/step", json={"sku_id": sku, "order_qty": 50})
    obs = r2.json()
check("5-step episode completes", obs["day"] == 5, f"day={obs['day']}, reward={obs['reward']}")

# Grader logic test
sys.path.insert(0, ROOT)
# inline grade check
def grade(rewards, stockout, demand):
    fill = max(0.0, 1 - stockout / max(demand, 1))
    avg_r = sum(rewards) / max(len(rewards), 1)
    norm = min(1.0, max(0.0, (avg_r + 300) / 300))
    return round(0.6 * fill + 0.4 * norm, 4)

s = grade([-100]*30, 200, 1500)
check("Grader produces score in [0,1]", 0 <= s <= 1, f"score={s}")

print(f"\n{'='*55}")
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("🎉 ALL TESTS PASSED — Server is fully functional!")
else:
    print(f"⚠️  {FAIL} test(s) failed")
print(f"{'='*55}")
sys.exit(0 if FAIL == 0 else 1)
