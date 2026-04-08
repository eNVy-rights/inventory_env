"""HTTP API smoke test against the running local server."""
import httpx
import json
import sys

BASE = "http://localhost:7860"
PASS = 0
FAIL = 0

def check(label, cond, detail=""):
    global PASS, FAIL
    icon = "✅" if cond else "❌"
    print(f"  {icon} {label}" + (f" — {detail}" if detail else ""))
    if cond:
        PASS += 1
    else:
        FAIL += 1

try:
    c = httpx.Client(base_url=BASE, timeout=10)

    # Health
    r = c.get("/health")
    check("GET /health 200", r.status_code == 200, str(r.json()))
    check("/health returns healthy", r.json().get("status") == "healthy")

    # Reset easy
    r = c.post("/reset", json={"task": "easy"})
    check("POST /reset easy 200", r.status_code == 200)
    obs = r.json()
    check("reset easy: day=0", obs["day"] == 0)
    check("reset easy: 1 SKU", len(obs["stock_levels"]) == 1, str(list(obs["stock_levels"].keys())))
    check("reset easy: done=False", obs["done"] == False)

    # Step
    r = c.post("/step", json={"sku_id": "SKU_A", "order_qty": 100})
    check("POST /step 200", r.status_code == 200)
    obs = r.json()
    check("step: day=1", obs["day"] == 1)
    check("step: reward is float", isinstance(obs["reward"], float), str(obs["reward"]))
    check("step: done=False (ep len 90)", obs["done"] == False)

    # State
    r = c.get("/state")
    check("GET /state 200", r.status_code == 200)
    check("state: day same", r.json()["day"] == 1)

    # Reset medium
    r = c.post("/reset", json={"task": "medium"})
    check("POST /reset medium 200", r.status_code == 200)
    obs = r.json()
    check("reset medium: 5 SKUs", len(obs["stock_levels"]) == 5, str(list(obs["stock_levels"].keys())))

    # Reset hard
    r = c.post("/reset", json={"task": "hard"})
    check("POST /reset hard 200", r.status_code == 200)
    obs = r.json()
    check("reset hard: 10 SKUs", len(obs["stock_levels"]) == 10, str(list(obs["stock_levels"].keys())))

    # Run a short hard episode (10 steps)
    r = c.post("/reset", json={"task": "hard"})
    obs = r.json()
    for i in range(10):
        sku = list(obs["stock_levels"].keys())[0]
        r = c.post("/step", json={"sku_id": sku, "order_qty": 80})
        obs = r.json()
    check("10-step hard episode runs", obs["day"] == 10, f"reward={obs['reward']}")

    print(f"\n{'='*50}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL == 0:
        print("🎉 ALL TESTS PASSED — Server is fully functional!")
    else:
        print(f"⚠️  {FAIL} test(s) failed — check output above")
    print(f"{'='*50}")
    sys.exit(0 if FAIL == 0 else 1)

except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("Make sure the server is running: py -3.12 -m uvicorn app:app --port 7860")
    sys.exit(1)
