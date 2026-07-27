import urllib.request
import json
import sys

BASE_URL = "http://127.0.0.1:8080"

def request(method, path, body=None):
    url = f"{BASE_URL}{path}"
    data = json.dumps(body).encode("utf-8") if body else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except:
            return e.code, None
    except Exception as e:
        print(f"Error connecting to server: {e}")
        sys.exit(1)

def run_tests():
    print("Testing /v1/healthz...")
    status, res = request("GET", "/v1/healthz")
    assert status == 200, f"Expected 200, got {status}"
    assert res["status"] == "ok", f"Expected 'ok', got {res['status']}"
    assert "contexts_loaded" in res, "Missing contexts_loaded"
    print("healthz OK:", res)

    print("Testing /v1/metadata...")
    status, res = request("GET", "/v1/metadata")
    assert status == 200, f"Expected 200, got {status}"
    assert res["team_name"] == "Team Antigravity", f"Expected Team Antigravity, got {res['team_name']}"
    print("metadata OK:", res)

    # Test context push
    import time
    test_cid = f"test_dentists_{int(time.time())}"
    print(f"Testing context push using dynamic ID '{test_cid}'...")
    cat_payload = {
        "slug": test_cid,
        "voice": {"tone": "peer_clinical"},
        "offer_catalog": []
    }
    status, res = request("POST", "/v1/context", {
        "scope": "category",
        "context_id": test_cid,
        "version": 1,
        "payload": cat_payload,
        "delivered_at": "2026-04-26T10:00:00Z"
    })
    assert status == 200, f"Expected 200, got {status}"
    assert res["accepted"] is True, f"Expected True, got {res['accepted']}"
    print("Context push OK:", res)

    # Test idempotency (version conflict)
    print("Testing version conflict handling...")
    status, res = request("POST", "/v1/context", {
        "scope": "category",
        "context_id": test_cid,
        "version": 1,
        "payload": cat_payload,
        "delivered_at": "2026-04-26T10:00:00Z"
    })
    assert status == 409, f"Expected 409, got {status}"
    assert res["accepted"] is False, f"Expected False, got {res['accepted']}"
    assert res["reason"] == "stale_version", f"Expected stale_version, got {res['reason']}"
    print("Versioning conflict handling OK:", res)

    # Test version bump
    print("Testing version bump...")
    status, res = request("POST", "/v1/context", {
        "scope": "category",
        "context_id": test_cid,
        "version": 2,
        "payload": cat_payload,
        "delivered_at": "2026-04-26T10:00:00Z"
    })
    assert status == 200, f"Expected 200, got {status}"
    assert res["accepted"] is True, f"Expected True, got {res['accepted']}"
    print("Version bump OK:", res)

    # Check healthz counts updated
    status, res = request("GET", "/v1/healthz")
    assert res["contexts_loaded"]["category"] >= 1, f"Expected category contexts >= 1, got {res['contexts_loaded']['category']}"
    print("healthz counts updated OK")

    print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
