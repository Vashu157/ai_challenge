import urllib.request
import json
import sys
import time

# Default target URL (can pass custom URL via command line: python test_server.py https://vashu-magicpin-vera-bot.onrender.com)
BASE_URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8080"

def request(method, path, body=None):
    url = f"{BASE_URL}{path}"
    data = json.dumps(body).encode("utf-8") if body else None
    headers = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except:
            return e.code, None
    except Exception as e:
        print(f"Error connecting to server at {url}: {e}")
        sys.exit(1)

def run_tests():
    print(f"============================================================")
    print(f"  TESTING ALL 5 REQUIRED ENDPOINTS ON: {BASE_URL}")
    print(f"============================================================\n")

    # -------------------------------------------------------------------
    # 1. Test GET /v1/healthz
    # -------------------------------------------------------------------
    print("1. Testing GET /v1/healthz...")
    status, res = request("GET", "/v1/healthz")
    assert status == 200, f"Expected 200, got {status}"
    assert res.get("status") == "ok", f"Expected status 'ok', got {res.get('status')}"
    assert "contexts_loaded" in res, "Missing 'contexts_loaded' in healthz response"
    print(f"   [PASS] /v1/healthz OK: {res}\n")

    # -------------------------------------------------------------------
    # 2. Test GET /v1/metadata
    # -------------------------------------------------------------------
    print("2. Testing GET /v1/metadata...")
    status, res = request("GET", "/v1/metadata")
    assert status == 200, f"Expected 200, got {status}"
    assert "team_name" in res, "Missing 'team_name' in metadata"
    assert "model" in res, "Missing 'model' in metadata"
    assert "approach" in res, "Missing 'approach' in metadata"
    print(f"   [PASS] /v1/metadata OK: Team='{res.get('team_name')}', Model='{res.get('model')}'\n")

    # -------------------------------------------------------------------
    # 3. Test POST /v1/context (Push & 409 Version Conflict)
    # -------------------------------------------------------------------
    ts = int(time.time())
    cat_id = f"test_cat_{ts}"
    merch_id = f"m_test_{ts}"
    trg_id = f"trg_test_{ts}"

    print(f"3. Testing POST /v1/context (Ingestion & Versioning)...")

    # Push Category Context (v1)
    cat_payload = {
        "slug": cat_id,
        "voice": {"tone": "peer_clinical", "vocab_taboo": ["cheap", "discount"]},
        "peer_stats": {"avg_recall_rate": 0.42}
    }
    status, res = request("POST", "/v1/context", {
        "scope": "category", "context_id": cat_id, "version": 1,
        "payload": cat_payload, "delivered_at": "2026-04-26T10:00:00Z"
    })
    assert status == 200, f"Expected 200, got {status}"
    assert res.get("accepted") is True, f"Expected accepted=True, got {res}"
    print(f"   [PASS] Category context pushed successfully (v1)")

    # Push Merchant Context (v1)
    merch_payload = {
        "category_slug": cat_id,
        "identity": {"name": "Test Dental Clinic", "owner_first_name": "Meera", "locality": "Delhi", "languages": ["en"]},
        "performance": {"views": 1000, "calls": 30, "ctr": 0.03}
    }
    status, res = request("POST", "/v1/context", {
        "scope": "merchant", "context_id": merch_id, "version": 1,
        "payload": merch_payload, "delivered_at": "2026-04-26T10:00:00Z"
    })
    assert status == 200, f"Expected 200, got {status}"
    assert res.get("accepted") is True, f"Expected accepted=True, got {res}"
    print(f"   [PASS] Merchant context pushed successfully (v1)")

    # Push Trigger Context (v1)
    trg_payload = {
        "id": trg_id,
        "merchant_id": merch_id,
        "kind": "recall_due",
        "urgency": 4,
        "payload": {"overdue_count": 15}
    }
    status, res = request("POST", "/v1/context", {
        "scope": "trigger", "context_id": trg_id, "version": 1,
        "payload": trg_payload, "delivered_at": "2026-04-26T10:00:00Z"
    })
    assert status == 200, f"Expected 200, got {status}"
    assert res.get("accepted") is True, f"Expected accepted=True, got {res}"
    print(f"   [PASS] Trigger context pushed successfully (v1)")

    # Test Version Conflict (Push duplicate v1 -> Expect 409 Conflict)
    print("   Testing version conflict detection (Duplicate v1)...")
    status, res = request("POST", "/v1/context", {
        "scope": "category", "context_id": cat_id, "version": 1,
        "payload": cat_payload, "delivered_at": "2026-04-26T10:00:00Z"
    })
    assert status == 409, f"Expected 409 Conflict, got {status}"
    assert res.get("accepted") is False, f"Expected accepted=False, got {res}"
    assert res.get("reason") == "stale_version", f"Expected stale_version, got {res.get('reason')}"
    print(f"   [PASS] Version conflict 409 handling OK\n")

    # -------------------------------------------------------------------
    # 4. Test POST /v1/tick (Batch Message Composition)
    # -------------------------------------------------------------------
    print(f"4. Testing POST /v1/tick (AI Message Composition)...")
    status, res = request("POST", "/v1/tick", {
        "now": "2026-04-26T10:05:00Z",
        "available_triggers": [trg_id]
    })
    assert status == 200, f"Expected 200, got {status}"
    assert "actions" in res, "Missing 'actions' in /v1/tick response"
    assert len(res["actions"]) > 0, "Expected at least 1 action from /v1/tick"
    
    action = res["actions"][0]
    assert "body" in action, "Missing 'body' in action"
    assert "cta" in action, "Missing 'cta' in action"
    assert "send_as" in action, "Missing 'send_as' in action"
    assert "suppression_key" in action, "Missing 'suppression_key' in action"
    
    # Assert URL scrubbing rule (No URLs allowed in body)
    body = action["body"]
    assert "http://" not in body and "https://" not in body and "www." not in body, "Rule Violation: URL found in body!"
    
    print(f"   [PASS] /v1/tick OK!")
    print(f"          Composed Body: \"{body}\"")
    print(f"          CTA: {action['cta']} | Send As: {action['send_as']}\n")

    # -------------------------------------------------------------------
    # 5. Test POST /v1/reply (Multi-Turn Chat Reasoning & Heuristics)
    # -------------------------------------------------------------------
    print(f"5. Testing POST /v1/reply (Multi-Turn AI Reasoning & Safety Rules)...")
    conv_id = f"conv_{merch_id}_{trg_id}"

    # Test 5A: Standard reply / Intent transition ("let's do it")
    print("   5A. Testing Intent Transition ('let's do it')...")
    status, res = request("POST", "/v1/reply", {
        "conversation_id": conv_id,
        "merchant_id": merch_id,
        "from_role": "merchant",
        "message": "Ok let's do it. What's next?",
        "received_at": "2026-04-26T10:10:00Z",
        "turn_number": 2
    })
    assert status == 200, f"Expected 200, got {status}"
    assert res.get("action") == "send", f"Expected action 'send', got {res.get('action')}"
    assert "body" in res, "Missing 'body' in reply action"
    print(f"       [PASS] Intent transition OK -> Action: {res['action']} | Reply: \"{res['body'][:80]}...\"")

    # Test 5B: Auto-Reply Detection ("Thank you for contacting us")
    print("   5B. Testing Auto-Reply Filter ('Thank you for contacting us')...")
    status, res = request("POST", "/v1/reply", {
        "conversation_id": f"conv_auto_{ts}",
        "merchant_id": merch_id,
        "from_role": "merchant",
        "message": "Thank you for contacting us! Our team will respond shortly.",
        "received_at": "2026-04-26T10:15:00Z",
        "turn_number": 2
    })
    assert status == 200, f"Expected 200, got {status}"
    assert res.get("action") == "wait", f"Expected action 'wait', got {res.get('action')}"
    assert res.get("wait_seconds") == 14400, f"Expected wait_seconds=14400, got {res.get('wait_seconds')}"
    print(f"       [PASS] Auto-reply filter OK -> Action: {res['action']} (Waiting {res['wait_seconds']}s)")

    # Test 5C: Hostile Opt-Out ("Stop messaging me")
    print("   5C. Testing Hostile Opt-Out Filter ('Stop messaging me')...")
    status, res = request("POST", "/v1/reply", {
        "conversation_id": f"conv_hostile_{ts}",
        "merchant_id": merch_id,
        "from_role": "merchant",
        "message": "Stop messaging me. This is useless spam.",
        "received_at": "2026-04-26T10:20:00Z",
        "turn_number": 2
    })
    assert status == 200, f"Expected 200, got {status}"
    assert res.get("action") == "end", f"Expected action 'end', got {res.get('action')}"
    print(f"       [PASS] Hostile opt-out filter OK -> Action: {res['action']}\n")

    print(f"============================================================")
    print(f"  ALL 5 REQUIRED API ENDPOINTS VERIFIED AND PASSED!")
    print(f"============================================================")

if __name__ == "__main__":
    run_tests()
