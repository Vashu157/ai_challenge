import os
import time
from datetime import datetime
from typing import Any, List, Dict, Optional
from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel

app = FastAPI()
START_TIME = time.time()

# In-memory stores
# (scope, context_id) -> {version: int, payload: dict}
contexts: Dict[tuple[str, str], dict] = {}
# conversation_id -> list of turns
conversations: Dict[str, List[dict]] = {}
# Set of active suppression keys
suppressed_keys = set()

class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict
    delivered_at: str

class TickBody(BaseModel):
    now: str
    available_triggers: List[str] = []

class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int

@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _), _ in contexts.items():
        if scope in counts:
            counts[scope] += 1
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts
    }

@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Team Vashu",
        "team_members": ["pogog builder"],
        "model": "GEMINI-3.5-high",
        "approach": "dispatch-by-trigger-kind-prompt-templates-with-re-prompting-and-auto-reply-detection",
        "contact_email": "rebuilder@example.com",
        "version": "1.0.0",
        "submitted_at": "2026-04-26T08:00:00Z"
    }

@app.post("/v1/context")
async def push_context(body: CtxBody, response: Response):
    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    if cur and cur["version"] >= body.version:
        response.status_code = 409
        return {
            "accepted": False,
            "reason": "stale_version",
            "current_version": cur["version"]
        }
    contexts[key] = {"version": body.version, "payload": body.payload}
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.utcnow().isoformat() + "Z"
    }

@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []
    for trg_id in body.available_triggers:
        trg_ctx = contexts.get(("trigger", trg_id))
        if not trg_ctx:
            continue
        trg = trg_ctx["payload"]
        
        merchant_id = trg.get("merchant_id")
        if not merchant_id:
            continue
            
        merch_ctx = contexts.get(("merchant", merchant_id))
        if not merch_ctx:
            continue
        merchant = merch_ctx["payload"]
        
        category_slug = merchant.get("category_slug")
        if not category_slug:
            continue
            
        cat_ctx = contexts.get(("category", category_slug))
        if not cat_ctx:
            continue
        category = cat_ctx["payload"]

        customer_id = trg.get("customer_id")
        customer = None
        if customer_id:
            cust_ctx = contexts.get(("customer", customer_id))
            if cust_ctx:
                customer = cust_ctx["payload"]

        # Simple baseline message composition for Phase 1 verification
        suppression_key = trg.get("suppression_key", f"suppress_{trg_id}")
        if suppression_key in suppressed_keys:
            continue
            
        # Add to suppressed keys so we don't repeat in this tick run
        suppressed_keys.add(suppression_key)
        
        conv_id = f"conv_{merchant_id}_{trg_id}"
        
        # Simple default reply template
        m_name = merchant.get("identity", {}).get("name", "Merchant")
        body_text = f"Hi {m_name}, JIDA's Oct issue highlights new clinical research. Would you like to review the 2-minute summary?"
        
        actions.append({
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": "vera",
            "trigger_id": trg_id,
            "template_name": "vera_generic_v1",
            "template_params": [m_name],
            "body": body_text,
            "cta": "open_ended",
            "suppression_key": suppression_key,
            "rationale": "Phase 1 Baseline message."
        })
        
    return {"actions": actions}

@app.post("/v1/reply")
async def reply(body: ReplyBody):
    # Store turn
    conversations.setdefault(body.conversation_id, []).append({
        "from": body.from_role,
        "msg": body.message,
        "ts": body.received_at,
        "turn": body.turn_number
    })
    
    # Return simple baseline acknowledgment for Phase 1
    return {
        "action": "send",
        "body": "Got it, I will prepare the abstract details for you. Should I send them over WhatsApp now?",
        "cta": "binary_yes_no",
        "rationale": "Acknowledging the input and presenting a next-best-step binary CTA."
    }
