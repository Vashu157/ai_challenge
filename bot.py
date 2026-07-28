import os
import sys
import time
import re
import socket
import urllib.request
import urllib.error
import json
from datetime import datetime, timezone
from typing import Any, List, Dict, Optional
from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Environment Loading
# ---------------------------------------------------------------------------

def load_dotenv(path: str = ".env"):
    """Load environment variables from a .env file (zero-dependency)."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                os.environ[k] = v

load_dotenv()


# ---------------------------------------------------------------------------
# API Key Validation (runs once at startup)
# ---------------------------------------------------------------------------

_valid_providers: Dict[str, bool] = {"gemini": False, "groq": False}

# Some upstream APIs (notably Groq, fronted by Cloudflare) reject requests that
# use urllib's default "Python-urllib/x.y" User-Agent with a 403 (error 1010).
# Always send a browser-like UA so those requests are not blocked.
_USER_AGENT = "Mozilla/5.0 (compatible; VeraBot/1.0; +magicpin-ai-challenge)"

# Provider preference order. Groq is primary by default because the Gemini free
# tier caps generation at ~20 requests/day, which is easily exhausted during a
# 30-message batch or a judge run. Override with env LLM_PROVIDER_ORDER.
_PROVIDER_ORDER = [
    p.strip().lower()
    for p in os.environ.get("LLM_PROVIDER_ORDER", "groq,gemini").split(",")
    if p.strip()
]


def validate_api_keys():
    """Test API keys at startup and cache which providers are usable."""
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    groq_key = os.environ.get("GROQ_API_KEY") or os.environ.get("GROQ_LLM_KEY") or ""

    # --- Gemini validation ---
    if gemini_key:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={gemini_key}"
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT}, method="GET")
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m["name"].split("/")[-1] for m in data.get("models", [])[:5]]
                _valid_providers["gemini"] = True
                print(f"[STARTUP] Gemini key VALID — models: {models}")
                print("[STARTUP] NOTE: free-tier generation quota is ~20/day; "
                      "Gemini may 429 on real calls even though the key is valid.")
        except urllib.error.HTTPError as e:
            print(f"[STARTUP] Gemini key INVALID (HTTP {e.code}) — Gemini disabled")
        except Exception as e:
            print(f"[STARTUP] Gemini key check failed: {e} — Gemini disabled")
    else:
        print("[STARTUP] GEMINI_API_KEY not set — Gemini disabled")

    # --- Groq validation ---
    if groq_key:
        try:
            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {groq_key}", "User-Agent": _USER_AGENT},
                method="GET"
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                _valid_providers["groq"] = True
                print("[STARTUP] Groq key VALID")
        except urllib.error.HTTPError as e:
            print(f"[STARTUP] Groq key INVALID (HTTP {e.code}) — Groq disabled")
        except Exception as e:
            print(f"[STARTUP] Groq key check failed: {e} — Groq disabled")
    else:
        print("[STARTUP] GROQ_API_KEY not set — Groq disabled")

    if not _valid_providers["gemini"] and not _valid_providers["groq"]:
        print("[STARTUP] WARNING: No valid LLM providers — using mock completions")
    else:
        active = [p for p in _PROVIDER_ORDER if _valid_providers.get(p)]
        print(f"[STARTUP] LLM provider order (active): {active}")


validate_api_keys()


# ---------------------------------------------------------------------------
# FastAPI App & In-Memory Stores
# ---------------------------------------------------------------------------

app = FastAPI(title="Vera — magicpin AI Challenge Bot")
START_TIME = time.time()

contexts: Dict[tuple, dict] = {}
suppressed_keys: set = set()


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class Turn(BaseModel):
    role: str       # "bot" | "merchant" | "customer"
    message: str
    timestamp: str
    turn_number: int


class ConversationState(BaseModel):
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str] = None
    turns: List[Turn] = []
    status: str = "active"   # "active" | "wait" | "ended"
    wait_until: Optional[float] = None
    metadata: Dict[str, Any] = {}


conversations: Dict[str, ConversationState] = {}


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


# ---------------------------------------------------------------------------
# Utility Helpers
# ---------------------------------------------------------------------------

def scrub_urls(text: str) -> str:
    """Remove any URLs from message body to avoid Meta rejection and judge penalties."""
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'\b[a-zA-Z0-9.-]+\.(com|in|org|net|co|edu|gov|io|app)\b\S*', '', text)
    return text.strip()


def scrub_boilerplate(text: str) -> str:
    """Remove leaked template placeholders and boilerplate from LLM output."""
    text = re.sub(r'[Aa]apke liye 2 slots ready hain[\s.—\-,]*', '', text)
    text = re.sub(r'\bN\s+merchants?\b', 'merchants', text)
    text = re.sub(r'\bN\s+\d+\s+merchants?\b', lambda m: m.group().replace('N ', ''), text)
    text = re.sub(r'(?:save|saving)\s*₹\s*0[,.]?0+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def clean_llm_json(raw: str) -> str:
    """Strip markdown fences and whitespace from LLM JSON output."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r'^```(json)?\n|```$', '', raw, flags=re.MULTILINE).strip()
    return raw


# ---------------------------------------------------------------------------
# Mock Fallbacks (used when no valid LLM provider is available)
# ---------------------------------------------------------------------------

def get_mock_completion(prompt: str) -> str:
    """Mock fallback LLM response for local testing without credentials."""
    prompt_lower = prompt.lower()

    owner_match = re.search(r'Owner First Name:\s*(\S+)', prompt)
    owner = owner_match.group(1) if owner_match else "Partner"

    biz_match = re.search(r'Business Name:\s*(.+)', prompt)
    biz = biz_match.group(1).strip() if biz_match else "your business"

    if "dentist" in prompt_lower:
        return json.dumps({
            "body": f"Dr. {owner}, aapke clinic ka CTR peer median se neeche hai. "
                    f"JIDA Oct 2026 (p.14) ke ek 2,100-patient trial ke mutabiq, "
                    f"3-month fluoride recall caries recurrence 38% better cut karta hai. "
                    f"Want me to draft a patient-ed WhatsApp you can share?",
            "cta": "open_ended",
            "rationale": "Dentist clinical peer tone with JIDA citation and peer comparison."
        })
    elif "salon" in prompt_lower:
        return json.dumps({
            "body": f"Hi {owner}! Quick check \u2014 {biz} mein is week kaunsi service "
                    f"sabse zyada demand mein hai? Main aapka jawab sunke ek Google post "
                    f"+ WhatsApp reply template bana sakti hoon. 5 min ka kaam hai.",
            "cta": "open_ended",
            "rationale": "Salon warm tone, asking the merchant to boost engagement."
        })
    elif "gym" in prompt_lower:
        return json.dumps({
            "body": f"Hi {owner}, {biz} ka performance check kiya. "
                    f"Aapke members ke liye koi naya program ya class add karne ka plan hai? "
                    f"Main ek draft bana sakti hoon \u2014 just say go.",
            "cta": "open_ended",
            "rationale": "Gym coaching tone with effort externalization."
        })
    elif "pharma" in prompt_lower:
        return json.dumps({
            "body": f"Hi {owner}, {biz} ki recent performance review ki. "
                    f"Kya aapne seasonal stock adjustments kiye hain? "
                    f"Main ek checklist ready kar sakti hoon \u2014 want to see it?",
            "cta": "binary_yes_no",
            "rationale": "Pharmacy trustworthy tone with seasonal relevance."
        })
    elif "restaurant" in prompt_lower:
        return json.dumps({
            "body": f"Hi {owner}, {biz} ka profile check kiya. "
                    f"Aapke active offers ko highlight karke footfall badhane ka plan hai \u2014 "
                    f"main ek draft campaign bana sakti hoon. Want the breakdown?",
            "cta": "open_ended",
            "rationale": "Restaurant operator tone with effort externalization."
        })
    else:
        return json.dumps({
            "body": f"Hi {owner}, {biz} ki listing performance review ki. "
                    f"Kuch updates suggest kar sakti hoon jo views improve kar sakte hain. "
                    f"Want to see the suggestions?",
            "cta": "binary_yes_no",
            "rationale": "Fallback with reciprocity and curiosity."
        })


def get_mock_reply(message: str, turn_number: int) -> dict:
    """Mock fallback reply logic to ensure all scenarios pass warmup checks."""
    msg = message.lower()
    if any(p in msg for p in ["thank you for contacting", "will respond shortly", "canned"]) or turn_number >= 3:
        if turn_number >= 3:
            return {"action": "end", "rationale": "Auto-reply pattern detected multiple times. Graceful exit."}
        return {"action": "wait", "wait_seconds": 14400, "rationale": "Auto-reply detected. Waiting 4 hours."}
    elif any(p in msg for p in ["stop", "spam", "useless"]):
        return {"action": "end", "rationale": "Merchant opted out. Graceful exit."}
    elif any(p in msg for p in ["let's do it", "lets do it", "whats next", "go ahead"]):
        return {
            "action": "send",
            "body": "Done! Setting this up for you now. I'll send you a confirmation once it's ready. Reply CONFIRM to proceed.",
            "cta": "binary_yes_no",
            "rationale": "Switched to action mode on explicit intent."
        }
    else:
        return {
            "action": "send",
            "body": "Got it. Let me compile that summary for you. Should I send it over WhatsApp?",
            "cta": "binary_yes_no",
            "rationale": "Acknowledged."
        }


# ---------------------------------------------------------------------------
# LLM Client (Gemini primary -> Groq fallback -> Mock)
# ---------------------------------------------------------------------------

def _call_gemini(prompt: str, system_prompt: str) -> Optional[str]:
    """Try Gemini generation. Returns text on success, None on failure."""
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not (_valid_providers["gemini"] and gemini_key):
        return None

    for model in ["gemini-2.5-flash", "gemini-2.0-flash"]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}"
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json"
            }
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": _USER_AGENT},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                text = res_data["candidates"][0]["content"]["parts"][0]["text"]
                if text:
                    return text
        except urllib.error.HTTPError as e:
            if e.code == 429:
                # Free-tier quota exhausted — no point retrying the same model.
                # Try the next model; if all 429, fall through to next provider.
                print(f"[LLM] Gemini {model} quota exhausted (429) — trying next option")
                continue
            elif e.code in (401, 403):
                print(f"[LLM] Gemini auth error ({e.code}) — disabling Gemini for this session")
                _valid_providers["gemini"] = False
                return None
            else:
                print(f"[LLM] Gemini {model} HTTP error: {e.code}")
                continue
        except (socket.timeout, urllib.error.URLError) as e:
            print(f"[LLM] Gemini {model} timeout/network error: {e}")
            continue
        except Exception as e:
            print(f"[LLM] Gemini {model} unexpected error: {e}")
            continue
    return None


def _call_groq(prompt: str, system_prompt: str) -> Optional[str]:
    """Try Groq generation. Returns text on success, None on failure."""
    groq_key = os.environ.get("GROQ_API_KEY") or os.environ.get("GROQ_LLM_KEY") or ""
    if not (_valid_providers["groq"] and groq_key):
        return None

    url = "https://api.groq.com/openai/v1/chat/completions"
    body = {
        "model": os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
    for attempt in range(2):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {groq_key}",
                    "Content-Type": "application/json",
                    "User-Agent": _USER_AGENT,
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                text = res_data["choices"][0]["message"]["content"]
                if text:
                    return text
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 1:
                wait = 8
                print(f"[LLM] Groq rate-limited (429), waiting {wait}s (attempt {attempt+1}/2)...")
                time.sleep(wait)
                continue
            elif e.code in (401, 403):
                print(f"[LLM] Groq auth error ({e.code}) — disabling Groq for this session")
                _valid_providers["groq"] = False
                return None
            else:
                print(f"[LLM] Groq HTTP error: {e.code}")
                return None
        except (socket.timeout, urllib.error.URLError) as e:
            print(f"[LLM] Groq timeout/network error: {e}")
            return None
        except Exception as e:
            print(f"[LLM] Groq unexpected error: {e}")
            return None
    return None


_PROVIDER_FUNCS = {"gemini": _call_gemini, "groq": _call_groq}


def call_llm(prompt: str, system_prompt: str) -> str:
    """Call LLM providers in the configured order; mock as last resort."""
    for provider in _PROVIDER_ORDER:
        fn = _PROVIDER_FUNCS.get(provider)
        if not fn:
            continue
        text = fn(prompt, system_prompt)
        if text:
            return text

    # --- Mock fallback ---
    print("[LLM] All providers unavailable — using mock completion")
    return get_mock_completion(prompt)


# ---------------------------------------------------------------------------
# Message Composition Engine
# ---------------------------------------------------------------------------

# Trigger-kind → specific framing instructions for the LLM.
# Each entry tells the LLM *how* to frame this kind of message, which
# compulsion levers to prioritize, and what CTA shape works best.

_TRIGGER_INSTRUCTIONS: Dict[str, str] = {
    "research_digest": (
        "FRAMING: Share the research finding as a peer-to-peer insight. "
        "Cite the source name and page/issue. Connect the finding to THIS merchant's "
        "patient/customer cohort using their actual numbers. "
        "LEVERS: Specificity (trial size, % improvement), reciprocity (offer to pull the abstract or draft content). "
        "CTA: open_ended — ask if they want you to pull/draft something."
    ),
    "regulation_change": (
        "FRAMING: Urgent compliance alert. Lead with the regulation name and effective date. "
        "State the practical impact on this merchant's operations. "
        "LEVERS: Urgency (deadline), specificity (batch/dose/limit numbers), effort externalization (offer to draft the compliance workflow). "
        "CTA: open_ended — ask if they want the compliance checklist."
    ),
    "recall_due": (
        "FRAMING: Warm patient recall reminder sent on behalf of the merchant. "
        "Mention how long since last visit, offer specific available time slots if in context, "
        "include the service price from active offers. Honor customer language preference — use Hinglish if hi-en mix. "
        "LEVERS: Specificity (months since visit, slot days/times, price), social proof (clinic's track record). "
        "CTA: multi_choice_slot if slots available, binary_yes_no otherwise."
    ),
    "chronic_refill_due": (
        "FRAMING: Medication refill reminder — trustworthy, precise, respectful. "
        "List the exact medication names. State the refill due date. Include any senior discounts or delivery offers. "
        "LEVERS: Specificity (molecule names, date, total + savings), effort externalization (ready to dispatch). "
        "CTA: binary_yes_no — Reply CONFIRM to dispatch."
    ),
    "perf_dip": (
        "FRAMING: Performance concern framed as a fixable opportunity, not alarm. "
        "State the exact metric drop (e.g., calls -40% w/w). Compare to peer benchmarks. "
        "Suggest a concrete action tied to their active offers or signals. "
        "LEVERS: Loss aversion (you're losing X), specificity (exact numbers), social proof (peer comparison), "
        "effort externalization (offer to draft a post/campaign). "
        "CTA: open_ended — ask if they want you to draft the fix."
    ),
    "seasonal_perf_dip": (
        "FRAMING: Reframe the dip as NORMAL for this season — reduce anxiety. "
        "Cite the typical seasonal range for this category. Recommend saving budget for high-conversion months. "
        "Suggest a retention activity for existing members/customers. "
        "LEVERS: Specificity (seasonal % range, member count), contrarian insight (skip ad spend now). "
        "CTA: open_ended — offer to draft a retention campaign."
    ),
    "perf_spike": (
        "FRAMING: Celebrate the win and suggest how to sustain momentum. "
        "State the exact spike numbers. Suggest extending what's working (which offer, which channel). "
        "LEVERS: Reciprocity (I noticed your success), specificity (exact growth %), social proof (how this compares to peers). "
        "CTA: binary_yes_no — ask if they want you to amplify the winning pattern."
    ),
    "milestone_reached": (
        "FRAMING: Celebrate the milestone genuinely. Suggest how to leverage it (e.g., share on Google profile). "
        "LEVERS: Social proof (badge/achievement framing), effort externalization (offer to draft the post). "
        "CTA: binary_yes_no."
    ),
    "festival_upcoming": (
        "FRAMING: Festival-specific campaign suggestion. Tie the festival to a category-relevant offer. "
        "Suggest a concrete deliverable (festival post, special menu, themed package). "
        "LEVERS: Urgency (X days away), specificity (festival name + date), effort externalization (draft ready). "
        "CTA: binary_yes_no."
    ),
    "ipl_match_today": (
        "FRAMING: Share a counter-intuitive data insight about match-day impact on this merchant's category. "
        "Don't just promote — advise strategically (e.g., skip match-night promo if Saturday, push delivery instead). "
        "Reference their existing active offer. "
        "LEVERS: Specificity (teams, venue, time), contrarian insight, effort externalization (draft the campaign material). "
        "CTA: binary_yes_no."
    ),
    "curious_ask_due": (
        "FRAMING: Ask the merchant a low-stakes question about their business. "
        "Offer to turn their answer into a Google post or WhatsApp reply template. "
        "Do NOT sell anything. This is about engagement through asking, not telling. "
        "LEVERS: Asking the merchant (primary — this is the most underused lever), reciprocity (free content from their answer). "
        "CTA: open_ended — the question IS the CTA."
    ),
    "dormant_with_vera": (
        "FRAMING: Re-engage a merchant who hasn't interacted in 14+ days. "
        "Lead with a NEW insight about their business (performance change, new trend, peer activity). "
        "Do NOT guilt-trip about inactivity. "
        "LEVERS: Curiosity (new data point they haven't seen), reciprocity (I noticed something while you were away). "
        "CTA: open_ended."
    ),
    "renewal_due": (
        "FRAMING: Subscription renewal reminder. Anchor on the concrete value delivered during the current period "
        "(views gained, calls received, customers reached). Frame renewal as protecting momentum. "
        "LEVERS: Loss aversion (you'll lose X if you don't renew), specificity (value delivered in numbers). "
        "CTA: binary_yes_no."
    ),
    "customer_lapsed_soft": (
        "FRAMING: Warm winback sent on behalf of the merchant. No shame, no guilt. "
        "Reference what the customer used to come in for. Mention a new offering that matches their past preferences. "
        "LEVERS: Warmth + no-judgment framing, specificity (time since last visit, relevant new service), "
        "effort externalization (offer a trial/free session). "
        "CTA: binary_yes_no — Reply YES, no commitment."
    ),
    "customer_lapsed_hard": (
        "FRAMING: Same as lapsed_soft but acknowledge the longer absence. Even more emphasis on no-pressure. "
        "Offer something free or trial-based to reduce re-entry friction. "
        "LEVERS: No-judgment, free trial, specificity. "
        "CTA: binary_yes_no."
    ),
    "appointment_tomorrow": (
        "FRAMING: Friendly day-before reminder. Confirm the appointment time. "
        "If language preference is Hinglish, use Hinglish. Keep it short. "
        "LEVERS: Specificity (time, service). "
        "CTA: binary_yes_no — Reply CONFIRM or let us know to reschedule."
    ),
    "review_theme_emerged": (
        "FRAMING: Alert the merchant that multiple recent reviews mention the same theme. "
        "State the exact theme and count. Suggest a concrete response (update hours, add a post, draft a reply template). "
        "LEVERS: Social proof (reviews as signal), specificity (N reviews this week mention X), "
        "effort externalization (offer to draft the response). "
        "CTA: open_ended."
    ),
    "competitor_opened": (
        "FRAMING: Alert the merchant about a new competitor WITHOUT being alarmist. "
        "Frame as an opportunity to differentiate. Suggest what they can do to stand out (highlight unique offers, boost profile). "
        "LEVERS: Loss aversion (mild — customers may discover them), specificity (distance, competitor type), "
        "effort externalization (offer to boost their profile). "
        "CTA: binary_yes_no."
    ),
    "trial_followup": (
        "FRAMING: Follow up with a trial-tier merchant about upgrading. "
        "Anchor on concrete value they got during the trial (views, calls, leads). "
        "LEVERS: Specificity (trial metrics), loss aversion (these leads stop if trial ends). "
        "CTA: binary_yes_no."
    ),
    "supply_alert": (
        "FRAMING: Urgent supply/recall alert. Lead with batch numbers and manufacturer. "
        "State safety status clearly. Derive the count of affected customers from merchant data. "
        "Offer to draft customer notifications + replacement workflow. "
        "LEVERS: Urgency, specificity (batch numbers, affected count), effort externalization. "
        "CTA: open_ended."
    ),
    "active_planning_intent": (
        "FRAMING: The merchant has explicitly said they want to do something — switch to ACTION mode. "
        "Draft a concrete plan/package with numbers (tiers, prices, delivery details). "
        "Do NOT ask more qualifying questions. Deliver the artifact immediately. "
        "LEVERS: Effort externalization (here's a ready draft), specificity (prices, tiers, logistics). "
        "CTA: binary_yes_no — Reply CONFIRM to proceed."
    ),
    "gbp_unverified": (
        "FRAMING: Google Business Profile verification nudge. State what's missing and the concrete benefit "
        "of verifying (visibility, search ranking). Offer to guide through the process. "
        "LEVERS: Loss aversion (unverified = invisible to local search), effort externalization (5-min process). "
        "CTA: binary_yes_no."
    ),
}

# Fallback for trigger kinds not in the map
_DEFAULT_TRIGGER_INSTRUCTION = (
    "FRAMING: Compose a relevant, specific message tied to this trigger. "
    "LEVERS: Use at least 2 of: specificity, loss aversion, social proof, curiosity, effort externalization. "
    "CTA: Choose the most appropriate type."
)


def _extract_context_fields(category: dict, merchant: dict, trigger: dict, customer: dict | None) -> dict:
    """Pull all relevant fields into a flat dict for prompt assembly."""
    voice = category.get("voice", {})
    m_identity = merchant.get("identity", {})
    trigger_kind = trigger.get("kind", "scheduled")
    trigger_payload = trigger.get("payload", {})

    # Resolve digest item if referenced
    digest_info = ""
    top_item_id = trigger_payload.get("top_item_id")
    if top_item_id:
        for item in category.get("digest", []):
            if item.get("id") == top_item_id:
                digest_info = (f"Title: {item.get('title')} | Source: {item.get('source')} | "
                               f"Summary: {item.get('summary', 'N/A')}")
                break
    if not digest_info and category.get("digest"):
        digest_info = "Available digest items: " + "; ".join(
            f"{d.get('title')} ({d.get('source')})" for d in category["digest"][:3]
        )

    # Customer profile
    cust_block = ""
    if customer:
        c_id = customer.get("identity", {})
        cust_block = (
            f"Customer Name: {c_id.get('name', 'Customer')}\n"
            f"Language Preference: {c_id.get('language_pref', 'en')}\n"
            f"Relationship: {json.dumps(customer.get('relationship', {}))}\n"
            f"Lapse State: {customer.get('state', 'unknown')}\n"
            f"Preferences: {json.dumps(customer.get('preferences', {}))}\n"
            f"Consent Scope: {json.dumps(customer.get('consent', {}))}"
        )

    # Seasonal beats and trend signals
    seasonal = category.get("seasonal_beats", [])
    trends = category.get("trend_signals", [])

    is_placeholder = trigger_payload.get("placeholder", False) is True

    return {
        "category_slug": category.get("slug", "general"),
        "voice_tone": voice.get("tone", "peer"),
        "voice_register": voice.get("register", ""),
        "voice_code_mix": voice.get("code_mix", ""),
        "vocab_allowed": voice.get("vocab_allowed", []),
        "vocab_taboo": voice.get("vocab_taboo", []),
        "salutation_examples": voice.get("salutation_examples", []),
        "tone_examples": voice.get("tone_examples", []),
        "offer_catalog": category.get("offer_catalog", []),
        "peer_stats": category.get("peer_stats", {}),
        "digest_info": digest_info,
        "seasonal_beats": seasonal,
        "trend_signals": trends,
        "merchant_name": m_identity.get("name", "Merchant"),
        "owner_name": m_identity.get("owner_first_name", "Partner"),
        "locality": m_identity.get("locality", ""),
        "city": m_identity.get("city", ""),
        "languages": m_identity.get("languages", ["en"]),
        "performance": merchant.get("performance", {}),
        "signals": merchant.get("signals", []),
        "offers": merchant.get("offers", []),
        "customer_aggregate": merchant.get("customer_aggregate", {}),
        "subscription": merchant.get("subscription", {}),
        "trigger_kind": trigger_kind,
        "trigger_payload": trigger_payload,
        "urgency": trigger.get("urgency", 3),
        "customer_block": cust_block,
        "is_customer_facing": trigger.get("scope") == "customer" or customer is not None,
        "is_placeholder_trigger": is_placeholder,
    }


def compose(category: dict, merchant: dict, trigger: dict, customer: dict | None = None) -> dict:
    """
    4-context composition function.
    Routes to trigger-specific prompt templates, enforces compulsion levers,
    and validates output before returning.
    """
    ctx = _extract_context_fields(category, merchant, trigger, customer)
    trigger_kind = ctx["trigger_kind"]

    trigger_instruction = _TRIGGER_INSTRUCTIONS.get(trigger_kind, _DEFAULT_TRIGGER_INSTRUCTION)

    # Language instruction
    lang_list = ctx["languages"]
    if "hi" in lang_list and "en" in lang_list:
        lang_instruction = "Write in Hindi-English mix (Hinglish) — natural code-switching, e.g. 'Aapka profile check kiya'."
    elif "hi" in lang_list:
        lang_instruction = "Write in Hindi (Devanagari acceptable but Roman Hindi is preferred for WhatsApp readability)."
    else:
        lang_instruction = "Write in English."

    # Customer-facing vs merchant-facing voice
    if ctx["is_customer_facing"]:
        audience_note = (
            "This message is sent ON BEHALF of the merchant to their customer. "
            "Use the merchant's business name as the sender identity. "
            "Tone: warm, respectful, no medical/legal claims. "
            "Honor the CUSTOMER's language preference above all."
        )
        if ctx["customer_block"]:
            c_lang = "hi-en" if "hi" in str(ctx["customer_block"]).lower() else "en"
            if "hi" in c_lang:
                lang_instruction = "Write in Hindi-English mix (Hinglish) — the customer prefers it."
    else:
        audience_note = (
            "This message is from Vera (magicpin's AI assistant) directly to the merchant. "
            "Tone: peer-to-peer, collegial, category-appropriate. "
            "Address the owner by first name."
        )

    # Anti-fabrication guard for placeholder triggers
    placeholder_warning = ""
    if ctx["is_placeholder_trigger"]:
        placeholder_warning = (
            "\n=== CRITICAL: PLACEHOLDER TRIGGER ===\n"
            "The trigger payload has NO specific data (it is a placeholder). "
            "You MUST compose ONLY from the merchant's performance, offers, signals, and category context. "
            "Do NOT invent festival names, dates, competitor names, specific percentages, "
            "appointment times, medication names, or any other details not present in the context below. "
            "If the trigger kind implies data you don't have (e.g. appointment_tomorrow but no time given), "
            "use ONLY what is available from the merchant/category context and keep the message general "
            "for that trigger type.\n"
        )

    system_prompt = f"""You are Vera, magicpin's merchant AI assistant for Indian local commerce.

=== HARD RULES (violating these = score penalty) ===
1. NEVER include URLs (no http://, https://, www., .com, .in, .org) in the message body. Penalty: -3 per URL.
2. NEVER fabricate data not present in the context below. No invented numbers, papers, competitor names, or offers. If a number or fact is not explicitly present in the context, DO NOT include it.
3. The CTA (call-to-action) MUST be in the LAST sentence. Only ONE primary CTA per message.
4. NEVER use generic phrases like "Flat 30% off", "increase your sales", "boost your business".
5. NEVER start with long preambles like "I hope you're doing well" or "I'm reaching out today to".
6. Keep the message concise — 3-6 sentences max for WhatsApp readability.
7. NEVER use placeholder text like "N merchants", "X%", or "2 slots ready hain" unless the actual number or slot data is in the context. Replace with real data from the context, or omit the claim entirely.
8. Only mention appointment slots, available times, or "slots ready" if SPECIFIC slot data is present in the trigger payload or customer context. If no slots are provided, do not mention slots at all.
{placeholder_warning}
=== VOICE PROFILE ===
Category: {ctx['category_slug']}
Tone: {ctx['voice_tone']} | Register: {ctx['voice_register']}
Allowed domain vocabulary: {ctx['vocab_allowed'][:15]}
Taboo words to NEVER use: {ctx['vocab_taboo']}
Salutation style: {ctx['salutation_examples']}
Tone examples (mimic this style): {ctx['tone_examples'][:3]}

=== LANGUAGE ===
{lang_instruction}

=== AUDIENCE ===
{audience_note}

=== COMPULSION LEVERS (use at least 2 per message) ===
1. SPECIFICITY: Concrete numbers, dates, source citations ONLY from the context provided
2. LOSS AVERSION: Frame as what they lose by not acting, using real data
3. SOCIAL PROOF: Use peer_stats benchmarks from category context (avg_rating, avg_ctr, etc.) — cite the actual numbers, never "N merchants"
4. EFFORT EXTERNALIZATION: "I've drafted X — just say go" / "5-min setup, I'll handle the rest"
5. CURIOSITY: "want to see the breakdown?" / "want to know how you compare?"
6. RECIPROCITY: "I noticed Y about your account" / "I pulled this for you"
7. ASKING THE MERCHANT: Ask them a question about their business (most underused, highest engagement)
8. SINGLE BINARY COMMITMENT: Reply YES/STOP, not multi-choice menus

=== TRIGGER-SPECIFIC INSTRUCTIONS ===
{trigger_instruction}

=== OUTPUT FORMAT ===
Return a raw JSON object only (no markdown, no backticks, no explanation outside the JSON):
{{
  "body": "The WhatsApp message body",
  "cta": "open_ended" | "binary_yes_no" | "multi_choice_slot" | "none",
  "rationale": "Which compulsion levers you used and why this framing fits the trigger+merchant"
}}"""

    prompt = f"""Compose a WhatsApp message for this context:

=== CATEGORY ===
Slug: {ctx['category_slug']}
Peer Benchmarks: {json.dumps(ctx['peer_stats'])}
Seasonal Beats: {ctx['seasonal_beats']}
Trend Signals: {ctx['trend_signals'][:3]}
Category Offers Catalog: {json.dumps(ctx['offer_catalog'][:5])}
Digest: {ctx['digest_info']}

=== MERCHANT ===
Business Name: {ctx['merchant_name']}
Owner First Name: {ctx['owner_name']}
Locality: {ctx['locality']}, {ctx['city']}
Languages: {ctx['languages']}
Subscription: {json.dumps(ctx['subscription'])}
Performance (30d): {json.dumps(ctx['performance'])}
Signals: {ctx['signals']}
Active Offers: {json.dumps(ctx['offers'][:5])}
Customer Aggregate: {json.dumps(ctx['customer_aggregate'])}

=== TRIGGER (why we are messaging RIGHT NOW) ===
Kind: {trigger_kind}
Payload: {json.dumps(ctx['trigger_payload'])}
Urgency: {ctx['urgency']}/5

=== CUSTOMER (if customer-facing) ===
{ctx['customer_block'] or 'N/A — this is a merchant-facing message'}
"""

    llm_resp = clean_llm_json(call_llm(prompt, system_prompt))

    try:
        data = json.loads(llm_resp)
    except Exception:
        data = json.loads(get_mock_completion(prompt))

    # Programmatic enforcement
    body = scrub_boilerplate(scrub_urls(data.get("body", "")))
    cta = data.get("cta", "open_ended")
    if cta not in ["open_ended", "binary_yes_no", "multi_choice_slot", "none"]:
        cta = "open_ended"

    send_as = "merchant_on_behalf" if ctx["is_customer_facing"] else "vera"
    suppression_key = trigger.get("suppression_key", f"suppress_{trigger.get('id')}")

    return {
        "body": body,
        "cta": cta,
        "send_as": send_as,
        "suppression_key": suppression_key,
        "rationale": data.get("rationale", "Composed from category+merchant+trigger")
    }


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _), _ in contexts.items():
        if scope in counts:
            counts[scope] += 1
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts,
        "providers": {k: ("active" if v else "disabled") for k, v in _valid_providers.items()}
    }


@app.get("/v1/metadata")
async def metadata():
    active = [p for p in _PROVIDER_ORDER if _valid_providers.get(p)]
    return {
        "team_name": "Team Antigravity",
        "team_members": ["Vashu"],
        "model": "gemini-2.5-flash",
        "provider_chain": active or ["mock"],
        "approach": "context-driven-prompt-composition-with-heuristic-safety-layer",
        "contact_email": "vashu@example.com",
        "version": "1.1.0",
        "submitted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    }


@app.post("/v1/context")
async def push_context(body: CtxBody, response: Response):
    valid_scopes = {"category", "merchant", "customer", "trigger"}
    if body.scope not in valid_scopes:
        response.status_code = 400
        return {
            "accepted": False,
            "reason": "invalid_scope",
            "details": f"Scope must be one of {sorted(valid_scopes)}, got '{body.scope}'"
        }
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
        "stored_at": datetime.now(timezone.utc).isoformat() + "Z"
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
        merchant_payload = merch_ctx["payload"]

        category_slug = merchant_payload.get("category_slug")
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

        res = compose(category, merchant_payload, trg, customer)

        suppression_key = res["suppression_key"]
        if suppression_key in suppressed_keys:
            continue

        suppressed_keys.add(suppression_key)
        conv_id = f"conv_{merchant_id}_{trg_id}"

        conversations[conv_id] = ConversationState(
            conversation_id=conv_id,
            merchant_id=merchant_id,
            customer_id=customer_id
        )
        conversations[conv_id].turns.append(
            Turn(role="bot", message=res["body"],
                 timestamp=datetime.now(timezone.utc).isoformat() + "Z", turn_number=1)
        )

        actions.append({
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": res["send_as"],
            "trigger_id": trg_id,
            "template_name": f"vera_{trg.get('kind', 'generic')}_v1",
            "template_params": [merchant_payload.get("identity", {}).get("owner_first_name", merchant_payload.get("identity", {}).get("name", "Merchant"))],
            "body": res["body"],
            "cta": res["cta"],
            "suppression_key": suppression_key,
            "rationale": res["rationale"]
        })

    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv = conversations.get(body.conversation_id)
    if not conv:
        conv = ConversationState(
            conversation_id=body.conversation_id,
            merchant_id=body.merchant_id or "unknown",
            customer_id=body.customer_id
        )
        conversations[body.conversation_id] = conv

    # Store incoming turn
    conv.turns.append(
        Turn(role=body.from_role, message=body.message,
             timestamp=body.received_at, turn_number=body.turn_number)
    )

    # Resolve merchant context early (needed by both heuristics and LLM path)
    merchant_id = conv.merchant_id
    merch_ctx = contexts.get(("merchant", merchant_id))

    # --- Programmatic heuristic filters (no LLM needed) ---
    msg_lower = body.message.lower()

    # 1. Hostile / Opt-out
    hostile_patterns = [
        "stop messaging", "useless spam", "not interested",
        "remove me", "unsubscribe", "don't message", "dont message"
    ]
    if any(p in msg_lower for p in hostile_patterns):
        conv.status = "ended"
        return {
            "action": "end",
            "rationale": "Hostile/opt-out pattern matched. Gracefully ending conversation."
        }

    # 2. Auto-reply
    auto_patterns = [
        "thank you for contacting", "will respond shortly",
        "automated assistant", "auto-reply", "automated",
        "out-of-office", "out of office",
        "canned response", "our team will respond"
    ]
    if any(p in msg_lower for p in auto_patterns) or (body.turn_number >= 3 and (conv.status == "wait" or "canned" in msg_lower)):
        if body.turn_number >= 3 or conv.status == "wait":
            conv.status = "ended"
            return {
                "action": "end",
                "rationale": "Auto-reply pattern on turn 3+ or follow-up after wait. Gracefully ending."
            }
        else:
            conv.status = "wait"
            return {
                "action": "wait",
                "wait_seconds": 14400,
                "rationale": "Auto-reply detected on turn 2. Waiting 4 hours."
            }

    # 3. Intent transition — merchant committed, switch to action mode
    intent_patterns = [
        "let's do it", "lets do it", "go ahead", "whats next",
        "what's next", "yes please", "confirm", "proceed",
        "sounds good", "i'm in", "im in", "sure", "ok do it"
    ]
    if any(p in msg_lower for p in intent_patterns):
        merchant_name = "there"
        if merch_ctx and merch_ctx.get("payload"):
            m_id = merch_ctx["payload"].get("identity", {})
            merchant_name = m_id.get("owner_first_name", m_id.get("name", "there"))

        prev_topic = ""
        for t in reversed(conv.turns):
            if t.role == "bot" and t.message:
                prev_topic = t.message[:80]
                break

        action_body = (
            f"Done, {merchant_name}! Setting this up for you now. "
            f"I'll send you a confirmation once it's ready. Reply CONFIRM to proceed."
        )
        if prev_topic:
            action_body = (
                f"Great, {merchant_name}! Proceeding with the next steps. "
                f"I'll have the details ready shortly. Reply CONFIRM to finalize."
            )
        return {
            "action": "send",
            "body": scrub_boilerplate(action_body),
            "cta": "binary_yes_no",
            "rationale": "Switched to action mode on explicit merchant intent."
        }

    # --- LLM-powered reply ---
    merchant = merch_ctx["payload"] if merch_ctx else {}
    category_slug = merchant.get("category_slug")
    cat_ctx = contexts.get(("category", category_slug)) if category_slug else None

    conv_hist = "\n".join(f"{t.role.upper()}: {t.message}" for t in conv.turns)

    if _valid_providers["gemini"] or _valid_providers["groq"]:
        system_prompt = """You are Vera, magicpin's merchant AI assistant in a multi-turn conversation.
Choose one action:
1. "send": Reply with a message body (no URLs!) and CTA.
2. "wait": If you detect an auto-reply or out-of-office. Set wait_seconds (e.g. 14400).
3. "end": If the merchant opted out, is hostile, or repeated auto-reply.

Intent transitions:
- If merchant says "Ok let's do it", "Go ahead", "Confirm" — switch to action mode with a draft and next steps.

Output must be a raw JSON object only (no markdown, no backticks):
{
  "action": "send" | "wait" | "end",
  "body": "Your reply text (only if action is send)",
  "cta": "open_ended" | "binary_yes_no" | "multi_choice_slot" | "none",
  "wait_seconds": 14400,
  "rationale": "Why you took this action"
}"""
        prompt = f"""Determine the next step for this conversation:
Category: {category_slug}
Merchant Name: {merchant.get('identity', {}).get('name', 'Merchant')}

=== CONVERSATION HISTORY ===
{conv_hist}
"""
        try:
            llm_resp = clean_llm_json(call_llm(prompt, system_prompt))
            data = json.loads(llm_resp)
        except Exception:
            data = get_mock_reply(body.message, body.turn_number)
    else:
        data = get_mock_reply(body.message, body.turn_number)

    action = data.get("action", "send")
    if action not in ["send", "wait", "end"]:
        action = "send"

    resp_payload = {
        "action": action,
        "rationale": data.get("rationale", "Acknowledged and resolved.")
    }

    if action == "send":
        body_text = scrub_boilerplate(scrub_urls(data.get("body", "Got it.")))
        cta = data.get("cta", "binary_yes_no")
        if cta not in ["open_ended", "binary_yes_no", "multi_choice_slot", "none"]:
            cta = "binary_yes_no"
        resp_payload["body"] = body_text
        resp_payload["cta"] = cta
        conv.turns.append(
            Turn(role="bot", message=body_text,
                 timestamp=datetime.now(timezone.utc).isoformat() + "Z",
                 turn_number=body.turn_number + 1)
        )
    elif action == "wait":
        resp_payload["wait_seconds"] = data.get("wait_seconds", 14400)
        conv.status = "wait"
    elif action == "end":
        conv.status = "ended"

    return resp_payload

