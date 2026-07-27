# magicpin AI Challenge — Vera Rebuilder (Team Antigravity)

## 1. Approach Overview

Our bot implementation follows a robust, hybrid architecture that bridges programmatic heuristics for strict rule compliance with LLM execution for fluid business reasoning:

1. **Lightweight FastAPI Backend**: Implemented with zero-dependency `load_dotenv` for environment parsing, and standard HTTP requests for LLM calls, ensuring fast startup and low resource consumption.
   - **Resilient provider chain**: Providers are tried in a configurable order (`LLM_PROVIDER_ORDER`, default `groq,gemini`) and fall through to a deterministic mock composer if all fail. Groq is primary because the Gemini free tier caps generation at ~20 requests/day, which a 30-message batch or a full judge run exhausts almost immediately (the symptom was every real call returning HTTP 429 even though the key validated fine). All outbound requests send a browser-like `User-Agent`, without which Groq's Cloudflare edge rejects them with a 403 (error 1010). Rate-limit (429) responses are retried with backoff.
2. **Context-Driven Prompt Assembly**: Formats and sanitizes category voices, derived performance triggers, benchmarks, and active catalog offer contexts directly into LLM system prompts.
3. **Structured Pydantic State Store**: Persists ongoing conversation histories (`Turn` and `ConversationState`) to handle multi-turn sequences gracefully.
4. **Heuristics Safety Layer**:
   - **Auto-reply and Opt-out**: Programmatic regex and pattern check inside `/v1/reply` handles canned auto-replies and opt-out keywords (e.g. "stop spam") instantly without calling the LLM. Wait transitions (4h) and graceful exits (`ended`) are enforced programmatically.
   - **Post-LLM URL Scrubbing**: Regular expression filter scans generated text blocks to remove URLs before returning them, preventing Graph API template rejection and the judge's `-3` penalty.

---

## 2. Tradeoffs Made

- **Local Memory vs Database**: Used in-memory dicts to persist state. Given the 60-minute test constraints, this completely bypasses network database latency.
- **Direct HTTP vs LLM SDKs**: Wrote direct JSON HTTP requests to Gemini/Groq endpoints. This removes package dependency overhead (`google-generativeai`, `openai`) and avoids compatibility errors.
- **Rule-based Auto-replies**: We handle opt-outs and canned messages programmatically rather than letting the LLM classify them. This reduces API billing and guarantees 100% accurate classification.

---

## 3. What Additional Context Would Help Most

- **Real WhatsApp Webhook Payload Examples**: Knowing standard metadata properties (like exact phone structures or locale metadata) would help refine user language matching.
- **WABA Approved Templates**: Direct catalog schema for Kaleyra or Graph API template parameters would enable pre-compilation checks.
