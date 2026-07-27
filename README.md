# magicpin AI Challenge — Vera Rebuilder (Vashu's team)

## 1. Approach Overview

### Architecture: Hybrid Heuristic + LLM Composition

The bot uses a two-layer architecture:

1. **Heuristic Layer (fast, deterministic)**: Handles auto-reply detection, hostile/opt-out routing, intent transitions, URL scrubbing, and suppression deduplication — all programmatically, without calling the LLM. This guarantees sub-100ms response times for the most common edge cases (auto-reply accounts for 40-70% of real merchant replies per the brief) and eliminates the risk of LLM timeout on critical-path decisions.

2. **LLM Composition Layer (rich, context-aware)**: For proactive messages and nuanced replies, the bot constructs trigger-specific prompts that include all 4 context layers, category voice profiles, and explicit compulsion lever instructions. The LLM is guided to use at least 2 of the 8 compulsion levers per message.

### Trigger-Kind Routing

Rather than using a single generic prompt for all messages, the bot routes each trigger kind to a **specialized prompt template** with framing instructions tailored to that conversation type:

| Trigger Kind | Framing Strategy | Primary Levers |
|---|---|---|
| `research_digest` | Peer-to-peer insight with source citation | Specificity, reciprocity |
| `perf_dip` | Fixable opportunity, not alarm | Loss aversion, social proof |
| `recall_due` | Warm patient reminder with slots | Specificity, effort externalization |
| `curious_ask_due` | Low-stakes question to the merchant | Asking the merchant, reciprocity |
| `festival_upcoming` | Time-bound campaign suggestion | Urgency, effort externalization |
| `dormant_with_vera` | Re-engage with a new data point | Curiosity, reciprocity |
| `active_planning_intent` | Deliver a draft artifact immediately | Effort externalization, specificity |

This routing matters because a compliance alert needs urgency framing while a curious-ask needs a question — the same generic prompt cannot optimize for both.

### Resilient Provider Chain

Providers are tried in order: **Groq (llama-3.3-70b)** → **Gemini (2.5-flash)** → **deterministic mock**. Groq is primary because the Gemini free tier caps at ~20 generations/day. All outbound API calls include a browser-like `User-Agent` header (without it, Groq's Cloudflare edge rejects with 403). Rate-limit (429) responses trigger exponential backoff retries.

---

## 2. Tradeoffs Made

| Decision | Rationale |
|---|---|
| **In-memory state vs database** | The 60-min test window means no persistence is needed across restarts. In-memory dicts eliminate network latency entirely. |
| **Direct HTTP vs LLM SDKs** | Zero external dependencies beyond FastAPI/uvicorn. Eliminates SDK version conflicts and reduces cold-start time on deployment platforms. |
| **Programmatic auto-reply detection** | Pattern matching is faster and more reliable than LLM classification for canned messages. Guarantees 100% accuracy on known patterns and stays well within the 30s deadline. |
| **Groq over Gemini as primary** | Groq's inference speed (sub-2s for 70B) fits the 30s tick deadline better. Gemini's free-tier quota exhausts during a single 30-message batch generation. |
| **Rich prompts over short prompts** | Longer prompts with explicit compulsion lever instructions and trigger-specific framing produce measurably better messages (higher specificity, category fit). The tradeoff is higher token consumption per call. |

---

## 3. Business Analysis: Strategic Observations

### Why current Vera underperforms on engagement

Based on the brief's production data (6,917 engaged merchants/day, 4.9 avg messages), Vera's core issue is **conversation quality, not reach**. The pain points map to specific composition failures:

1. **Auto-reply pollution (40-70% of replies)**: This is a routing problem, not a composition problem. Vera burns 2-3 turns per auto-reply because it doesn't recognize the canned pattern fast enough. Fix: programmatic detection on turn 1, wait immediately, don't waste an LLM call.

2. **Intent-handoff failures**: When a merchant says "let's do it", Vera goes back to qualifying. This is a state-machine failure — the bot doesn't track conversation phase (qualifying → committed → action). Fix: explicit intent patterns trigger immediate mode switch.

3. **Generic copy**: "10% off" doesn't work for Indian local merchants because they think in service+price ("Haircut @ ₹99"), not discount percentages. This is a category insight that should be baked into the prompt, not left for the LLM to figure out.

### Metrics I'd track in production

| Metric | Why it matters |
|---|---|
| **Reply rate by trigger kind** | Identifies which conversation types merchants actually engage with vs ignore |
| **Turn 2 drop-off rate** | Measures whether the opening message is compelling enough to get a reply |
| **Auto-reply detection accuracy** | False positives waste a real merchant's patience; false negatives waste bot turns |
| **Time-to-action (intent → confirmation)** | Measures how quickly the bot converts expressed interest into a completed action |
| **Suppression hit rate** | If suppression is too aggressive, we're leaving engagement on the table |
| **CTA type vs reply rate** | Determines whether binary YES/NO outperforms open-ended asks per category |

### A/B tests I'd run first

1. **Social proof vs loss aversion** (for `perf_dip` triggers): "3 dentists in Lajpat Nagar boosted CTR this week" vs "You're losing 45 potential walk-ins per month at current CTR". Hypothesis: social proof wins for high-performing merchants, loss aversion wins for underperformers.

2. **Asking the merchant vs telling** (for `curious_ask_due`): "What treatment is most asked-for this week?" vs "Your top-searched treatment this month was teeth whitening." Hypothesis: asking outperforms telling for engaged merchants (3+ prior Vera interactions), telling outperforms for dormant ones.

3. **Hinglish vs English** for merchants with `["en", "hi"]` language preference: Some merchants may prefer English for professional communications even if they speak Hindi. Track reply rates by language choice.

---

## 4. What Additional Context Would Help Most

- **Historical reply rates by trigger kind**: Knowing which trigger types merchants actually respond to would let me weight the trigger priority queue better during `/v1/tick`.
- **Real WhatsApp session window timestamps**: The 24h template window logic is approximated; real session state from the WABA API would enable precise template-vs-freeform decisions.
- **Merchant engagement segments**: A simple engaged/disengaged/new segmentation would let me adjust message aggressiveness (new merchants get softer CTAs, engaged merchants get direct asks).
- **Geographic/seasonal conversion data**: Knowing that "September-October converts 2x for gyms" (as referenced in the brief) as structured data per category per city would enable contrarian recommendations like the IPL case study.

---

## 5. Technical Details

- **Runtime**: Python 3.13 + FastAPI + uvicorn
- **LLM**: Groq (llama-3.3-70b-versatile) primary, Gemini (2.5-flash) fallback
- **State**: In-memory dicts (conversations, contexts, suppression keys)
- **Deployment**: Render (see `render.yaml`) or any platform supporting `Procfile`
- **Dependencies**: `fastapi`, `uvicorn`, `pydantic` — no LLM SDK dependencies
