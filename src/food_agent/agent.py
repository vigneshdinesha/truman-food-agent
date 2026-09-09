"""
Truman — tool-calling agent (v0).

Flow: user names a food (or an ingredient) -> the model decides which tool to
call (Open Food Facts for a food's ingredients, PubChem for a chemical's facts)
-> results are fed back -> the model produces a GROUNDED, structured verdict and
must abstain when the tools return no reliable data (never invent).

Observability: every run writes a JSON trace to traces/ (steps, tool calls,
tokens, estimated cost).

Run:  PYTHONPATH=src python -m food_agent.agent "mcdonalds big mac"

RAG note: v0 grounds on the live tools. The curated RAG corpus (rag/retriever.py)
plugs in as an additional tool once the index is built — same loop, one more tool.
"""
from __future__ import annotations
import json, time, uuid, sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from .config import AGENT_NAME, AGENT_TAGLINE, CHAT_MODEL, TRACES_DIR, require_openai_key
from .tools import openfoodfacts, pubchem
from .rag import retriever

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# gpt-4o-mini pricing (USD per token) for the cost trace
_PRICE_IN, _PRICE_OUT = 0.15 / 1e6, 0.60 / 1e6

SYSTEM = f"""You are {AGENT_NAME}, {AGENT_TAGLINE}.
You explain what is really in a food and what its ingredients mean for a
non-scientist — clearly, calmly, and HONESTLY. You are not alarmist and not a
salesperson; you give the balanced, evidence-based picture.

Rules:
- GROUND every claim in tool results. Do not rely on memory for ingredient facts.
- If the user names a food, call `lookup_food_by_name` to get its real ingredients first.
- For each notable additive/chemical, call `search_health_context` (the curated,
  sourced corpus) FIRST. If it reports coverage is "thin", THEN call `lookup_chemical`
  (PubChem) for that ingredient.
- If neither the corpus nor PubChem has reliable data on something, SAY SO
  ("I don't have reliable data on X") — never invent effects, numbers, or sources.
- Cite the sources you actually used (corpus source names and/or PubChem).
- End with a short verdict for the food overall: "clean", "worth knowing", or
  "concerning", plus a one-line why.
"""

TOOLS = [
    {"type": "function", "function": {
        "name": "lookup_food_by_name",
        "description": "Get a food product's real ingredient list from Open Food Facts by name.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "food/product name, e.g. 'Oreo' or 'McDonald's Big Mac'"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "search_health_context",
        "description": "Retrieve curated, sourced health context for an ingredient/additive from the vetted corpus. Use this FIRST for each additive.",
        "parameters": {"type": "object", "properties": {
            "ingredient": {"type": "string", "description": "ingredient/additive name, e.g. 'Red 40', 'TBHQ'"}},
            "required": ["ingredient"]}}},
    {"type": "function", "function": {
        "name": "lookup_chemical",
        "description": "Get facts about a chemical/additive from PubChem. Use only when search_health_context coverage is 'thin'.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "chemical or additive name, e.g. 'aspartame', 'Red 40'"}},
            "required": ["name"]}}},
]

DISPATCH = {
    "lookup_food_by_name": lambda a: openfoodfacts.lookup_by_name(a["query"]),
    "search_health_context": lambda a: retriever.search_health_context(a["ingredient"]),
    "lookup_chemical": lambda a: pubchem.lookup_chemical(a["name"]),
}

# ── Structured verdict (for the UI) ──────────────────────────────────────────
# A second, grounded pass that distils the analysis the agent JUST produced into a
# strict JSON verdict: a 0–100 cleanliness score, a tier, a conversational one-liner,
# and per-ingredient flags. It introduces NO new facts — it only structures what the
# tool-grounded analysis already established (so honesty/grounding is preserved).
VERDICT_SYSTEM = f"""You convert {AGENT_NAME}'s food analysis into a strict JSON verdict.
Use ONLY facts already stated in the analysis. Do NOT invent ingredients, effects, or sources.

Score the food 0–100 on how "clean" it is:
- 100 = whole, recognizable ingredients; no additives of concern.
- 0   = many additives that agencies flag or restrict.
Higher = cleaner. Map the score to a tier by these bands (be consistent):
  85–100 "Fully Clean" · 65–84 "Mostly Clean" · 45–64 "Some Concerns" ·
  25–44 "Questionable" · 0–24 "Highly Questionable".

Return ONLY this JSON object:
{{
  "food_name": "short display name of the food",
  "score": <int 0-100>,
  "tier": "<one of the five tier labels above>",
  "summary": "1–2 sentences, warm and direct — the honest bottom line a friend would give you. No hedging, no markdown.",
  "ingredients": [
    {{"name": "ingredient/additive", "level": "clean|caution|concern", "note": "<=8 words, plain English"}}
  ],
  "sources": ["source names actually cited in the analysis"]
}}
List the most notable ingredients first (concerns before clean ones), at most 8.
If the analysis abstained (no reliable data), set score 50, tier "Some Concerns",
and say so plainly in summary with an empty ingredients list."""


def _verdict(client, answer: str | None, trace: dict) -> dict | None:
    """Grounded second pass -> structured JSON verdict for the UI. Never throws."""
    if not answer or answer.startswith("(stopped"):
        return None
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL, temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": VERDICT_SYSTEM},
                      {"role": "user", "content": f"Analysis:\n\n{answer}\n\nReturn the JSON verdict now."}])
        u = resp.usage
        trace["tokens_in"] += u.prompt_tokens
        trace["tokens_out"] += u.completion_tokens
        v = json.loads(resp.choices[0].message.content)
        v["score"] = max(0, min(100, int(round(float(v.get("score", 50))))))
        if not isinstance(v.get("ingredients"), list):
            v["ingredients"] = []
        return v
    except Exception:
        return None


def analyze(user_input: str, max_steps: int = 6, verbose: bool = True) -> dict:
    require_openai_key()
    client = OpenAI()
    trace = {"id": str(uuid.uuid4())[:8], "input": user_input, "steps": [],
             "tokens_in": 0, "tokens_out": 0}
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_input}]

    for step in range(max_steps):
        resp = client.chat.completions.create(
            model=CHAT_MODEL, messages=messages, tools=TOOLS, temperature=0.2)
        u = resp.usage
        trace["tokens_in"] += u.prompt_tokens
        trace["tokens_out"] += u.completion_tokens
        msg = resp.choices[0].message

        if not msg.tool_calls:                      # model produced its final answer
            trace["answer"] = msg.content
            break

        messages.append(msg)                        # record the assistant's tool request
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            if verbose:
                print(f"  → tool: {tc.function.name}({args})")
            result = DISPATCH[tc.function.name](args)
            trace["steps"].append({"tool": tc.function.name, "args": args,
                                   "result_found": result.get("found")})
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(result)[:6000]})
    else:
        trace["answer"] = "(stopped: hit max reasoning steps)"

    trace["verdict"] = _verdict(client, trace.get("answer"), trace)
    trace["cost_usd"] = round(trace["tokens_in"] * _PRICE_IN + trace["tokens_out"] * _PRICE_OUT, 6)
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    (TRACES_DIR / f"{trace['id']}.json").write_text(json.dumps(trace, indent=2))
    return trace


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "nutella"
    print(f"\n=== {AGENT_NAME} analyzing: {q} ===\n")
    t = analyze(q)
    print("\n" + "─" * 70)
    print(t["answer"])
    print("─" * 70)
    print(f"[trace {t['id']}: {len(t['steps'])} tool calls · "
          f"{t['tokens_in']}+{t['tokens_out']} tokens · ${t['cost_usd']}]")
