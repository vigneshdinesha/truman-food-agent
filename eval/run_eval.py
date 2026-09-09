"""
Evaluation harness for Truman.

Runs every case in golden_set.jsonl through the agent, then grades each answer
with an LLM-as-judge on four dimensions:
  - grounded:        cites sources / no invented facts
  - correct_verdict: overall verdict matches the expected one (when specified)
  - abstained:       for unknown inputs, correctly refused to invent
  - mentions:        required key terms are present

Prints a scorecard and writes eval/scorecard.json. Designed to be run as a
regression gate (exits nonzero if the pass rate drops below THRESHOLD).

Run:  PYTHONPATH=src .venv/bin/python eval/run_eval.py
"""
from __future__ import annotations
import json, sys, os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from food_agent.agent import analyze          # noqa: E402
from food_agent.config import CHAT_MODEL       # noqa: E402

GOLDEN = ROOT / "eval" / "golden_set.jsonl"
SCORECARD = ROOT / "eval" / "scorecard.json"
THRESHOLD = 0.80    # regression gate: pass rate must be >= this

JUDGE_RUBRIC = """You are a strict grader for a food-ingredient assistant. Given a
test case and the assistant's answer, return JSON with boolean fields:
- "grounded": true if EITHER the answer cites sources (e.g. FDA, EFSA, PubChem, corpus)
  for its factual claims, OR it correctly abstains ("no reliable data") without
  inventing anything. An honest abstention counts as grounded (nothing to cite).
- "correct_verdict": true if the overall verdict is reasonable given expect_verdict
  (if expect_verdict is "any" or missing, set true).
- "abstained": true if the case requires abstention (must_abstain) AND the answer
  clearly says it lacks reliable data / does not invent facts. If the case does not
  require abstention, set true.
- "mentions": true if all strings in must_mention appear (case-insensitive) in the
  answer, else false. If must_mention is missing, set true.
Return ONLY the JSON object."""


def judge(client: OpenAI, case: dict, answer: str) -> dict:
    payload = {"case": case, "answer": answer}
    resp = client.chat.completions.create(
        model=CHAT_MODEL, temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": JUDGE_RUBRIC},
                  {"role": "user", "content": json.dumps(payload)}])
    return json.loads(resp.choices[0].message.content)


def main():
    client = OpenAI()
    cases = [json.loads(l) for l in GOLDEN.read_text().splitlines() if l.strip()]
    rows, dims = [], {"grounded": 0, "correct_verdict": 0, "abstained": 0, "mentions": 0}
    total_cost = 0.0

    for c in cases:
        t = analyze(c["query"], verbose=False)
        total_cost += t.get("cost_usd", 0)
        g = judge(client, c, t["answer"])
        passed = all(g.get(k, False) for k in dims)
        for k in dims:
            dims[k] += int(bool(g.get(k)))
        rows.append({"id": c["id"], "query": c["query"], "pass": passed, **g,
                     "tool_calls": len(t["steps"])})
        print(f"  {'PASS' if passed else 'FAIL'}  {c['id']:<18} "
              f"grounded={g.get('grounded')} verdict={g.get('correct_verdict')} "
              f"abstained={g.get('abstained')} mentions={g.get('mentions')}")

    n = len(cases)
    pass_rate = sum(r["pass"] for r in rows) / n
    card = {"n": n, "pass_rate": round(pass_rate, 3),
            "dimensions": {k: round(v / n, 3) for k, v in dims.items()},
            "eval_cost_usd": round(total_cost, 4), "rows": rows}
    SCORECARD.write_text(json.dumps(card, indent=2))

    print("\n" + "=" * 60)
    print(f"PASS RATE: {pass_rate:.0%}  ({sum(r['pass'] for r in rows)}/{n})")
    for k, v in card["dimensions"].items():
        print(f"  {k:<16} {v:.0%}")
    print(f"  eval cost: ${card['eval_cost_usd']}")
    print(f"Scorecard -> {SCORECARD}")

    if pass_rate < THRESHOLD:
        print(f"\n❌ REGRESSION: pass rate {pass_rate:.0%} < threshold {THRESHOLD:.0%}")
        sys.exit(1)
    print(f"\n✅ pass rate meets threshold ({THRESHOLD:.0%})")


if __name__ == "__main__":
    main()
